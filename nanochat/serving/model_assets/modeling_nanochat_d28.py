"""nanochat d28 architecture for vLLM (v1 engine, vllm >= 0.31).

Implements the custom nanochat GPT (nanochat/nanochat/gpt.py) so
model_000177.pt (exported to safetensors) can be served with vLLM.

Architecture (d28: n_layer=28, n_embd=1792, n_head=14, n_kv_head=14, GQA):
- RMSNorm (no learnable weights) after token embedding and before lm_head
- rotary embeddings, base 1e5, half-dim rotation with NEGATIVE theta
  convention (nanochat gpt.apply_rotary_emb: y1 = x1*cos + x2*sin,
  y2 = -x1*sin + x2*cos)
- QK-norm (RMSNorm on q and k, no weights), then q and k each scaled by 1.2
  => effective logit scale 1.44 (passed to vLLM Attention as `scale`)
- value embeddings (ResFormer): on layers where layer_idx % 2 == (n_layer-1)%2
  (odd layers for n_layer=28), a second embedding lookup is gated into v by
  3*sigmoid(ve_gate(x[..., :12])) per kv-head
- smear gate: previous token's normalized embedding mixed into the current
  token's embedding: x += smear_lambda * sigmoid(smear_gate(x[..., :24])) * x_prev
- per-layer residual scalars: x = resid_lambdas[i]*x + x0_lambdas[i]*x0
  where x0 is the post-smear normalized embedding
- backout: after the trunk, x -= backout_lambda * x_mid, x_mid = residual at
  layer n_layer//2
- MLP: relu^2, 4x hidden
- sliding window attention, pattern SSSL tiled (short window = ceil(2048/4/128)*128
  = 768), final layer full context
- logit softcap 15 (15*tanh(logits/15)), vocab 32768 (no padding needed)
- pre-norm blocks: x = x + attn(norm(x)); x = x + mlp(norm(x))
"""
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor.layers.attention import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              extract_layer_index,
                                              make_layers, maybe_prefix)
from vllm.sequence import IntermediateTensors

try:
    from .configuration_nanochat import NanochatD28Config
except ImportError:  # loaded from the model dir by trust_remote_code
    from configuration_nanochat import NanochatD28Config


def rms_norm(x):
    return torch.rms_norm(x.float(), (x.size(-1),)).to(x.dtype)


def apply_rotary_emb(x, cos, sin):
    """nanochat convention: rotate by -theta (see gpt.py apply_rotary_emb).

    x: (num_tokens, n_heads, head_dim); cos/sin: (num_tokens, 1, head_dim/2)
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


class NanochatAttention(nn.Module):
    def __init__(self, config: NanochatD28Config, layer_idx: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.ve_gate_channels = config.ve_gate_channels
        tp_size = get_tensor_model_parallel_world_size()
        assert self.n_head % tp_size == 0
        assert self.n_kv_head % tp_size == 0
        self.n_local_heads = self.n_head // tp_size
        self.n_local_kv_heads = self.n_kv_head // tp_size
        self.q_size = self.n_local_heads * self.head_dim
        self.kv_size = self.n_local_kv_heads * self.head_dim
        self.ve = has_ve(layer_idx, config.n_layer)

        self.c_q = ColumnParallelLinear(
            self.n_embd, self.n_head * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.c_q")
        self.c_k = ColumnParallelLinear(
            self.n_embd, self.n_kv_head * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.c_k")
        self.c_v = ColumnParallelLinear(
            self.n_embd, self.n_kv_head * self.head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.c_v")
        self.c_proj = RowParallelLinear(
            self.n_embd, self.n_embd, bias=False, input_is_parallel=True,
            quant_config=quant_config, prefix=f"{prefix}.c_proj")
        if self.ve:
            self.ve_gate = ColumnParallelLinear(
                self.ve_gate_channels, self.n_kv_head, bias=False,
                quant_config=quant_config, prefix=f"{prefix}.ve_gate")

        # sliding window: SSSL tiled across layers, final layer full context.
        # Windows derive from the TRAINED context length, not the
        # YaRN-extended one: the model learned attention ranges at
        # 512/2048, and extending the context must not change them.
        # Backward compatible: without rope_scaling, trained == sequence_len.
        # For natively long-context-trained models (Stage 7), config carries
        # short_window explicitly (512) — use it when present.
        pattern = config.window_pattern.upper()
        rs = getattr(config, "rope_scaling", None)
        cfg_short = getattr(config, "short_window", None)
        if isinstance(rs, dict) and \
                str(rs.get("rope_type", "")).lower() in ("yarn", "ntk"):
            trained_len = int(rs.get("original_max_position_embeddings",
                                     config.sequence_len))
        else:
            trained_len = config.sequence_len
        # S layers keep the TRAINED short window (512): their attention
        # range is part of what the model learned. L layers are the
        # "full context" layers - they must cover the whole extended
        # context, so their window is the served sequence_len.
        long_window = config.sequence_len
        if cfg_short is not None:
            short_window = int(cfg_short)
        else:
            short_window = -(-trained_len // 4 // 128) * 128
        char = pattern[layer_idx % len(pattern)]
        window = long_window if char == "L" else short_window
        if layer_idx == config.n_layer - 1:
            window = long_window

        # nanochat scales q and k by 1.2 after QK-norm; FA3 then
        # applies its default 1/sqrt(head_dim). vLLM's Attention
        # scale REPLACES the default, so pass the full effective
        # scale: 1.44 / sqrt(head_dim).
        self.attn = Attention(
            self.n_local_heads, self.head_dim,
            config.qk_scale ** 2 / (self.head_dim ** 0.5),
            num_kv_heads=self.n_local_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=window,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x, ve, positions, cos_sin):
        q, _ = self.c_q(x)
        k, _ = self.c_k(x)
        v, _ = self.c_v(x)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        k = k.view(-1, self.n_local_kv_heads, self.head_dim)
        v = v.view(-1, self.n_local_kv_heads, self.head_dim)
        if ve is not None:
            ve = ve.view(-1, self.n_local_kv_heads, self.head_dim)
            gate, _ = self.ve_gate(x[..., :self.ve_gate_channels])
            gate = 3 * torch.sigmoid(gate)
            gate = gate.view(-1, self.n_local_kv_heads, 1)
            v = v + gate * ve
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)
        y = self.attn(q, k, v)
        y, _ = self.c_proj(y)
        return y


class NanochatMLP(nn.Module):
    def __init__(self, config: NanochatD28Config,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        hidden = config.mlp_hidden_mult * config.n_embd
        self.c_fc = ColumnParallelLinear(config.n_embd, hidden, bias=False,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.c_fc")
        self.c_proj = RowParallelLinear(hidden, config.n_embd, bias=False,
                                        input_is_parallel=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.c_proj")

    def forward(self, x):
        # Chunk over flattened tokens for very long prefills: the relu^2
        # intermediate at 107k+ tokens x 7168 x bf16 is ~1.5 GB per tensor,
        # and with the KV cache sized for 131k there is no headroom for the
        # full-length spike. Per-token math, so chunking is exact. (Same trick
        # as the Stage 7 training script.) vLLM passes flattened (num_tokens,
        # C); training-style (B, T, C) also handled.
        n_tokens = x.shape[0] if x.dim() == 2 else x.shape[1]
        if n_tokens > 32768:
            outs = []
            M = 16384
            for s in range(0, n_tokens, M):
                e = min(s + M, n_tokens)
                xc = x[s:e, :] if x.dim() == 2 else x[:, s:e]
                h, _ = self.c_fc(xc)
                h = torch.relu(h).square()
                o, _ = self.c_proj(h)
                outs.append(o)
            return torch.cat(outs, dim=0 if x.dim() == 2 else 1)
        x, _ = self.c_fc(x)
        x = torch.relu(x).square()
        x, _ = self.c_proj(x)
        return x


class NanochatBlock(nn.Module):
    def __init__(self, config: NanochatD28Config, layer_idx: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.attn = NanochatAttention(config, layer_idx, cache_config,
                                      quant_config, prefix=f"{prefix}.attn")
        self.mlp = NanochatMLP(config, quant_config, prefix=f"{prefix}.mlp")

    def forward(self, x, ve, positions, cos_sin):
        x = x + self.attn(rms_norm(x), ve, positions, cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x


class NanochatD28Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        self.n_layer = config.n_layer
        self.vocab_size = config.vocab_size
        tp_size = get_tensor_model_parallel_world_size()
        assert config.vocab_size % tp_size == 0

        self.wte = VocabParallelEmbedding(config.vocab_size, config.n_embd,
                                          org_num_embeddings=config.vocab_size)
        # value embeddings: kv_dim = head_dim * n_kv_head
        kv_dim = (config.n_embd // config.n_head) * config.n_kv_head
        self.value_embeds = nn.ModuleDict({
            str(i): VocabParallelEmbedding(config.vocab_size, kv_dim,
                                           org_num_embeddings=config.vocab_size)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.start_layer, self.end_layer, self.h = make_layers(
            config.n_layer,
            lambda prefix: NanochatBlock(config, extract_layer_index(prefix),
                                         cache_config, self.quant_config,
                                         prefix=prefix),
            prefix=f"{prefix}.h",
        )
        # scalars (loaded from checkpoint as weights)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.full((1,), 0.2))
        self.smear_gate = ColumnParallelLinear(24, 1, bias=False,
                                               quant_config=self.quant_config,
                                               prefix=f"{prefix}.smear_gate")
        self._cos = None
        self._sin = None
        self._cos_len = 0
        self._prev_emb_buf = None
        self._block_size = None
        self._cache_config = vllm_config.cache_config

    def _prev_embeddings(self, input_ids, positions, dtype):
        """Normalized embedding of the previous token of each input token.

        Uses the KV-cache slot mapping: slot(pos) = block_table[req][
        pos // block_size] * block_size + pos % block_size. A persistent
        buffer prev_norm_emb over all slots is written with the current
        tokens' normalized embeddings at their slots each forward, after
        reading the previous slots. Exact across steps, requests and
        chunked prefill.
        """
        if self._prev_emb_buf is None:
            cache_cfg = self._cache_config
            num_slots = (cache_cfg.num_gpu_blocks or 0) * cache_cfg.block_size
            self._prev_emb_buf = torch.zeros(
                max(num_slots, 1), self.config.n_embd,
                dtype=torch.bfloat16, device=input_ids.device)
            self._block_size = cache_cfg.block_size
        buf = self._prev_emb_buf

        meta = self._attn_metadata()
        if meta is None:
            return torch.zeros(input_ids.shape[0], self.config.n_embd,
                               dtype=dtype, device=input_ids.device)

        slot_mapping = meta.slot_mapping
        block_table = meta.block_table
        query_start_loc = meta.query_start_loc
        bs = self._block_size
        num_tokens = min(input_ids.shape[0], slot_mapping.shape[0],
                         meta.num_actual_tokens)

        # req index per token (from query_start_loc on cpu)
        qsl_list = query_start_loc.tolist()
        num_reqs = len(qsl_list) - 1
        req_list = []
        for r in range(num_reqs):
            req_list.extend([r] * (qsl_list[r + 1] - qsl_list[r]))
        while len(req_list) < num_tokens:
            req_list.append(0)
        req_idx = torch.tensor(req_list[:num_tokens], dtype=torch.long,
                               device=input_ids.device)

        pos = positions
        prev_pos = (pos - 1).clamp(min=0)
        bt = block_table.to(input_ids.device)[req_idx]  # (num_tokens, max_blocks)
        blk_idx = (prev_pos // bs).clamp(max=bt.shape[1] - 1)
        prev_block = bt.gather(1, blk_idx.unsqueeze(1)).squeeze(1)
        prev_slot = (prev_block * bs + prev_pos % bs).long()
        cur_slot = slot_mapping[:num_tokens].long().to(input_ids.device)
        valid = cur_slot >= 0
        need = max(int(prev_slot.max().item()),
                    int(cur_slot[valid].max().item() if valid.any() else 0)) + 1
        if need > buf.shape[0]:
            grown = torch.zeros(need, self.config.n_embd,
                                dtype=buf.dtype, device=buf.device)
            grown[:buf.shape[0]] = buf
            self._prev_emb_buf = buf = grown
        # write current tokens' normalized embeddings into their slots
        # FIRST: within a prefill, token i reads slot(i-1), which
        # is written by this same forward, matching training-time
        # smear (x[:, 1:] + gate * x[:, :-1]). Decode and
        # chunked-prefill reads of earlier forwards' slots stay
        # correct. Position-0 tokens read their own slot but the
        # gate is zeroed at pos 0.
        # Chunked: rms_norm in fp32 over 107k+ tokens is a 0.77 GB fp32
        # copy per tensor; with the KV cache sized for 131k there is no
        # headroom, so compute + write in 16384-token slices.
        M = 16384
        for s in range(0, num_tokens, M):
            e = min(s + M, num_tokens)
            cur_norm = rms_norm(self.wte(input_ids[s:e]))
            buf[cur_slot[s:e]] = cur_norm.to(buf.dtype)
        prev_x = buf[prev_slot]
        if num_tokens < input_ids.shape[0]:
            pad = torch.zeros(input_ids.shape[0] - num_tokens,
                              self.config.n_embd, dtype=prev_x.dtype,
                              device=prev_x.device)
            prev_x = torch.cat([prev_x, pad], dim=0)
        return prev_x.to(dtype)

    def _attn_metadata(self):
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        am = ctx.attn_metadata
        if isinstance(am, list):
            am = am[0] if am else None
        if isinstance(am, dict):
            for v in am.values():
                if v is not None:
                    return v
            return None
        return am

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def _rotary(self, positions, dtype):
        # cos/sin of shape (num_tokens, 1, head_dim//2), nanochat convention.
        # Optional YaRN extension (config.rope_scaling.rope_type == "yarn"):
        # inv_freq interpolation per HF _compute_yarn_parameters, cos/sin
        # scaled by the YaRN mscale attention factor (factor on cos/sin only;
        # NOT folded into the attention scale, which would double-apply).
        # Without rope_scaling this is the original nanochat rotary.
        max_pos = int(positions.max().item()) + 1
        if self._cos is None or self._cos_len < max_pos:
            base = self.config.rotary_base
            head_dim = self.config.n_embd // self.config.n_head
            channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32,
                                         device=positions.device)
            inv_freq = 1.0 / (base ** (channel_range / head_dim))
            attn_factor = 1.0
            rs = getattr(self.config, "rope_scaling", None)
            if isinstance(rs, dict) and \
                    str(rs.get("rope_type", "")).lower() == "ntk":
                # NTK-aware: raise the base so high-frequency channels keep
                # their trained wavelengths at short positions while the
                # effective context stretches. Short-context behavior is
                # nearly unchanged (unlike YaRN interpolation).
                factor = float(rs["factor"])
                head_dim2 = head_dim - 2
                base = base * (factor ** (head_dim / head_dim2))
                inv_freq = 1.0 / (base ** (channel_range / head_dim))
            if isinstance(rs, dict) and \
                    str(rs.get("rope_type", "")).lower() == "yarn":
                import math
                factor = float(rs["factor"])
                orig_len = int(rs.get("original_max_position_embeddings",
                                      2048))
                beta_fast = float(rs.get("beta_fast", 32))
                beta_slow = float(rs.get("beta_slow", 1))
                truncate = bool(rs.get("truncate", True))
                # HF _compute_yarn_parameters
                pos_freqs = base ** (channel_range / head_dim)
                inv_freq_extrapolation = 1.0 / pos_freqs
                inv_freq_interpolation = 1.0 / (factor * pos_freqs)

                def find_correction_dim(num_rotations, dim, base, max_pos):
                    return (dim * math.log(max_pos /
                            (num_rotations * 2 * math.pi))) / \
                           (2 * math.log(base))

                # NOTE: HF passes the full head_dim here (not head_dim//2)
                low = find_correction_dim(beta_fast, head_dim, base,
                                          orig_len)
                high = find_correction_dim(beta_slow, head_dim, base,
                                           orig_len)
                if truncate:
                    low = math.floor(low)
                    high = math.ceil(high)
                low = max(low, 0)
                high = min(high, head_dim - 1)

                def linear_ramp_factor(min_, max_, dim):
                    if min_ == max_:
                        max_ = max_ + 0.001
                    lin = (torch.arange(dim, dtype=torch.float32,
                                        device=positions.device) - min_) / \
                          (max_ - min_)
                    return torch.clamp(lin, 0, 1)

                extr_factor = 1.0 - linear_ramp_factor(low, high,
                                                       head_dim // 2)
                inv_freq = (inv_freq_interpolation *
                            (1.0 - extr_factor) +
                            inv_freq_extrapolation * extr_factor)
                # YaRN mscale (HF get_mscale, mscale=mscale_all_dim=1)
                attn_factor = 0.1 * math.log(factor) + 1.0 if factor > 1 \
                    else 1.0
            t = torch.arange(max_pos, dtype=torch.float32,
                             device=positions.device)
            freqs = torch.outer(t, inv_freq)
            self._cos = (freqs.cos() * attn_factor).to(dtype)[:, None, :]
            self._sin = (freqs.sin() * attn_factor).to(dtype)[:, None, :]
            self._cos_len = max_pos
        # Index by position so each token gets its own cos/sin row. For a
        # single contiguous prefill (positions == arange(n)) this is
        # identical to returning the whole table; for mixed batches
        # (prefill + decode of other requests) it is the correct semantics.
        cos = self._cos.to(dtype)[positions]
        sin = self._sin.to(dtype)[positions]
        return cos, sin

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                x = inputs_embeds
            else:
                x = self.wte(input_ids)
        else:
            x = intermediate_tensors["hidden_states"]
        x = rms_norm(x)

        # smear: mix the previous token's normalized embedding into x.
        # The previous token's normalized embedding is looked up from a
        # persistent slot-keyed buffer (the KV-cache slot of position p-1),
        # which is exact for prefill, decode and chunked prefill, any batch
        # size. Position-0 tokens get a zeroed gate.
        prev_x = self._prev_embeddings(input_ids, positions, x.dtype)
        gate, _ = self.smear_gate(x[..., :24])
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(gate)
        gate = torch.where((positions == 0).unsqueeze(-1),
                           torch.zeros_like(gate), gate)
        x = x + gate * prev_x

        x0 = x
        backout_layer = self.n_layer // 2
        x_backout = None
        cos_sin = self._rotary(positions, x.dtype)
        for i in range(self.start_layer, self.end_layer):
            block = self.h[i - self.start_layer]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = (self.value_embeds[str(i)](input_ids).to(x.dtype)
                  if str(i) in self.value_embeds else None)
            x = block(x, ve, positions, cos_sin)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = rms_norm(x)
        return x


class NanochatD28ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.model = NanochatD28Model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size, config.n_embd,
                                          quant_config=self.quant_config,
                                          prefix=maybe_prefix(prefix, "lm_head"))
            self.logits_processor = LogitsProcessor(
                config.vocab_size, soft_cap=config.logit_softcap)
        else:
            self.lm_head = None
        self.make_empty_intermediate_tensors = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)

    def compute_logits(self, hidden_states):
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # remap native nanochat checkpoint names to the vLLM module tree:
        #   transformer.wte.weight        -> model.wte.weight
        #   transformer.h.{i}.*           -> model.h.{i}.*
        #   value_embeds.{i}.weight       -> model.value_embeds.{i}.weight
        #   smear_gate.weight             -> model.smear_gate.weight
        #   lm_head.weight                -> lm_head.weight
        #   resid_lambdas / x0_lambdas / smear_lambda / backout_lambda
        #                                 -> model.{name}
        def remap(names):
            for name, tensor in weights:
                if name.startswith("transformer.h."):
                    name = "model.h." + name[len("transformer.h."):]
                elif name.startswith("transformer.wte."):
                    name = "model.wte." + name[len("transformer.wte."):]
                elif name.startswith("value_embeds."):
                    name = "model." + name
                elif name.startswith("smear_gate."):
                    name = "model." + name
                elif name in ("resid_lambdas", "x0_lambdas", "smear_lambda",
                              "backout_lambda"):
                    name = "model." + name
                yield name, tensor

        loader = AutoWeightsLoader(self)
        return loader.load_weights(remap(weights))
