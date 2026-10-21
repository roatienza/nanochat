"""Config for the nanochat d28 architecture."""
from transformers import PretrainedConfig


class NanochatD28Config(PretrainedConfig):
    model_type = "nanochat_d28"

    def __init__(self,
                 sequence_len: int = 2048,
                 vocab_size: int = 32768,
                 n_layer: int = 28,
                 n_head: int = 14,
                 n_kv_head: int = 14,
                 n_embd: int = 1792,
                 window_pattern: str = "SSSL",
                 rotary_base: int = 100000,
                 qk_scale: float = 1.2,
                 logit_softcap: float = 15.0,
                 mlp_hidden_mult: int = 4,
                 ve_gate_channels: int = 12,
                 tie_word_embeddings: bool = False,
                 **kwargs):
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern
        self.rotary_base = rotary_base
        self.qk_scale = qk_scale
        self.logit_softcap = logit_softcap
        self.mlp_hidden_mult = mlp_hidden_mult
        self.ve_gate_channels = ve_gate_channels
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.head_dim = n_embd // n_head
        self.max_position_embeddings = sequence_len
        super().__init__(**kwargs)