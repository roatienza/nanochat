"""
Verify a PACKED model dir: the weights, the config, and the aux files.

Stage 10 converts a nanochat .pt into a HuggingFace-style directory that
vLLM loads with --trust-remote-code. Two failures are invisible to a file
listing:

  1. the weights in model.safetensors do not match the model's shape
     (a silent mis-export);
  2. config.json does not match the weights (a silent mis-serve: the
     right weights with the wrong architecture, which vLLM would either
     reject or, worse, load with the wrong shapes).

This checks both, plus the files vLLM needs, by loading the safetensors
on the CPU -- no GPU needed, so it can run while the pipeline is using
the cards.

Usage:
  python -m pytest tests/test_packed_model.py -- --dir /path/to/model
  python tests/test_packed_model.py --dir /path/to/model
"""
import argparse
import json
import os

import pytest

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None

# d28: n_layer 28, n_embd 1792, n_head 14, n_kv_head 14, head_dim 128.
EXPECT = dict(
    n_layer=28, n_embd=1792, n_head=14, n_kv_head=14, vocab_size=32768,
    sequence_len=131072, max_position_embeddings=131072,
    model_type="nanochat_d28",
)
# measured with the real GPT class (nanochat/param_audit.py)
EXPECT_PARAMS = 2_018_511_234
# the value embeddings live on every other layer of 28 -> 14 tables
EXPECT_VE_TABLES = 14

AUX_FILES = (
    "model.safetensors", "config.json", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja",
    "modeling_nanochat_d28.py", "configuration_nanochat.py",
    "generation_config.json",
)


def _dir():
    # pytest passes extra args after "--"
    argv = [a for a in os.sys.argv if a.startswith("--dir=")]
    if argv:
        return argv[0].split("=", 1)[1]
    if "--dir" in os.sys.argv:
        i = os.sys.argv.index("--dir")
        if i + 1 < len(os.sys.argv):
            return os.sys.argv[i + 1]
    return os.environ.get("PREFLIGHT_MODEL_DIR", "/tmp/utak-preflight/model")


MODEL_DIR = _dir()


def _skip_if_absent():
    if not os.path.isdir(MODEL_DIR):
        pytest.skip(f"no packed model dir at {MODEL_DIR}")
    if safe_open is None:
        pytest.skip("safetensors is not installed")


def _config():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        return json.load(f)


def _shapes():
    out = {}
    st = os.path.join(MODEL_DIR, "model.safetensors")
    with safe_open(st, framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = tuple(f.get_slice(k).get_shape())
    return out


def test_the_config_matches_the_architecture():
    _skip_if_absent()
    cfg = _config()
    for k, v in EXPECT.items():
        assert cfg.get(k) == v, f"config.json {k} is {cfg.get(k)!r}, expected {v!r}"


def test_the_weights_have_the_right_parameter_count():
    _skip_if_absent()
    total = 0
    for shape in _shapes().values():
        n = 1
        for s in shape:
            n *= s
        total += n
    assert total == EXPECT_PARAMS, (
        f"the safetensors hold {total:,} parameters, expected "
        f"{EXPECT_PARAMS:,} (a difference of {total - EXPECT_PARAMS:,})"
    )


def test_the_key_shapes_are_right():
    _skip_if_absent()
    sh = _shapes()
    def shape_of(prefix):
        hits = [k for k in sh if k.startswith(prefix)]
        assert hits, f"no tensor starts with {prefix!r}"
        return sh[hits[0]]
    assert shape_of("transformer.wte.") == (32768, 1792)
    assert shape_of("lm_head.") == (32768, 1792)
    assert shape_of("value_embeds.1.") == (32768, 1792)
    assert shape_of("transformer.h.0.attn.c_q.") == (1792, 1792)
    assert shape_of("transformer.h.0.attn.c_k.") == (1792, 1792)
    assert shape_of("transformer.h.0.mlp.c_fc.") == (7168, 1792)


def test_all_the_value_embedding_tables_are_there():
    _skip_if_absent()
    ve = [k for k in _shapes() if k.startswith("value_embeds")]
    assert len(ve) == EXPECT_VE_TABLES, (
        f"{len(ve)} value embedding tables, expected {EXPECT_VE_TABLES} "
        f"(every other layer of 28)"
    )


def test_every_aux_file_is_present():
    _skip_if_absent()
    for f in AUX_FILES:
        p = os.path.join(MODEL_DIR, f)
        assert os.path.exists(p), f"missing {f} in the packed model dir"
        assert os.path.getsize(p) > 0, f"{f} is empty"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=MODEL_DIR)
    a = ap.parse_args()
    MODEL_DIR = a.dir
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
