"""Minimal shim so vendored vlmevalkit/vlmeval/vlm/internvlu.py's
`from paths import add_internvlu_to_path` resolves.

That file walks 3 directories up from its own location (vlmeval/vlm/) looking
for a `paths.py` -- in the old ssl_mllm tree that landed on `Evaluation/
paths.py` (the real, fuller version, with data()/results() helpers too); in
this repo's layout (`eval/inference/vlmevalkit/vlmeval/vlm/`) it resolves
here instead. Only `add_internvlu_to_path()` is needed by vlmevalkit.

Same "known dependency gap" as suites/magicbrush, suites/risebench,
suites/aurorabench: the `internvlu` package (InternVLUPipeline) lives at
Model_Related/InternVLU/InternVL-U in the old ssl_mllm tree, not this repo's
`training/models/internvl-u` submodule. Hardcodes the same absolute path
those suites' inference scripts use, for consistency -- not fixed generically
here either.
"""
import os
import sys

_SSL_ROOT = "/scratch/network/ssd2/junlin/ssl_mllm"
INTERNVLU_PKG = os.path.join(_SSL_ROOT, "Model_Related", "InternVLU", "InternVL-U")
INTERNVL_CHAT = os.path.join(_SSL_ROOT, "Model_Related", "InternVLU", "InternVL", "internvl_chat")
SENSENOVA_SRC = os.path.join(_SSL_ROOT, "Model_Related", "SenseNova-U1", "src")
SENSENOVA_LOADER = os.path.join(_SSL_ROOT, "Model_Related", "SenseNova-U1", "model_loader")


def _add(p):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def add_internvlu_to_path():
    """Make `from internvlu import InternVLUPipeline` importable."""
    _add(INTERNVL_CHAT)
    _add(INTERNVLU_PKG)


def add_sensenova_to_path():
    """Make `sensenova_u1` model + `snu1_lora` loader importable.

    Only present so `vlmeval/vlm/sensenova_u1.py`'s module-level import
    resolves (vlmeval/vlm/__init__.py imports every model wrapper
    unconditionally) -- this repo's suites don't use SenseNova-U1.
    """
    _add(SENSENOVA_SRC)
    _add(SENSENOVA_LOADER)
