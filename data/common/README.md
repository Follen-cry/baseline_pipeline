# `data/common/`

Cross-source processing code — things used by more than one entry in `data/sources/`, so they
don't belong inside any single source's own directory.

## `build_real_video_extra_settings.py`

Generic S1/S2 row builder. Per `../../PROVENANCE.md` §2/§7, this script — not the source-specific
builders — is the actual producer of 4 of the 20 cells in the S0-S3 merge table:
`epic_gen_S1`, `nwm_gen_S1`, `panda70m_gen_S1`, `panda70m_gen_S2`. None of
`sources/epic_kitchens/processing/build_epic_ssl.py`, `sources/nwm/processing/build_nwm_ssl.py`,
or `sources/panda70m_epic/processing/build_panda70m_epic_ssl.py` contain their own S1 (`mixed`
direction) or, for panda70m_epic, S2 (`--with-caption`) logic — this script fills that gap
generically across sources by reading each source's already-built anchor/FFS jsonl and
re-emitting it with a different direction/caption setting. `sources/panda70m_v1/` is the
exception — its own `build_panda70m_ssl.py` module handles all 3 settings natively (see that
source's README), so this script is never invoked for it.

**Don't** assume the 4 named source builders are self-sufficient for reproducing S0-S3 without
this script — it's load-bearing.

## Also relevant, but not duplicated here

`make_nwm_dual_ssl.py` — the dataset-agnostic S3 dual-MCQ converter reused unmodified by both
NWM and EPIC (`sources/nwm/`, `sources/epic_kitchens/`) — already lives inside the
`training/models/internvl-u` submodule at `internvl_chat/tools/make_nwm_dual_ssl.py` (that's
where the original codebase put it). Not copied here a second time to avoid two sources of
truth; both source builders import it from that path.
