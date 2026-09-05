# `data/sources/epic_kitchens/`

Full detail: `../../../PROVENANCE.md` §7. Summary:

- **Raw data**: 116 EPIC-KITCHENS-100 validation-split MP4s, node-pinned on `torrnode11`
  (`/scratch/local/ssd/junlin/data/worldprediction/epic-kitchen/`). Extracted frame JPGs
  (what the training jsonls actually reference) live on network storage
  (`/scratch/network/ssd/junlin/epic_ssl_frames/`, 22G). Neither migrated in this pass.
- **Processing**: `processing/build_epic_ssl.py`, stages `splits → anchors → ffs → gen → dual`.
  Only `anchors` touches raw video (must run on torrnode11); the rest are portable.
- **Train/eval split**: video-level, seeded (`SPLIT_SEED=42`), 20 of 116 videos held out whole
  before any row-level sampling — eval rows never share a source video with train rows.
- **S1 rows** (`epic_gen_S1`) are built by `../../common/build_real_video_extra_settings.py`,
  not this script — see that file's README.
- This is the only one of the 4 real-world sources with genuine human-written narration
  (matched from an EPIC annotation CSV), used as the caption text in its S2/action rows.
