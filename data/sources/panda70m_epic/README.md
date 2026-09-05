# `data/sources/panda70m_epic/`

Full detail: `../../../PROVENANCE.md` §7. Summary:

- **Raw data**: clips downloaded via `yt-dlp` to node-pinned local SSD
  (`/scratch/local/ssd/junlin/data/datasets/panda70m_epic_ssl/clips`, on `torrnode11`);
  extracted frames on network storage (`/scratch/network/ssd/junlin/panda70m_epic_ssl_frames`,
  1.7G). Source pool: Panda-70M clips regex + VLM-judge-filtered to kitchen/manipulation
  content. Neither migrated in this pass.
- **Honesty caveat carried from the script itself**: despite the "epic" name, this is **not**
  actually egocentric footage — only ~25-30% of clips have hands/object-dominant framing per a
  manual spot check. Don't let prompts/docs downstream imply "first-person."
- **Processing**: `processing/build_panda70m_epic_ssl.py`, stages `download → anchors → gen →
  dual` — no standalone `splits`/`ffs` stage, unlike epic_kitchens/nwm.
- **Train/eval split**: clip-level, decided inside `build_anchors()` itself — a **fraction**
  (15% of eligible clips) rather than a fixed count, the only one of the 4 sources split this
  way. `SPLIT_SEED=42`.
- **Both S1 and S2 rows** (`panda70m_gen_S1`, `panda70m_gen_S2`) are built by
  `../../common/build_real_video_extra_settings.py` — this source has no native action/caption
  gen variant of its own, unlike epic_kitchens/nwm.
