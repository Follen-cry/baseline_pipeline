# `data/sources/panda70m_v1/`

Full detail: `../../../PROVENANCE.md` §7. Summary:

- **Raw data**: COCO/3DSRBench-filtered Panda-70M clips, node-pinned local SSD
  (`/scratch/local/ssd/junlin/data/datasets/panda70m/3dsrbench_sample5000`, on `torrnode11`,
  video2dataset sharded format, 1,073 clips); extracted frames on network storage
  (`/scratch/network/ssd/junlin/panda70m_ssl_v1_frames`, 1.7G). Neither migrated in this pass.
- **Processing**: `processing/build_panda70m_ssl_v1_driver.py` (thin config-override CLI
  wrapper) → `processing/build_panda70m_ssl.py` (the shared module, monkey-patched via its own
  module-level path constants — this is intentional, not a packaging mistake). Stages
  `splits → anchors → ffs → dual → gen` — note `gen` comes **last** here, unlike the other 3
  real-world sources.
- **Train/eval split**: clip-level, `SPLIT_SEED=42`, 130 of ~1,068 valid clips (~12%) held out
  whole. Per the module's own comment: S0/S1/S2 (`gen`) take **no** eval holdout at all — the
  held-out clips exist solely to populate S3's MCQ eval pool.
- Unlike `panda70m_epic`, this source's own `build_panda70m_ssl.py` handles all 3 gen settings
  natively (`--direction`/`--with-caption` flags) — `build_real_video_extra_settings.py` is
  never invoked for this source.
- **Naming inconsistency to be aware of**: its S3 output is named `panda3dsr_dual_noaction`, not
  `panda70m_v1_dual_*` like the S0/S1/S2 outputs (`panda70m_v1_gen_S{0,1,2}`) — an artifact
  carried from an older pipeline naming convention. Don't assume the S3 pair follows the same
  naming pattern as S0-S2 when writing tooling against this source.
