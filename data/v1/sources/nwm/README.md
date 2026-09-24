# `data/v1/sources/nwm/`

Full detail: `../../../PROVENANCE.md` §7. Summary:

- **Raw data**: RECON trajectory images referenced in place on network storage
  (`/scratch/network/ssd/junlin/nwm_data/recon`, 26G — never copied by the builder itself).
  Trajectory train/test name lists come from a sibling repo,
  `Evaluation/nwm/data_splits/recon/{train,test}/traj_names.txt` — imported via a deliberate
  one-off `sys.path` insert (flagged in the original script's own docstring as a narrow
  exception to the usual dependency direction). Neither migrated in this pass.
- **Processing**: `processing/build_nwm_ssl.py`, stages `splits → anchors → ffs → gen → dual`
  (+ an `edit` stage, not used by S0-S3).
- **Train/eval split**: trajectory-level, only ever drawn from RECON's official *train* list
  (zero overlap with RECON's official test list, reserved for the separate NWM eval benchmark),
  `SPLIT_SEED=42`, 500 trajectories held out whole.
- **S1 rows** (`nwm_gen_S1`) are built by `../../common/build_real_video_extra_settings.py`, not
  this script.
- **Known leak caveat** (documented in the script itself): consecutive sliding windows mean
  ~89% of `nwm_dual_*` rows have `cond_image` pixel-duplicating a wrong MCQ *option* (never the
  ground-truth answer) — carry this caveat forward into any future eval-integrity work.
