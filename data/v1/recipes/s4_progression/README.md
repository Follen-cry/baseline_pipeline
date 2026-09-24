# `s4_progression/`

S4 = S3 with the MCQ dropped, replaced by a progression-description +
next-frame text prediction, trained jointly with the same image-generation
prediction S0-S3 already use. Design/decision log below; code in this dir
and in each source's `build_s4_caption_prefix.py` implements exactly this.

## What changed vs. S3, and why

S3 (`../s0s3_baseline/`) = 3 context frames + a 4-way MCQ (1 true + 3
distractor candidate frames), with the GPT turn supervising only a single
answer letter (`LM_LOSS_WEIGHT=0.5`) jointly with the image-generation loss.

S4 drops the MCQ entirely. In its place, the GPT turn's text now carries
real content: a description of how the scene changes across the 3 context
frames, followed by a separate textual prediction of what the 4th
(target) frame will look like -- **in addition to**, not instead of, the
existing image-generation prediction. All three are supervised jointly:
progression-description text, next-frame text prediction, and next-frame
pixels.

This was checked against the actual model code
(`Model_Related/InternVLU/InternVL/internvl_chat/internvl/model/internvlu/modeling_internvlu_unified.py`)
before committing to the design: the generation head's conditioning span is
"every token from the turn's second `<|im_start|>` through the `<img>`
trigger token, inclusive" of the LLM's causal hidden states, with no
`.detach()`/`no_grad()` anywhere on that path. So whatever text precedes
`<img>` in the same GPT turn -- the progression description, the next-frame
text prediction -- causally conditions the generation head's output, not
just an independent, decorrelated text loss. That's what makes "describe,
then predict, then generate" a real reasoning-then-generate chain rather
than three unrelated heads bolted together.

## Base dataset: S2's full window set, not S3's

S4 is built from `../../data/v1/datasets/final_s0s3/S2_train.jsonl` (49,113
rows), **not** `S3_train.jsonl` (47,770 rows), even though S3 is "closer" in
spirit (it's the setting S4 replaces). Traced why S3 has fewer rows before
deciding this (2026-09-06):

- **VBVR: S3 has 24,000 windows vs. S2's 25,000.** Not a random per-row
  drop -- `vbvr_next_frame_make_dual.py` requires >=4 usable windows per
  VBVR task to build even one MCQ row (1 anchor + 3 distractor donors);
  any task under that threshold is dropped **entirely** from S3, while
  still present in S2 (which needs no distractors). Task-level exclusion,
  not row-level.
- **panda70m_epic: S3 drops ~8.2% of rows to the shared missing-file filter
  (`filter_final_s0s3.py`) vs. S2's ~2.2%.** Both settings share the same
  underlying ~2.7%-corrupt-frame problem (cv2 seek failures near clip
  boundaries), but S3 rows reference 3 extra MCQ-distractor image paths
  per row on top of the 3 context + cond + target images S2 needs -- more
  images referenced per row means more chances of hitting a known-bad
  frame, even when the row's own anchor window is intact.
- **epic, nwm, panda70m_v1: S2 and S3 are exactly row-for-row identical**
  (same counts pre- and post-filter).

Since S4 needs no distractors, neither exclusion applies to it. Building
from S2 recovers the ~1,343 rows (1,000 VBVR windows + ~343 panda70m_epic
rows) that were only ever dropped for MCQ-specific reasons, while staying
exactly aligned with the S0/S1/S2 window pool (confirmed identical
per-source row counts across S0/S1/S2 before this work started).

Confirmed 2026-09-06: all 5 sources use exactly 3 context frames for
S0-S2 (`CONTEXT_SIZE = 3` in each source's own build script; VBVR has no
such constant but its S2 rows all carry exactly 3 `image` paths, verified
directly against `S2_train.jsonl`).

## Human turn: S2's caption kept, closing replaced

Every source's S2 human turn is two segments: (1) a task-specific
caption/description, (2) a generic "given these frames, do X" closing.
S4 keeps segment (1) untouched per-source and replaces segment (2) with one
instruction shared across all 5 sources (`s4_progression_prompt.py`'s
`PROGRESSION_INSTRUCTION_TEMPLATE`):

> You are shown three consecutive frames, in temporal order. First, in 1-2
> sentences (no more than about 40 words), describe how the scene changes
> across them, based only on what is visible in these three frames. Then,
> prefaced with "Next frame prediction:", predict in 1-2 sentences (no more
> than about 40 words) what the next frame will look like.

Design choices made along the way, and why:

- **No explicit format delimiters** (e.g. dashes) between the two
  instruction parts in the human turn -- natural "First... Then..." phrasing
  was judged less likely to produce templated/robotic VLM captions than a
  rigid dash/field format, and there's a good machine-parseable anchor
  available anyway (see GPT turn below).
- **Word budget stated explicitly** ("no more than about 40 words") rather
  than a vague "concisely" -- keeps caption length (and thus LM-loss signal
  density and generation-conditioning length) consistent across ~49k rows
  from 5 very different sources.
- **`concat = <image>*3 + caption + "\n\n" + instruction`**, no image tag at
  the end -- matches the existing convention across all 5 sources'
  S0-S3 human turns exactly (verified: all `<image>` tags lead, followed by
  caption then closing, nothing after). `cond_image`/`target_image` stay as
  separate structured jsonl fields, never rendered as extra inline tags.
- **NWM is the one exception**: its S2 caption jumps straight into "Action
  to execute from the last frame: ..." with no scene-setting sentence
  (unlike EPIC's "These are N consecutive first-person camera frames from
  an egocentric kitchen video..." or panda70m's "Video description: ...").
  S4 prepends a fixed, non-VLM sentence for NWM only:
  `"These are {n} consecutive first-person camera frames from an egocentric
  navigation video sequence."` (`s4_progression_prompt.nwm_domain_prefix`).
  This also sidesteps a pre-existing bug found while doing this: NWM's S2
  prompt hardcodes the word "four" ("Observe these four frames...") even
  though `CONTEXT_SIZE` was shrunk 4->3 on 2026-08-31 and never updated --
  S4's new prefix uses the real context length instead. **S0-S3's existing
  data/code is untouched**; this fix only applies to newly-generated S4 text.

All caption-prefix recovery was verified byte-for-byte against the real,
already-rendered S2 text before this design was implemented (see
"Verification" below) -- none of it is re-derived by guessing at the
original builder scripts' logic from memory.

## GPT turn: progression + text prediction + generation trigger

```
{progression description} Next frame prediction: {frame-4 text description}. The next frame should look like this: <img>
```

- `"Next frame prediction:"` is a deliberate, lightweight parse anchor
  (matches this codebase's existing "Label: content" convention --
  "Video description:", "Action to execute from the last frame:" -- rather
  than an artificial dash-delimited format) so that, if eval ever needs to
  score progression-description quality, next-frame-text quality, and
  generated-image quality separately, the three components can be split
  out of the raw model output reliably.
- `"The next frame should look like this: <img>"` is the unchanged S0/S1
  `GEN_GPT_ANSWER` trigger phrase/token -- kept identical for compatibility
  with the existing `<img>`-token-position logic in the modeling code.

## Ground-truth labeling (DONE 2026-09-06 -- see final numbers below)

Both text segments come from an external VLM given **4** images per row --
the 3 context frames **and** the true target (4th) frame, labeled Frame 1-4
-- plus the row's own task-specific caption as context. Frame 4 access is
intentional ("privileged" at labeling time only, never seen by the trained
model): letting the VLM verify against the true continuation, rather than
guessing blind from 3 possibly-ambiguous frames, keeps the progression text
and the actual generation target mutually consistent (otherwise, on
real-world footage with subtle/occluded motion, a blind 3-frame guess could
describe a direction inconsistent with the correct target image, teaching
the model two contradictory things from the same row).

**Update 2026-09-06: VLM chosen and full labeling run launched.**
Model: `Qwen3-VL-30B-A3B-Instruct-FP8` (local checkpoint at
`/scratch/local/ssd/junlin/models/Qwen3-VL-30B-A3B-Instruct-FP8`), served via
vLLM 0.11.2 (`conda env vllm`). Quality-checked first on 2 samples per
task/source (5 VBVR tasks + 4 real-world sources, 18 rows total, via
`label_with_vlm.py --print-only`) before committing to the full run:
real-world sources (EPIC/NWM/panda70m) produced well-grounded, plausible
descriptions; VBVR's abstract tasks got the general category of change right
but not always exact rotation angle/direction on `rotation_puzzle`
specifically (spot-checked against the actual frame/target images) — this
is the previously-flagged, already-accepted VLM-vs-programmatic-caption
tradeoff for VBVR, not a new problem, and not something a different
general-purpose VLM would likely fix, so the primary model choice was kept
rather than falling back.

Infra note for reproducing/resuming: `vllm serve` crashed under
CUDA OOM when the client sent 16 concurrent 4-image requests with no
server-side cap (default `--max-num-seqs`) — fixed by adding
`--max-num-seqs 4` (hard server-side concurrency cap, independent of
client worker count) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
At that concurrency, one A40 does ~0.65 rows/sec (~21h for all 49,113 rows),
so the full run is sharded 3-way across 3 separate vLLM instances on 3
different GPUs (ports 8123/8124/8125), each fed one third of
`S4_captioning_manifest.jsonl` (`manifest_shard{0,1,2}.jsonl`) via
`label_with_vlm.py`, writing to `labels/labels_shard{0,1,2}.jsonl`. Not
every GPU with free-looking `nvidia-smi` memory works: `--gpu-memory-
utilization` is a fraction of *total* device memory, so a GPU already
hosting another user's process needs a correspondingly lower value or it
OOMs at startup (hit this on one GPU, moved to a fully-free one instead).

**Second infra issue found mid-run**: the initial 3-shard pass used
`--max-model-len 8192`, which is enough for VBVR's small synthetic images
but not for EPIC's higher-resolution frames -- 4 EPIC images + text
encoded to ~8349 tokens, over budget, so every EPIC row (the shard
containing them) failed with `400 Bad Request: decoder prompt ... longer
than the maximum model length`. Fixed by re-serving on 2 GPUs with
`--max-model-len 12288` and re-running just the ~6798 affected EPIC ids
(`manifest_retry_epic_{a,b}.jsonl`) split across both. Lesson for any
future re-run: size `--max-model-len` off the *largest* source's image
resolution, not VBVR's (smallest); 12288 was comfortably sufficient for
all 5 sources here.

**Final run complete.** `fill_s4_captions.py` joined all label files
(3 shards + 2 EPIC retries) onto `S4_skeleton.jsonl` by row id:

```
data/v1/datasets/s4_progression/S4_train.jsonl   49,112 rows
data/v1/meta/final_S4_meta.json                  {"final_S4": {..., "length": 49112, "max_dynamic_patch": 3, "task_type": "imgen"}}
```

Per-source: `{vbvr: 24999, epic: 6999, nwm: 5284, panda70m_epic: 5576,
panda70m_v1: 6254}` -- exactly S2's counts minus the 1 row
(`final_S2_013523`, a VBVR row) that failed to parse across every attempt
and was dropped (id logged in `S4_dropped_ids.txt`). Verified after
assembly: every row's `conversations` has exactly a human turn followed by
a GPT turn ending in `<img>`, valid JSON throughout, 0 malformed.

This dataset is ready to point a training config at directly (same
`root`/`max_dynamic_patch`/`task_type` shape as the other `final_S{0,1,2,3}`
meta entries) -- see "Still open" below for the one remaining tuning
question before actually training on it.

**Still open**: whether `LM_LOSS_WEIGHT`/`GEN_LOSS_WEIGHT` need retuning now
that the text target is a real multi-sentence caption instead of S3's
single letter -- deferred to whoever sets up the actual training run.

## Files

```
data/v1/common/s4_progression_prompt.py           shared instruction/prompt templates (all sources)
data/v1/common/s4_caption_lookup.py               join-by-context_paths helper (4 real-world sources)
data/v1/sources/<name>/processing/
    build_s4_caption_prefix.py                 per-source: recover S2's caption text for S4 reuse
                                                (one script per source: vbvr, epic_kitchens, nwm,
                                                panda70m_epic, panda70m_v1)
data/v1/recipes/s4_progression/
    build_all.sh                                orchestrates all 5 sources' build_s4_caption_prefix.py
                                                 + derive_s4_rows.py in one command
    derive_s4_rows.py                           S2_train.jsonl + the 5 sources' caption-prefix files
                                                 -> S4_skeleton.jsonl + S4_captioning_manifest.jsonl
    label_with_vlm.py                           calls a vLLM OpenAI-compatible server to label rows
                                                 (progression + next-frame text), --print-only for
                                                 spot-checking or --out-jsonl for a real run
    fill_s4_captions.py                         joins label_with_vlm.py output onto S4_skeleton.jsonl
                                                 by row id -> final S4_train.jsonl + meta json
data/v1/datasets/s4_progression/
    S4_skeleton.jsonl                           human turn built, gpt turn NOT in this file
    S4_captioning_manifest.jsonl                per-row VLM prompt + image paths for the labeling pass
    manifest_shard{0,1,2}.jsonl                 S4_captioning_manifest.jsonl split 3 ways for the
                                                 3-GPU parallel labeling run (2026-09-06)
    labels/labels_shard{0,1,2}.jsonl            label_with_vlm.py output per shard
    S4_train.jsonl                              final training-ready output (after fill_s4_captions.py)
```

No merge/filter step is needed beyond `fill_s4_captions.py` -- S4's rows
are exactly S2's already-filtered window set, so the missing-image-file
problem `filter_final_s0s3.py` exists for doesn't recur here.

## Verification performed (2026-09-06)

- All 5 sources' `build_s4_caption_prefix.py` output checked against the
  real `conversations[0]` text already in `S2_train.jsonl`: for vbvr, epic,
  panda70m_epic, panda70m_v1, the recovered caption prefix + `<image>*3`
  reproduces the original S2 text **exactly**, byte-for-byte, for all rows
  (25000/25000, 6999/6999, 5576/5576, 6254/6254). For nwm, it matches
  exactly except for the intentionally-added domain-framing sentence
  (verified separately: 5284/5284 match once that addition is excluded).
- `derive_s4_rows.py` end-to-end run: 49,113 total rows, per-source
  breakdown `{vbvr: 25000, epic: 6999, nwm: 5284, panda70m_epic: 5576,
  panda70m_v1: 6254}` -- matches `S2_train.jsonl` exactly, confirming no
  rows were dropped or duplicated in the join.
