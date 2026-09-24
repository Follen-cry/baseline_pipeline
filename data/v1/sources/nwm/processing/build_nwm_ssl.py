"""Builder for NWM/RECON SSL training data (next-frame selection FFS + generation
+ dual FFS/generation), for InternVL-U SFT. See data/specs/NWM_SSL_DATA_SPEC.md.

Reuses the pose->action->transition logic already built and validated for the
NWM eval benchmark (Evaluation/nwm/vlm_benchmark/core.py) -- that is a separate
git repo, so this script does a one-off sys.path insert to import it. This is a
deliberate, narrow exception to the usual dependency direction (Evaluation/
depends on data/, not the reverse) -- justified because the validated
pose-to-frame-selection math only exists there; do not generalize this pattern
to other data/ builders.

Only reads Evaluation/nwm/data_splits/recon/train/traj_names.txt (9468
trajectories). Never reads .../test/traj_names.txt (2367 trajectories, reserved
for the eval benchmark) -- enforced by an explicit assertion in
build_trajectory_splits().

Images are NOT copied (ssd2 is nearly full); "image"/"target_image" paths in
the output jsonls are absolute paths directly into
/scratch/network/ssd/junlin/nwm_data/recon/<traj>/<i>.jpg.

Stages (run with --stage, in order; each writes its output so later stages can
be re-run independently once earlier ones exist):
  splits   -- carve RECON's train split into our own internal train/eval
              trajectory pools (data/datasets/nwm_ssl/splits/build_{train,eval}_traj.txt)
  anchors  -- compile_transitions + build_selection_examples on each pool,
              subsample to 4000/300 with a per-trajectory cap, write
              data/datasets/nwm_ssl/anchors/anchors_{train,eval}.jsonl (SelectionExample
              records -- these ARE the shared ground truth for FFS, generation,
              and dual)
  ffs      -- build nwm_ffs_{action,noaction}_{train,eval}.jsonl from the
              anchors (task_type "understanding")
  gen      -- build nwm_gen_{action,noaction}_{train,eval}.jsonl from the same
              anchors (task_type "imgen"; target_image = the anchor's
              ground_truth candidate, i.e. gen target == FFS true frame)
  dual     -- convert nwm_ffs_{action,noaction}_{train,eval}.jsonl 1:1 into
              nwm_dual_{action,noaction}_{train,eval}.jsonl via
              Model_Related/InternVLU/InternVL/internvl_chat/tools/
              make_nwm_dual_ssl.py (task_type "imgen", imgen_form "multimodal")

Usage:
  python data/scripts/build_nwm_ssl.py --stage splits
  python data/scripts/build_nwm_ssl.py --stage anchors
  python data/scripts/build_nwm_ssl.py --stage ffs
  python data/scripts/build_nwm_ssl.py --stage gen
  python data/scripts/build_nwm_ssl.py --stage dual
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

NWM_ROOT = "/scratch/network/ssd2/junlin/ssl_mllm/Evaluation/nwm"
if NWM_ROOT not in sys.path:
    sys.path.insert(0, NWM_ROOT)
from vlm_benchmark.core import (  # noqa: E402
    compile_transitions, build_selection_examples, SelectionExample,
    read_jsonl, write_jsonl,
)
from vlm_benchmark.adapters import build_selection_prompt_interleaved  # noqa: E402

DUAL_TOOL_DIR = ("/scratch/network/ssd2/junlin/ssl_mllm/Model_Related/InternVLU/"
                  "InternVL/internvl_chat/tools")
if DUAL_TOOL_DIR not in sys.path:
    sys.path.insert(0, DUAL_TOOL_DIR)

RECON_DATA = "/scratch/network/ssd/junlin/nwm_data/recon"
TRAIN_TRAJ_LIST = f"{NWM_ROOT}/data_splits/recon/train/traj_names.txt"
TEST_TRAJ_LIST = f"{NWM_ROOT}/data_splits/recon/test/traj_names.txt"
DATA = "/scratch/network/ssd2/junlin/ssl_mllm/data"
OUT = f"{DATA}/nwm_ssl"

CONTEXT_SIZE = 3   # was 4; shrunk 2026-08-31 to match the shared real-video-mix
                    # convention (VBVR/EPIC/panda70m all use 3 context frames)
STRIDE = 16         # was 8; widened so the wider window (context+horizon covers
                    # more native frames now) still gives ~4.0s between windows
                    # at native 4.0fps, matching EPIC's window spacing
HORIZON = 2         # was 8 (== old STRIDE, deliberate chaining); now 0.5s at
                    # native 4.0fps -- matches EPIC/panda70m's 0.5s "next frame"
                    # horizon. NWM's native pre-extraction is fixed at 4.0fps
                    # (0.25s/frame), so context frames stay 0.25s apart even
                    # though the TARGET horizon (0.5s) now matches everyone else.
INPUT_FPS = 4.0
NUM_CANDIDATES = 4
SPLIT_SEED = 42
ANCHOR_SEED = 0
EVAL_HOLDOUT_TRAJ = 500  # trajectories carved out for our internal eval pool
TARGET_TRAIN = 5284      # was 4000; part of the 2026-08-31 real-video-mix rebalance
                          # (25k total: v1=7016 full, EPIC=7000, panda70m_epic_ssl=5700,
                          # NWM=5284 -- deliberately capped well below its ~22k ceiling
                          # so navigation footage doesn't dominate the real-world mix)
TARGET_EVAL = 300
PER_TRAJ_CAP = 2


def _read_lines(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def build_trajectory_splits():
    """Carve RECON's official train split (9468 trajs) into our OWN internal
    train/eval trajectory pools. Never touches the official test split (2367
    trajs, reserved for the eval benchmark) -- asserted below."""
    train_names = set(_read_lines(TRAIN_TRAJ_LIST))
    test_names = set(_read_lines(TEST_TRAJ_LIST))
    assert not (train_names & test_names), "official train/test splits overlap?!"

    present = {p.name for p in Path(RECON_DATA).iterdir() if p.is_dir()}
    pool = sorted(train_names & present)
    missing = train_names - present
    if missing:
        print(f"[splits] {len(missing)}/{len(train_names)} train trajectories not materialized locally, skipping")

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pool)
    eval_trajs = sorted(pool[:EVAL_HOLDOUT_TRAJ])
    train_trajs = sorted(pool[EVAL_HOLDOUT_TRAJ:])

    assert not (set(eval_trajs) & set(train_trajs))
    assert not (set(eval_trajs) & test_names) and not (set(train_trajs) & test_names)

    out_dir = Path(OUT) / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "build_train_traj.txt").write_text("\n".join(train_trajs) + "\n")
    (out_dir / "build_eval_traj.txt").write_text("\n".join(eval_trajs) + "\n")
    print(f"[splits] train={len(train_trajs)} eval={len(eval_trajs)} trajectories -> {out_dir}")


def _subsample_capped(examples, target_n, cap, seed):
    """Deterministic subsample with a per-trajectory cap for scene diversity;
    relaxes the cap only if the capped pool can't reach target_n."""
    rng = random.Random(seed)
    order = list(examples)
    rng.shuffle(order)

    counts, selected, leftover = {}, [], []
    for ex in order:
        c = counts.get(ex.trajectory_id, 0)
        if c < cap:
            selected.append(ex)
            counts[ex.trajectory_id] = c + 1
            if len(selected) >= target_n:
                return selected
        else:
            leftover.append(ex)
    for ex in leftover:
        selected.append(ex)
        if len(selected) >= target_n:
            break
    if len(selected) < target_n:
        print(f"[anchors] WARNING: only found {len(selected)}/{target_n} examples after full relax")
    return selected


def build_anchor_pool(traj_list_path, target_n):
    transitions = compile_transitions(
        data_root=RECON_DATA, trajectory_names_path=traj_list_path, dataset="recon",
        horizons=[HORIZON], context_size=CONTEXT_SIZE, stride=STRIDE, input_fps=INPUT_FPS,
        split="train", require_images=True)
    print(f"[anchors] {traj_list_path}: {len(transitions)} raw transitions compiled")
    examples = build_selection_examples(transitions, num_candidates=NUM_CANDIDATES, seed=ANCHOR_SEED)
    print(f"[anchors] {len(examples)} selection examples built (candidate mining succeeded)")
    return _subsample_capped(examples, target_n, PER_TRAJ_CAP, ANCHOR_SEED)


def build_anchors():
    splits_dir = Path(OUT) / "splits"
    anchors_dir = Path(OUT) / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)

    for split, traj_file, target_n in (
        ("train", splits_dir / "build_train_traj.txt", TARGET_TRAIN),
        ("eval", splits_dir / "build_eval_traj.txt", TARGET_EVAL),
    ):
        anchors = build_anchor_pool(str(traj_file), target_n)
        write_jsonl(anchors_dir / f"anchors_{split}.jsonl", anchors)
        traj_used = len({a.trajectory_id for a in anchors})
        print(f"[anchors] {split}: wrote {len(anchors)} anchors ({traj_used} distinct trajectories) "
              f"-> {anchors_dir}/anchors_{split}.jsonl")

    # sanity: oracle check -- answer_index must always point at a "ground_truth" candidate
    for split in ("train", "eval"):
        anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
        bad = [a.example_id for a in anchors if a.candidate_types[a.answer_index] != "ground_truth"]
        assert not bad, f"oracle sanity failed for {len(bad)} anchors in {split}: {bad[:5]}"
    print("[anchors] oracle sanity check passed (answer_index always ground_truth)")


def build_ffs_prompt_noaction(example: SelectionExample) -> str:
    """No-action counterpart of vlm_benchmark.adapters.build_selection_prompt_interleaved
    -- the action-related line is omitted ENTIRELY (not replaced with a
    placeholder sentence), matching how IntPhys2 never had an action concept
    to begin with. Intentionally NOT added to Evaluation/nwm/vlm_benchmark
    (that repo is eval-only; this is training-data-only)."""
    n, k = len(example.context_paths), len(example.candidate_paths)
    labels = [chr(ord("A") + i) for i in range(k)]
    head = "".join(f"Frame {i + 1}: <image>\n" for i in range(n))
    body = f"These are {n} consecutive first-person camera frames in order.\n"
    opts = "".join(f"Option {labels[i]}: <image>\n" for i in range(k))
    return (head + body + opts +
            f"Which option is the true next frame? Answer with a single letter ({', '.join(labels)}).")


def _build_ffs_split(anchors, variant):
    build_prompt = build_selection_prompt_interleaved if variant == "action" else build_ffs_prompt_noaction
    rows, manifest = [], []
    for idx, ex in enumerate(anchors):
        images = list(ex.context_paths) + list(ex.candidate_paths)
        prompt = build_prompt(ex)
        rows.append({
            "id": idx, "task_type": "understanding", "image": images,
            "conversations": [{"from": "human", "value": prompt},
                               {"from": "gpt", "value": ex.answer_label}],
        })
        manifest.append(ex.to_dict())
    return rows, manifest


def build_ffs():
    anchors_dir = Path(OUT) / "anchors"
    for variant in ("action", "noaction"):
        task_dir = Path(OUT) / f"nwm_ffs_{variant}"
        task_dir.mkdir(parents=True, exist_ok=True)
        lengths = {}
        for split in ("train", "eval"):
            anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
            rows, manifest = _build_ffs_split(anchors, variant)
            with open(task_dir / f"nwm_ffs_{variant}_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            write_jsonl(task_dir / f"manifest_{split}.jsonl", manifest)
            lengths[split] = len(rows)
            print(f"[ffs:{variant}] {split}: {len(rows)} rows -> {task_dir}/nwm_ffs_{variant}_{split}.jsonl")

        meta_dir = Path(DATA) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        json.dump({f"nwm_ffs_{variant}": {
            "root": RECON_DATA,
            "annotation": str(task_dir / f"nwm_ffs_{variant}_train.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
            "repeat_time": 1, "length": lengths["train"], "task_type": "understanding"}},
            open(meta_dir / f"nwm_ffs_{variant}_meta.json", "w"), indent=2)
        json.dump({f"nwm_ffs_{variant}_eval": {
            "root": RECON_DATA,
            "annotation": str(task_dir / f"nwm_ffs_{variant}_eval.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
            "repeat_time": 1, "length": lengths["eval"], "task_type": "understanding"}},
            open(meta_dir / f"nwm_ffs_{variant}_eval_meta.json", "w"), indent=2)

        with open(task_dir / "VERSION", "w") as f:
            f.write(
                "version: v1\n"
                "task: next-frame selection (understanding, 4-option MCQ)\n"
                f"action_variant: {variant}\n"
                "dataset: RECON (NWM), our internal train/eval carve-out of the OFFICIAL "
                "train split only (9468 trajectories total; official test split of 2367 "
                "trajectories is untouched, reserved for Evaluation/nwm's eval benchmark)\n"
                f"context_size: {CONTEXT_SIZE}, stride: {STRIDE}, input_fps: {INPUT_FPS}, "
                f"horizon: {HORIZON} frames (1 context-stride)\n"
                f"split: {lengths['train']} train + {lengths['eval']} eval "
                f"(trajectory-disjoint, held out {EVAL_HOLDOUT_TRAJ} trajectories for eval, "
                f"cap {PER_TRAJ_CAP} examples/trajectory)\n"
                f"split_seed: {SPLIT_SEED}, anchor_seed: {ANCHOR_SEED}\n"
                "candidates: pose-derived hard negatives (wrong_yaw / wrong_distance / "
                "nearby_wrong_state / same_trajectory_distractor), via "
                "Evaluation/nwm/vlm_benchmark/core.py:_negatives (reused unmodified)\n"
                "note: nwm_ffs_action and nwm_ffs_noaction share IDENTICAL context/candidates/"
                "answers (same anchors file) -- only the human-turn prompt text differs\n"
                "created: 2026-07-26\n"
                "spec: ../NWM_SSL_DATA_SPEC.md\n"
            )
    print("[ffs] done")


GEN_GPT_ANSWER = "The next frame should look like this: <img>"


def build_gen_prompt_action(n: int, action_text: str) -> str:
    head = "".join("<image>\n" for _ in range(n))
    return (head + f"Action to execute from the last frame: {action_text}\n"
            "Observe these four frames and, given the action, produce the next frame of the video.")


def build_gen_prompt_noaction(n: int) -> str:
    head = "".join("<image>\n" for _ in range(n))
    return head + "Observe these four frames and produce the next frame of the video."


def _build_gen_split(anchors, variant):
    """One correct-condition row per anchor -- gen target = the SAME anchor's
    ground_truth candidate (== the FFS true frame), no eval-style opposite/
    shuffled/null conditions (those are eval-only controls, not SFT targets)."""
    rows, manifest = [], []
    for idx, ex in enumerate(anchors):
        n = len(ex.context_paths)
        target = ex.candidate_paths[ex.answer_index]
        prompt = (build_gen_prompt_action(n, ex.action_text) if variant == "action"
                  else build_gen_prompt_noaction(n))
        rows.append({
            "id": idx, "task_type": "imgen",
            "image": list(ex.context_paths), "target_image": target,
            "conversations": [{"from": "human", "value": prompt},
                               {"from": "gpt", "value": GEN_GPT_ANSWER}],
        })
        manifest.append(ex.to_dict())
    return rows, manifest


EDIT_GPT_ANSWER = "Here is the edited frame after executing this camera action: <img>"


def build_edit_prompt_action(action_text: str) -> str:
    """Image-editing framing of the SAME next-frame-generation target: single
    input image (last context frame only, not the 4-frame context), the
    action described as an edit instruction to apply to it. Reuses the
    identical action_text produced for nwm_ffs_action/nwm_gen_action -- only
    the image count and prompt wording differ."""
    return ("<image>\n"
            "This image is a single frame captured by a forward-facing camera. "
            f"Apply the following camera action to it: {action_text}\n"
            "Edit the image to depict what the camera would see immediately "
            "after performing this action, keeping the same scene but "
            "updating the viewpoint accordingly.")


def _build_edit_split(anchors):
    """One row per anchor, same ground_truth target as nwm_gen_action, but
    image = ONLY the last context frame (ctx4), not all 4 context frames."""
    rows, manifest = [], []
    for idx, ex in enumerate(anchors):
        last_frame = ex.context_paths[-1]
        target = ex.candidate_paths[ex.answer_index]
        prompt = build_edit_prompt_action(ex.action_text)
        rows.append({
            "id": idx, "task_type": "imgen",
            "image": [last_frame], "target_image": target,
            "conversations": [{"from": "human", "value": prompt},
                               {"from": "gpt", "value": EDIT_GPT_ANSWER}],
        })
        manifest.append(ex.to_dict())
    return rows, manifest


def build_edit():
    anchors_dir = Path(OUT) / "anchors"
    task_dir = Path(OUT) / "nwm_edit_action"
    task_dir.mkdir(parents=True, exist_ok=True)
    lengths = {}
    for split in ("train", "eval"):
        anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
        rows, manifest = _build_edit_split(anchors)
        with open(task_dir / f"nwm_edit_action_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_jsonl(task_dir / f"manifest_{split}.jsonl", manifest)
        lengths[split] = len(rows)
        print(f"[edit:action] {split}: {len(rows)} rows -> {task_dir}/nwm_edit_action_{split}.jsonl")

    meta_dir = Path(DATA) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"nwm_edit_action": {
        "root": RECON_DATA,
        "annotation": str(task_dir / "nwm_edit_action_train.jsonl"),
        "data_augment": False, "max_dynamic_patch": 1,
        "repeat_time": 1, "length": lengths["train"], "task_type": "imgen"}},
        open(meta_dir / "nwm_edit_action_meta.json", "w"), indent=2)
    json.dump({"nwm_edit_action_eval": {
        "root": RECON_DATA,
        "annotation": str(task_dir / "nwm_edit_action_eval.jsonl"),
        "data_augment": False, "max_dynamic_patch": 1,
        "repeat_time": 1, "length": lengths["eval"], "task_type": "imgen"}},
        open(meta_dir / "nwm_edit_action_eval_meta.json", "w"), indent=2)

    with open(task_dir / "VERSION", "w") as f:
        f.write(
            "version: v1\n"
            "task: next-frame generation reframed as single-image editing "
            "(imgen); camera action_text is the edit instruction\n"
            "dataset: RECON (NWM), same internal train/eval trajectory carve-out, "
            "anchors, and ground-truth targets as nwm_gen_action (official test "
            "split of 2367 trajectories is untouched)\n"
            "image: ONLY the last context frame (ctx4), NOT the 4-frame context "
            "used by nwm_gen_action/nwm_ffs_action -- this is the key difference; "
            "target_image is identical to nwm_gen_action's target for the same anchor\n"
            f"context_size: 1 (single input frame), stride: {STRIDE}, input_fps: {INPUT_FPS}, "
            f"horizon: {HORIZON} frames (1 context-stride)\n"
            f"split: {lengths['train']} train + {lengths['eval']} eval "
            "(same anchors as nwm_gen_action)\n"
            f"split_seed: {SPLIT_SEED}, anchor_seed: {ANCHOR_SEED}\n"
            "no noaction counterpart: the task IS the action-as-edit-instruction, "
            "so an action-free version would have no edit instruction to follow\n"
            "conditioning: image passes through BOTH ViT and VAE, same as nwm_gen_*, via "
            "INTERNVLU_VAE_COND=1 (falls back to the sole input image as cond_image)\n"
            "created: 2026-07-27\n"
            "spec: ../NWM_SSL_DATA_SPEC.md\n"
        )
    print("[edit] done")


def build_gen():
    anchors_dir = Path(OUT) / "anchors"
    for variant in ("action", "noaction"):
        task_dir = Path(OUT) / f"nwm_gen_{variant}"
        task_dir.mkdir(parents=True, exist_ok=True)
        lengths = {}
        for split in ("train", "eval"):
            anchors = read_jsonl(anchors_dir / f"anchors_{split}.jsonl", SelectionExample.from_dict)
            rows, manifest = _build_gen_split(anchors, variant)
            with open(task_dir / f"nwm_gen_{variant}_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            write_jsonl(task_dir / f"manifest_{split}.jsonl", manifest)
            lengths[split] = len(rows)
            print(f"[gen:{variant}] {split}: {len(rows)} rows -> {task_dir}/nwm_gen_{variant}_{split}.jsonl")

        meta_dir = Path(DATA) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        json.dump({f"nwm_gen_{variant}": {
            "root": RECON_DATA,
            "annotation": str(task_dir / f"nwm_gen_{variant}_train.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE,
            "repeat_time": 1, "length": lengths["train"], "task_type": "imgen"}},
            open(meta_dir / f"nwm_gen_{variant}_meta.json", "w"), indent=2)
        json.dump({f"nwm_gen_{variant}_eval": {
            "root": RECON_DATA,
            "annotation": str(task_dir / f"nwm_gen_{variant}_eval.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE,
            "repeat_time": 1, "length": lengths["eval"], "task_type": "imgen"}},
            open(meta_dir / f"nwm_gen_{variant}_eval_meta.json", "w"), indent=2)

        with open(task_dir / "VERSION", "w") as f:
            f.write(
                "version: v1\n"
                "task: next-frame generation (imgen)\n"
                f"action_variant: {variant}\n"
                "dataset: RECON (NWM), same internal train/eval trajectory carve-out and "
                "anchors as nwm_ffs_* (gen target == FFS true frame; official test split "
                "of 2367 trajectories is untouched)\n"
                f"context_size: {CONTEXT_SIZE}, stride: {STRIDE}, input_fps: {INPUT_FPS}, "
                f"horizon: {HORIZON} frames (1 context-stride)\n"
                f"split: {lengths['train']} train + {lengths['eval']} eval "
                "(same anchors as nwm_ffs_*, correct-action condition only)\n"
                f"split_seed: {SPLIT_SEED}, anchor_seed: {ANCHOR_SEED}\n"
                "note: nwm_gen_action and nwm_gen_noaction share IDENTICAL context/target "
                "images (same anchors file) -- only the human-turn prompt text differs\n"
                "created: 2026-07-26\n"
                "spec: ../NWM_SSL_DATA_SPEC.md\n"
            )
    print("[gen] done")


def build_dual():
    import make_nwm_dual_ssl as dual_tool

    for variant in ("action", "noaction"):
        src_dir = Path(OUT) / f"nwm_ffs_{variant}"
        dst_dir = Path(OUT) / f"nwm_dual_{variant}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        lengths = {}
        for split in ("train", "eval"):
            src = src_dir / f"nwm_ffs_{variant}_{split}.jsonl"
            dst = dst_dir / f"nwm_dual_{variant}_{split}.jsonl"
            n = dual_tool.convert_split(src, dst, n_ctx=CONTEXT_SIZE)
            lengths[split] = n
            print(f"[dual:{variant}] {split}: {n} rows -> {dst}")

        meta_dir = Path(DATA) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        json.dump({f"nwm_dual_{variant}": {
            "root": RECON_DATA,
            "annotation": str(dst_dir / f"nwm_dual_{variant}_train.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
            "repeat_time": 1, "length": lengths["train"], "task_type": "imgen"}},
            open(meta_dir / f"nwm_dual_{variant}_meta.json", "w"), indent=2)
        json.dump({f"nwm_dual_{variant}_eval": {
            "root": RECON_DATA,
            "annotation": str(dst_dir / f"nwm_dual_{variant}_eval.jsonl"),
            "data_augment": False, "max_dynamic_patch": CONTEXT_SIZE + NUM_CANDIDATES,
            "repeat_time": 1, "length": lengths["eval"], "task_type": "imgen"}},
            open(meta_dir / f"nwm_dual_{variant}_eval_meta.json", "w"), indent=2)

        with open(dst_dir / "VERSION", "w") as f:
            f.write(
                "version: v1\n"
                "task: dual FFS-selection + generation (imgen, imgen_form=multimodal)\n"
                f"action_variant: {variant}\n"
                f"converted_from: nwm_ffs_{variant}_{{train,eval}}.jsonl (1:1, via "
                "Model_Related/InternVLU/InternVL/internvl_chat/tools/make_nwm_dual_ssl.py, "
                "N_CTX=4, zero re-extraction)\n"
                f"split: {lengths['train']} train + {lengths['eval']} eval\n"
                "cond_image: last context frame (ctx4). Verified NEVER equals target_image "
                "(the ground-truth answer is never exposed via VAE conditioning), but DOES "
                "frequently (~89% of rows) pixel-duplicate a wrong-answer OPTION, because "
                "horizon==stride==8 chains consecutive sliding windows (window k's target == "
                "window k+1's ctx4). Not a leak of the answer; just not pixel-disjoint from "
                "every option the way the IntPhys2 precedent is. See NWM_SSL_DATA_SPEC.md.\n"
                "is_option: [0,0,0,0,1,1,1,1] -- read by the trainer to build "
                "option_context_mask; masking of option images out of the GENERATION "
                "conditioning only happens if --gen_no_leak is passed at train time "
                "(default False = leak ON). To get the requested 'do not mask FFS option "
                "images for generation' behavior, do NOT pass --gen_no_leak.\n"
                "created: 2026-07-26\n"
                "spec: ../NWM_SSL_DATA_SPEC.md\n"
            )
    print("[dual] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                     choices=["splits", "anchors", "ffs", "gen", "dual", "edit"])
    args = ap.parse_args()
    if args.stage == "splits":
        build_trajectory_splits()
    elif args.stage == "anchors":
        build_anchors()
    elif args.stage == "ffs":
        build_ffs()
    elif args.stage == "gen":
        build_gen()
    elif args.stage == "dual":
        build_dual()
    elif args.stage == "edit":
        build_edit()


if __name__ == "__main__":
    main()
