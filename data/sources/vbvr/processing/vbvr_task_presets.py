"""Named --tasks presets shared across every VBVR-DataGeneration script that
accepts a ``--tasks nargs="*"`` argument (next_frame/ and target_pred/'s
generate/sample/build scripts).

Why this exists: the 6 scripts that select VBVR tasks each validate --tasks
against their OWN local task dict (vbvr_next_frame_generate.py's ALL_TASKS,
vbvr_target_pred_generate.py's TASKS_ID/TASKS_OOD, etc. -- deliberately kept
independent per script, see vbvr_target_pred_generate.py's TASKS_ID comment).
A flat named list here lets you pick a task subset ONCE and reuse it
everywhere via --task-preset, without retyping/keeping-in-sync a long
--tasks list by hand across every script, while each script still only
picks up the subset of the preset that's actually valid for it -- see
resolve_tasks().

Add a new preset by adding a list entry below; nothing else needs to change.
"""

TASK_PRESETS = {
    # 2026-08-30: the 5-task VBVR subset chosen for the S0-S3 training-paradigm
    # redesign (3 ID canonical + 2 OOD-folded-into-training). Categories:
    # Perception x1 (multi_object_placement), Abstraction x2
    # (shape_color_then_move, shape_outline_then_move), Transformation x2
    # (rotation_puzzle, 2d_geometric_transformation) -- Knowledge/Spatiality
    # not covered, noted and accepted.
    "pilot5": [
        "multi_object_placement",
        "shape_color_then_move",
        "shape_outline_then_move",
        "rotation_puzzle",
        "2d_geometric_transformation",
    ],
}


def resolve_tasks(explicit_tasks, preset_name, valid_tasks, default_tasks, *, label="tasks"):
    """Decide the final task-name list for a script's main(), given its own
    --tasks/--task-preset args plus that script's local notion of "valid" and
    "default" tasks.

    - explicit_tasks (args.tasks): wins outright; --task-preset must be unset
      alongside it (ambiguous otherwise -- fail loud, not silently pick one).
    - preset_name (args.task_preset): looked up in TASK_PRESETS, then
      filtered down to whatever's in valid_tasks (e.g. a script that only
      knows TASKS_OOD silently drops the preset's ID-domain task names --
      this is what makes ONE preset usable across scripts that each cover
      only part of it, e.g. vbvr_target_pred_generate.py --domain id vs.
      vbvr_target_pred_generate_ood_train.py). Tasks dropped this way are
      printed so it's never silent in practice. Raises if that leaves nothing.
    - neither given: falls back to the script's own default_tasks, unchanged
      behavior from before --task-preset existed.
    """
    if explicit_tasks and preset_name:
        raise SystemExit("pass either --tasks or --task-preset, not both")
    if explicit_tasks:
        return list(explicit_tasks)
    if preset_name:
        if preset_name not in TASK_PRESETS:
            raise SystemExit(f"unknown --task-preset {preset_name!r} (valid: {list(TASK_PRESETS)})")
        preset = TASK_PRESETS[preset_name]
        applicable = [t for t in preset if t in valid_tasks]
        skipped = [t for t in preset if t not in valid_tasks]
        if skipped:
            print(f"[task-preset {preset_name}] not valid here, skipping: {skipped} "
                  f"({label})", flush=True)
        if not applicable:
            raise SystemExit(
                f"--task-preset {preset_name} has no {label} valid for this script "
                f"(preset: {preset}, valid here: {sorted(valid_tasks)})"
            )
        return applicable
    return default_tasks
