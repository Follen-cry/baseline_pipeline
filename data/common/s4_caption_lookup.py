"""Shared helper for the 4 real-world sources' S4 caption-prefix recovery:
each source already writes a manifest/anchors jsonl (alongside its S2
training jsonl) that carries the raw action_text/caption per row, keyed by
`context_paths` -- the same 3 paths that end up in the training row's
`image` field. Joining on that tuple (rather than row `id`, which gets
renumbered by merge_final_s0s3.py) is what stays valid after merging.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def load_context_path_lookup(
    manifest_paths: list[Path],
    extract: Callable[[dict], str],
) -> dict[tuple, str]:
    """extract(manifest_row) -> caption/action text. Returns
    {tuple(context_paths): text}."""
    lookup = {}
    for p in manifest_paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                lookup[tuple(d["context_paths"])] = extract(d)
    return lookup


def derive_source_caption_prefixes(
    s2_jsonl: Path, source_name: str, manifest_paths: list[Path],
    extract: Callable[[dict], str],
) -> list[dict]:
    """Standard driver: read S2_train.jsonl, keep rows for `source_name`,
    join each row's `image` (== context_paths) against the manifest lookup.
    Returns a list of {"context_paths": [...], "caption_prefix": str} dicts;
    raises with a clear count of misses rather than silently dropping rows,
    since a join miss here means the manifest and S2_train.jsonl have
    drifted out of sync and the recipe should stop, not guess.
    """
    lookup = load_context_path_lookup(manifest_paths, extract)
    out, misses = [], []
    with open(s2_jsonl) as f:
        for line in f:
            row = json.loads(line)
            if row.get("source") != source_name:
                continue
            key = tuple(row["image"])
            if key not in lookup:
                misses.append(row.get("id"))
                continue
            out.append({"context_paths": row["image"], "caption_prefix": lookup[key]})
    if misses:
        raise RuntimeError(
            f"{source_name}: {len(misses)} S2 rows had no matching manifest "
            f"entry (join on context_paths) -- e.g. ids {misses[:5]}. "
            "Manifest and S2_train.jsonl are out of sync; investigate before "
            "proceeding."
        )
    return out
