#!/usr/bin/env python
"""MagicBrush metrics: L1, L2, CLIP-I, DINO, CLIP-T.

A close port of OSU-NLP-Group/MagicBrush's official
`evaluation/image_eval.py` (v3, "add CLIP Score and CLIP I"): same models
(CLIP ViT-B/32 via the `openai/CLIP` package, DINO ViT-S/16 via
`torch.hub.load('facebookresearch/dino:main', 'dino_vits16')`), same
preprocessing (CLIP-I/CLIP-T: model's own `clip.load` transform; DINO:
Resize(256, bicubic) -> CenterCrop(224) -> ImageNet normalize), same formulas
(L1/L2 = mean over per-pair `nn.L1Loss`/`nn.MSELoss`; CLIP-I/DINO/CLIP-T =
mean cosine similarity between encoded features), same final_turn / all_turn
split. Reads directly from our `<out>/images/<img_id>/<gen_filename>.png` +
`<data_root>/images/<img_id>/<img_id>-output{n}.png` layout instead of the
upstream flat `testResults/` tree (both use the same substring-based file
lookup, so no restructuring is needed).

Usage (env: geneval-eval-env, has openai-clip + scipy + torch/cuda):
  python eval_magicbrush.py \
      --data_root /scratch/local/ssd/junlin/data/MagicBrush/test \
      --generated /scratch/local/ssd/junlin/results/MagicBrush/InternVL-U/images \
      --save_path /scratch/local/ssd/junlin/results/MagicBrush/InternVL-U
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image
from scipy import spatial
from torch import nn
from torchvision.transforms import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_SUITE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _SUITE_ROOT)
from dataset import load_items, filter_missing  # noqa: E402


# ---- metric primitives (ported verbatim from upstream image_eval.py) ----

RESIZE = 0  # set from --resize; 0 = native resolution (existing behavior)


def _open(path):
    """Open + RGB-convert, optionally prescaling to RESIZE x RESIZE (control setting)."""
    img = Image.open(path).convert("RGB")
    if RESIZE:
        img = img.resize((RESIZE, RESIZE), Image.BICUBIC)
    return img


def eval_distance(image_pairs, metric="l1"):
    criterion = nn.L1Loss() if metric == "l1" else nn.MSELoss()
    eval_score = 0.0
    for gen_path, gt_path in image_pairs:
        gen_img = _open(gen_path)
        gt_img = _open(gt_path)
        gen_img = gen_img.resize(gt_img.size)
        gen_t = transforms.ToTensor()(gen_img)
        gt_t = transforms.ToTensor()(gt_img)
        eval_score += criterion(gen_t, gt_t).detach().cpu().numpy().item()
    return eval_score / len(image_pairs)


def _encode_image(image, model, transform, device, metric):
    image_input = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if metric == "clip_i":
            feat = model.encode_image(image_input).detach().cpu().float()
        else:  # dino
            feat = model(image_input).detach().cpu().float()
    return feat


def eval_clip_i(image_pairs, model, transform, device, metric="clip_i"):
    eval_score = 0.0
    for gen_path, gt_path in image_pairs:
        gen_feat = _encode_image(_open(gen_path), model, transform, device, metric)
        gt_feat = _encode_image(_open(gt_path), model, transform, device, metric)
        sim = 1 - spatial.distance.cosine(gen_feat.view(-1), gt_feat.view(-1))
        eval_score += sim
    return eval_score / len(image_pairs)


def eval_clip_t(image_pairs, captions, model, transform, device):
    """image_pairs: (gen_path, gt_path); captions: {gt_path: caption}."""
    import clip as _clip
    gen_score, gt_score = 0.0, 0.0
    for gen_path, gt_path in image_pairs:
        caption = captions[gt_path]
        gen_feat = _encode_image(_open(gen_path), model, transform, device, "clip_i")
        gt_feat = _encode_image(_open(gt_path), model, transform, device, "clip_i")
        text_tok = _clip.tokenize(caption, truncate=True).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(text_tok).detach().cpu().float()
        gen_score += 1 - spatial.distance.cosine(gen_feat.view(-1), text_feat.view(-1))
        gt_score += 1 - spatial.distance.cosine(gt_feat.view(-1), text_feat.view(-1))
    return gen_score / len(image_pairs), gt_score / len(image_pairs)


# ---- pairing: our items already carry exact gen/gt paths, no dir scanning needed ----

def build_pairs(data_root, generated_root, img_ids_file=None):
    items = filter_missing(load_items(data_root))
    if img_ids_file:
        allowed = set(json.load(open(img_ids_file))["img_ids"])
        items = [it for it in items if it.img_id in allowed]
    all_turn_pairs, final_turn_pairs, captions = [], [], {}
    missing = 0
    last_turn_of_session = {}
    for it in items:
        last_turn_of_session[it.img_id] = max(last_turn_of_session.get(it.img_id, 0), it.turn)

    by_session = {}
    for it in items:
        gen_path = os.path.join(generated_root, it.img_id, it.gen_filename)
        if not os.path.exists(gen_path):
            missing += 1
            continue
        pair = (gen_path, it.gt_output_path)
        all_turn_pairs.append(pair)
        captions[it.gt_output_path] = it.target_caption
        by_session.setdefault(it.img_id, []).append(it)

    for img_id, turns in by_session.items():
        last = max(turns, key=lambda x: x.turn)
        final_turn_pairs.append((os.path.join(generated_root, last.img_id, last.gen_filename), last.gt_output_path))

    if missing:
        print(f"[eval] WARNING: {missing} items have no generated image (skipped)", flush=True)
    return all_turn_pairs, final_turn_pairs, captions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="MagicBrush test data_root (has edit_sessions.json, images/)")
    ap.add_argument("--generated", required=True, help="<out>/images dir from a gen_magicbrush_*.py run")
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metric", default="l1,l2,clip-i,dino,clip-t")
    ap.add_argument("--resize", type=int, default=0,
                     help="if set, prescale both generated and GT images to resize x resize before "
                          "any metric (control setting; 0 = native resolution, existing behavior)")
    ap.add_argument("--img_ids_file", default=None,
                     help="optional JSON file with an 'img_ids' list; restricts items to those sessions")
    args = ap.parse_args()
    metrics = args.metric.split(",")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    global RESIZE
    RESIZE = args.resize

    all_turn_pairs, final_turn_pairs, captions = build_pairs(args.data_root, args.generated, args.img_ids_file)
    print(f"[eval] all_turn pairs: {len(all_turn_pairs)}  final_turn pairs: {len(final_turn_pairs)}", flush=True)

    results = {"final_turn": {}, "all_turn": {}}

    if "l1" in metrics:
        results["final_turn"]["l1"] = eval_distance(final_turn_pairs, "l1")
        results["all_turn"]["l1"] = eval_distance(all_turn_pairs, "l1")
        print("L1:", results["final_turn"]["l1"], results["all_turn"]["l1"], flush=True)
    if "l2" in metrics:
        results["final_turn"]["l2"] = eval_distance(final_turn_pairs, "l2")
        results["all_turn"]["l2"] = eval_distance(all_turn_pairs, "l2")
        print("L2:", results["final_turn"]["l2"], results["all_turn"]["l2"], flush=True)

    if "clip-i" in metrics or "clip-t" in metrics:
        import clip as _clip
        clip_model, clip_transform = _clip.load("ViT-B/32", device)
        clip_model.eval()

    if "clip-i" in metrics:
        results["final_turn"]["clip-i"] = eval_clip_i(final_turn_pairs, clip_model, clip_transform, device, "clip_i")
        results["all_turn"]["clip-i"] = eval_clip_i(all_turn_pairs, clip_model, clip_transform, device, "clip_i")
        print("CLIP-I:", results["final_turn"]["clip-i"], results["all_turn"]["clip-i"], flush=True)

    if "dino" in metrics:
        dino_model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        dino_model.eval().to(device)
        dino_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        results["final_turn"]["dino"] = eval_clip_i(final_turn_pairs, dino_model, dino_transform, device, "dino")
        results["all_turn"]["dino"] = eval_clip_i(all_turn_pairs, dino_model, dino_transform, device, "dino")
        print("DINO:", results["final_turn"]["dino"], results["all_turn"]["dino"], flush=True)

    if "clip-t" in metrics:
        # Upstream's __main__ passes the same clip.load preprocessing transform
        # (proper CLIP resize/crop/normalize) to eval_clip_t as eval_clip_i uses;
        # the separately-defined Resize/CenterCrop/ToTensor transform inside
        # eval_clip_score is dead code (that function is never called from
        # __main__), so we don't reproduce it here.
        fg, fo = eval_clip_t(final_turn_pairs, captions, clip_model, clip_transform, device)
        ag, ao = eval_clip_t(all_turn_pairs, captions, clip_model, clip_transform, device)
        results["final_turn"]["clip-t"] = fg
        results["final_turn"]["clip-t_oracle"] = fo
        results["all_turn"]["clip-t"] = ag
        results["all_turn"]["clip-t_oracle"] = ao
        print("CLIP-T:", fg, ag, "(oracle:", fo, ao, ")", flush=True)

    os.makedirs(args.save_path, exist_ok=True)
    out_json = os.path.join(args.save_path, "evaluation_metrics.json")
    json.dump(results, open(out_json, "w"), indent=2)
    print(f"[eval] wrote {out_json}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
