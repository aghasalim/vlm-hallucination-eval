"""Select and download adversarial candidate images from COCO val2017.

This produces a *draft*. It is not the evaluation set.

The critical caveat, which is the reason a manual pass exists at all: COCO
annotations cannot certify absence. They label 80 categories and miss instances
even within those, so "not in the annotations" is not the same as "not in the
image". An automatically generated absence probe is therefore an unverified
claim, and any hallucination rate computed from it would partly be measuring
COCO's annotation gaps rather than the model's behaviour.

So this script picks hard candidates and proposes probes; every probe is then
checked against the actual pixels by eye before it enters data/eval_set.json.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import config


def _iou(a: list[float], b: list[float]) -> float:
    """IoU of two COCO [x, y, w, h] boxes -- used as an occlusion proxy."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def load_coco() -> tuple[dict, dict, dict]:
    inst = json.loads((config.ANNOTATIONS / "instances_val2017.json").read_text())
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    images = {im["id"]: im for im in inst["images"]}
    per_image = defaultdict(list)
    for a in inst["annotations"]:
        if a.get("iscrowd"):
            continue
        per_image[a["image_id"]].append(a)
    return cats, images, per_image


def group_of(name: str) -> str | None:
    for g, members in config.CONFUSABLE_GROUPS.items():
        if name in members:
            return g
    return None


def score_images(cats, images, per_image) -> list[dict]:
    """Rank images by how likely they are to induce hallucination.

    Three signals, all computable from annotations:
      clutter    - many distinct instances give the model more to confabulate about
      smallness  - small objects are where attribute and identity errors live
      occlusion  - overlapping boxes mean partially hidden objects
    """
    rows = []
    for img_id, anns in per_image.items():
        if len(anns) < 4:
            continue
        im = images[img_id]
        area = im["width"] * im["height"]
        names = sorted({cats[a["category_id"]] for a in anns})
        rel_areas = [a["area"] / area for a in anns]
        boxes = [a["bbox"] for a in anns]
        max_iou = 0.0
        for i in range(min(len(boxes), 20)):
            for j in range(i + 1, min(len(boxes), 20)):
                max_iou = max(max_iou, _iou(boxes[i], boxes[j]))

        clutter = min(len(anns) / 15.0, 1.0)
        smallness = 1.0 - min(sum(rel_areas) / len(rel_areas) * 20, 1.0)
        # An image is only interesting for *adversarial* probing if one of its
        # objects has confusable siblings that are genuinely not in the scene.
        siblings = set()
        for n in names:
            g = group_of(n)
            if g:
                siblings |= set(config.CONFUSABLE_GROUPS[g])
        candidate_probes = sorted(siblings - set(names))

        rows.append({
            "image_id": img_id,
            "file_name": im["file_name"],
            "coco_url": im["coco_url"],
            "width": im["width"], "height": im["height"],
            "n_instances": len(anns),
            "present": names,
            "candidate_adversarial": candidate_probes,
            "max_iou": round(max_iou, 3),
            "difficulty": round(0.4 * clutter + 0.3 * smallness + 0.3 * min(max_iou * 2, 1.0), 4),
        })
    return sorted(rows, key=lambda r: -r["difficulty"])


def select(rows: list[dict], n: int) -> list[dict]:
    """Take the hardest images while keeping the scene types varied.

    Pure top-n by difficulty returns a pile of near-identical crowd shots, which
    would make the eval set easy to game and unrepresentative. Capping how many
    images may share a dominant category keeps the set diverse.
    """
    picked, dominant_counts = [], Counter()
    for r in rows:
        if not r["candidate_adversarial"]:
            continue
        dominant = r["present"][0]
        if dominant_counts[dominant] >= 3:
            continue
        picked.append(r)
        dominant_counts[dominant] += 1
        if len(picked) >= n:
            break
    return picked


def add_probes(rows: list[dict], all_names: list[str], freq: Counter) -> list[dict]:
    """Attach the three POPE probe splits to each image."""
    rng = random.Random(config.SEED)
    popular = [n for n, _ in freq.most_common(20)]
    for r in rows:
        present = set(r["present"])
        n_probe = min(3, len(r["candidate_adversarial"]))
        r["probes"] = {
            "adversarial": rng.sample(r["candidate_adversarial"], n_probe),
            "popular": [n for n in popular if n not in present][:3],
            "random": rng.sample([n for n in all_names if n not in present], 3),
        }
    return rows


def main() -> None:
    cats, images, per_image = load_coco()
    freq = Counter()
    for anns in per_image.values():
        freq.update({cats[a["category_id"]] for a in anns})

    rows = score_images(cats, images, per_image)
    picked = select(rows, config.N_IMAGES)
    picked = add_probes(picked, sorted(set(cats.values())), freq)

    config.DATA.mkdir(parents=True, exist_ok=True)
    draft = config.DATA / "eval_set_draft.json"
    draft.write_text(json.dumps(picked, indent=2))
    print(f"{len(picked)} candidates -> {draft}")
    print(f"  mean instances {sum(r['n_instances'] for r in picked) / len(picked):.1f}, "
          f"mean difficulty {sum(r['difficulty'] for r in picked) / len(picked):.3f}")
    print("  NOT an evaluation set yet: every probe still needs visual confirmation.")


if __name__ == "__main__":
    main()
