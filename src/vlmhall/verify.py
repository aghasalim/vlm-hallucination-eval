"""CLIP-based grounding verification.

The problem with using raw CLIP similarity as a truth test is that the scale is
not comparable across objects or images: "a photo of a person" scores highly
against almost anything, and every score for a dim monochrome photo sits lower
than every score for a bright one. A single global threshold on raw similarity
therefore mostly measures prompt frequency and image brightness.

So each object is scored *relative to the same image's own distribution*: we
embed all 80 COCO prompts against the image once and take the z-score of the
object of interest. That makes the score "how much does this image stand out for
this object, compared with everything else it could contain", which is
per-image-normalised and directly comparable.

The threshold is fitted on a calibration split of images and reported on the
held-out remainder. Fitting and reporting on the same images would make any
improvement meaningless.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from . import config, models
from .evaluate import COCO_80


def prompt(obj: str) -> str:
    return f"a photo of a {obj}"


def image_z_scores(img: Image.Image) -> dict[str, float]:
    """z-scored CLIP similarity for every COCO category against one image."""
    sims = np.array(models.clip_scores(img, [prompt(o) for o in COCO_80]))
    z = (sims - sims.mean()) / (sims.std() + 1e-8)
    return dict(zip(COCO_80, z.tolist()))


def score_all(rows: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for i, r in enumerate(rows, 1):
        img = Image.open(config.IMAGES / r["file_name"]).convert("RGB")
        out[r["file_name"]] = image_z_scores(img)
        if i % 10 == 0:
            print(f"  scored {i}/{len(rows)}")
    return out


def split_images(rows: list[dict]) -> tuple[set[str], set[str]]:
    """Deterministic image-level split. Splitting by *image* not by probe matters:
    probes from one image share a CLIP score vector, so a probe-level split would
    leak calibration information into the test set."""
    files = sorted(r["file_name"] for r in rows)
    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(files))
    n_cal = max(1, int(len(files) * config.CALIBRATION_FRAC))
    cal = {files[i] for i in idx[:n_cal]}
    return cal, set(files) - cal


def best_threshold(probes: list[dict], z: dict, files: set[str]) -> tuple[float, float]:
    """Pick the z threshold maximising F1 on the calibration images."""
    subset = [p for p in probes if p["file_name"] in files]
    best_t, best_f1 = 0.0, -1.0
    for t in np.arange(-1.0, 3.01, 0.05):
        tp = sum(1 for p in subset if p["truth"] and z[p["file_name"]][p["object"]] >= t)
        fp = sum(1 for p in subset if not p["truth"] and z[p["file_name"]][p["object"]] >= t)
        fn = sum(1 for p in subset if p["truth"] and z[p["file_name"]][p["object"]] < t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def apply(probes: list[dict], z: dict, threshold: float, files: set[str]) -> dict:
    """Score three decision rules on the held-out images.

    baseline  - the VLM's own yes/no
    clip_only - CLIP's z-score thresholded, ignoring the VLM
    verified  - accept only if the VLM says yes AND CLIP agrees (logical AND)

    The AND rule can only ever remove "yes" answers, so it cannot invent new
    hallucinations -- but it also cannot recover the objects the VLM missed, and
    it will destroy true positives that CLIP scores low. Both effects are
    reported, because a verifier that removes hallucinations by rejecting
    everything is worthless.
    """
    subset = [p for p in probes if p["file_name"] in files]

    def summarise(name: str, pred_fn) -> dict:
        tp = sum(1 for p in subset if p["truth"] and pred_fn(p))
        fp = sum(1 for p in subset if not p["truth"] and pred_fn(p))
        fn = sum(1 for p in subset if p["truth"] and not pred_fn(p))
        tn = sum(1 for p in subset if not p["truth"] and not pred_fn(p))
        n_absent = sum(1 for p in subset if not p["truth"])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return {
            "rule": name, "n": len(subset),
            "accuracy": (tp + tn) / len(subset),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "hallucination_rate": fp / n_absent if n_absent else 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    return {
        "threshold": threshold,
        "baseline": summarise("baseline (VLM alone)", lambda p: p["pred"] is True),
        "clip_only": summarise("CLIP alone",
                               lambda p: z[p["file_name"]][p["object"]] >= threshold),
        "verified": summarise("VLM AND CLIP",
                              lambda p: p["pred"] is True
                              and z[p["file_name"]][p["object"]] >= threshold),
    }


def tradeoff(probes: list[dict], z: dict, files: set[str]) -> list[dict]:
    """Sweep the threshold to expose what verification actually costs.

    Reporting one operating point hides the trade: the AND rule only ever deletes
    "yes" answers, so hallucination falls monotonically and recall falls with it.
    The useful question is not "does it help" but "how much recall does a given
    hallucination reduction cost", and that needs the whole curve.
    """
    subset = [p for p in probes if p["file_name"] in files]
    n_absent = sum(1 for p in subset if not p["truth"])
    n_present = sum(1 for p in subset if p["truth"])
    rows = []
    for t in np.arange(-1.5, 3.01, 0.1):
        keep = lambda p: p["pred"] is True and z[p["file_name"]][p["object"]] >= t  # noqa: E731
        tp = sum(1 for p in subset if p["truth"] and keep(p))
        fp = sum(1 for p in subset if not p["truth"] and keep(p))
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / n_present if n_present else 0.0
        rows.append({"threshold": round(float(t), 2),
                     "hallucination_rate": fp / n_absent if n_absent else 0.0,
                     "recall": rec, "precision": prec,
                     "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})
    return rows


def main() -> None:
    rows = json.loads(config.EVAL_SET.read_text())
    styles = json.loads((config.REPORTS / "prompt_styles.json").read_text())

    print("scoring images with CLIP ...")
    z = score_all(rows)
    cal, test = split_images(rows)
    print(f"  calibration images {len(cal)}, held-out test images {len(test)}")

    results = {}
    for style, payload in styles.items():
        probes = payload["probes"]
        t, cal_f1 = best_threshold(probes, z, cal)
        res = apply(probes, z, t, test)
        res["calibration_f1"] = cal_f1
        res["tradeoff"] = tradeoff(probes, z, test)
        results[style] = res
        print(f"\n=== {style} (threshold z >= {t:.2f}, fitted on calibration) ===")
        for key in ("baseline", "clip_only", "verified"):
            r = res[key]
            print(f"  {r['rule']:<22} acc {r['accuracy']:6.1%}  P {r['precision']:.3f}  "
                  f"R {r['recall']:.3f}  F1 {r['f1']:.3f}  "
                  f"hallucination {r['hallucination_rate']:6.1%}")

    (config.REPORTS / "verification.json").write_text(
        json.dumps({"results": results,
                    "calibration_images": sorted(cal),
                    "test_images": sorted(test),
                    "z_scores": z}, indent=2))
    print(f"\n-> {config.REPORTS / 'verification.json'}")


if __name__ == "__main__":
    main()
