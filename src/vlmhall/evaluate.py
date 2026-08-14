"""Baseline hallucination measurement.

Two complementary measurements, because they fail differently:

*Probe-level (POPE-style).* Ask "Is there a {object} in the image?" for objects
verified present and objects verified absent. A "yes" on a verified-absent object
is a hallucination. This isolates object existence from language fluency, and the
yes-rate exposes the acquiescence bias that makes accuracy alone misleading -- a
model that answers "yes" to everything scores 100% recall.

*Caption-level (CHAIR-style).* Generate a free caption, extract the COCO objects
it mentions, and count the ones that are not in the image. This measures what the
model volunteers unprompted, which is the failure mode a user actually sees.
"""
from __future__ import annotations

import json
import re
import time

from . import config

# `models` is imported lazily inside run(), not here. Everything else in this
# module -- the COCO vocabulary, the synonym table, mention extraction, the
# metrics -- is pure Python, and importing it should not require torch. That
# keeps the eval-set integrity tests runnable (and CI fast) without model weights.

# Synonyms mapping surface forms in captions back to COCO category names. Without
# this, "sofa" in a caption never matches the "couch" annotation and the caption
# metric silently under-reports both hits and hallucinations.
SYNONYMS: dict[str, str] = {
    "sofa": "couch", "settee": "couch", "tv": "tv", "television": "tv",
    "monitor": "tv", "screen": "tv", "cellphone": "cell phone",
    "mobile phone": "cell phone", "phone": "cell phone",
    "motorbike": "motorcycle", "bike": "bicycle", "cycle": "bicycle",
    "plane": "airplane", "aeroplane": "airplane", "aircraft": "airplane",
    "jet": "airplane", "lorry": "truck", "van": "truck", "auto": "car",
    "automobile": "car", "table": "dining table", "desk": "dining table",
    "doughnut": "donut", "hotdog": "hot dog", "puppy": "dog", "kitten": "cat",
    "cattle": "cow", "calf": "cow", "lamb": "sheep", "pony": "horse",
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "people": "person", "men": "person", "women": "person",
    "player": "person", "skier": "person", "surfer": "person", "rider": "person",
    "glass": "wine glass", "mug": "cup", "plant": "potted plant",
    "fridge": "refrigerator", "stove": "oven", "computer": "laptop",
    "remote control": "remote", "purse": "handbag", "bag": "handbag",
    "luggage": "suitcase", "racket": "tennis racket", "bat": "baseball bat",
    "glove": "baseball glove", "ball": "sports ball", "board": "surfboard",
    "signal": "traffic light", "stoplight": "traffic light",
}

COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def mentioned_objects(text: str) -> set[str]:
    """COCO categories a caption mentions, after synonym normalisation."""
    t = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    t = re.sub(r"\s+", " ", t)
    found = set()
    for surface, canon in list(SYNONYMS.items()) + [(c, c) for c in COCO_80]:
        # Word-boundary match plus a naive plural, so "dogs" counts as "dog"
        # but "hotdog" does not count as "dog".
        if re.search(rf" {re.escape(surface)}(s|es)? ", t):
            found.add(canon)
    return found


# Question phrasings, from neutral to presupposing. The baseline run showed BLIP
# is *conservative* rather than hallucination-prone, so the obvious next question
# is whether that survives a question that smuggles in the object's existence.
# Acquiescence to a false premise is the mechanism POPE attributes most object
# hallucination to, and it costs nothing to test.
QUESTION_STYLES = {
    "neutral": "Is there {art} {obj} in the image?",
    "leading": "The image contains {art} {obj}, correct?",
    "presupposing": "Can you see the {obj} in this image?",
}


def question_for(obj: str, style: str = "neutral") -> str:
    art = "an" if obj[0] in "aeiou" else "a"
    return QUESTION_STYLES[style].format(art=art, obj=obj)


def run(limit: int | None = None, style: str = "neutral") -> dict:
    from PIL import Image

    from . import models

    rows = json.loads(config.EVAL_SET.read_text())[:limit]
    probes, captions = [], []
    t0 = time.time()

    for i, r in enumerate(rows, 1):
        img = Image.open(config.IMAGES / r["file_name"]).convert("RGB")

        for obj in r["present"]:
            ans = models.ask(img, question_for(obj, style))
            probes.append({"file_name": r["file_name"], "object": obj,
                           "truth": True, "kind": "present",
                           "answer": ans, "pred": models.yes_no(ans)})
        for obj, kind in r["absent"].items():
            ans = models.ask(img, question_for(obj, style))
            probes.append({"file_name": r["file_name"], "object": obj,
                           "truth": False, "kind": kind,
                           "answer": ans, "pred": models.yes_no(ans)})

        cap = models.caption(img)
        mentioned = mentioned_objects(cap)
        present = set(r["present"])
        # Only objects we can adjudicate count: a caption mentioning "sand" is
        # neither right nor wrong here, and COCO cannot rule most nouns in or out.
        # Hallucination is scored strictly against objects verified absent.
        hallucinated = sorted(mentioned & set(r["absent"]))
        captions.append({"file_name": r["file_name"], "caption": cap,
                         "mentioned": sorted(mentioned),
                         "grounded": sorted(mentioned & present),
                         "hallucinated": hallucinated})
        print(f"  [{i}/{len(rows)}] {r['file_name']}: {cap[:60]}")

    print(f"  {len(probes)} probes in {time.time() - t0:.0f}s")
    return {"probes": probes, "captions": captions}


def metrics(probes: list[dict]) -> dict:
    def summarise(subset: list[dict]) -> dict:
        n = len(subset)
        if not n:
            return {}
        unparsed = sum(1 for p in subset if p["pred"] is None)
        tp = sum(1 for p in subset if p["truth"] and p["pred"] is True)
        fn = sum(1 for p in subset if p["truth"] and p["pred"] is not True)
        fp = sum(1 for p in subset if not p["truth"] and p["pred"] is True)
        tn = sum(1 for p in subset if not p["truth"] and p["pred"] is not True)
        acc = (tp + tn) / n
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        yes = sum(1 for p in subset if p["pred"] is True) / n
        return {"n": n, "accuracy": acc, "precision": prec, "recall": rec,
                "f1": f1, "yes_rate": yes, "unparsed": unparsed,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    present = [p for p in probes if p["truth"]]
    out = {"overall": summarise(probes), "present": summarise(present)}
    for kind in ("adversarial", "random", "popular"):
        subset = [p for p in probes if p["kind"] == kind]
        if subset:
            # Hallucination rate on an absent-only subset is just the yes-rate.
            s = summarise(present + subset)
            s["hallucination_rate"] = sum(1 for p in subset if p["pred"] is True) / len(subset)
            s["n_absent"] = len(subset)
            out[kind] = s
    absent = [p for p in probes if not p["truth"]]
    out["absent_all"] = {"n": len(absent),
                         "hallucination_rate": sum(1 for p in absent if p["pred"] is True) / len(absent)}
    return out


def main() -> None:
    res = run()
    res["metrics"] = metrics(res["probes"])
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "baseline.json").write_text(json.dumps(res, indent=2))

    m = res["metrics"]
    print("\n=== baseline (BLIP-VQA, no verification) ===")
    print(f"  overall accuracy {m['overall']['accuracy']:.1%}  "
          f"F1 {m['overall']['f1']:.3f}  yes-rate {m['overall']['yes_rate']:.1%}")
    print(f"  recall on present objects   {m['present']['recall']:.1%}")
    for k in ("adversarial", "random", "popular"):
        if k in m:
            print(f"  hallucination rate ({k:<11}) {m[k]['hallucination_rate']:.1%} "
                  f"of {m[k]['n_absent']}")
    n_cap = sum(1 for c in res["captions"] if c["hallucinated"])
    print(f"  captions containing a verified-absent object: {n_cap}/{len(res['captions'])}")
    print(f"\n-> {config.REPORTS / 'baseline.json'}")


if __name__ == "__main__":
    main()
