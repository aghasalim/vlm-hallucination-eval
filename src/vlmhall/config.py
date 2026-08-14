"""Configuration and the confusable-object taxonomy the adversarial set is built on."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
IMAGES = DATA / "images"
ANNOTATIONS = DATA / "annotations"
EVAL_SET = DATA / "eval_set.json"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
VQA_MODEL = os.getenv("VQA_MODEL", "Salesforce/blip-vqa-base")
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

N_IMAGES = int(os.getenv("N_IMAGES", "45"))
SEED = 20260814

# COCO categories grouped by what they can plausibly be mistaken for. This is the
# whole basis of the "adversarial" probe split: asking whether a picture of a dog
# contains a *toothbrush* measures almost nothing, because the answer is obvious
# from context alone. Asking whether it contains a *cat* probes whether the model
# is actually looking. Groups are hand-written, not derived from co-occurrence
# statistics, because co-occurrence would pair things that appear together
# (person+tie) rather than things that get confused for each other.
CONFUSABLE_GROUPS: dict[str, list[str]] = {
    "pets_and_livestock": ["dog", "cat", "horse", "sheep", "cow", "bear", "zebra",
                           "giraffe", "elephant"],
    "birds": ["bird"],
    "road_vehicles": ["car", "truck", "bus", "motorcycle", "bicycle"],
    "large_vehicles": ["train", "boat", "airplane"],
    "seating": ["chair", "couch", "bench", "bed"],
    "drinkware": ["cup", "wine glass", "bottle"],
    # No "plate": it is not one of COCO's 80 categories, so it can never appear
    # in the annotations and every image would look as though it had no plate.
    # Probing for it would measure COCO's label vocabulary, not hallucination.
    # Same reason "pillow" and "blanket" are absent from the seating/soft groups.
    "tableware": ["bowl", "fork", "knife", "spoon"],
    "screens": ["tv", "laptop", "cell phone"],
    "computer_peripherals": ["keyboard", "mouse", "remote"],
    "round_fruit": ["apple", "orange"],
    "baked_goods": ["donut", "cake", "sandwich", "pizza", "hot dog"],
    "board_sports": ["skateboard", "surfboard", "snowboard", "skis"],
    "racket_sports": ["tennis racket", "baseball bat", "baseball glove"],
    "bags": ["backpack", "handbag", "suitcase"],
    "kitchen_appliances": ["microwave", "oven", "toaster", "refrigerator", "sink"],
    "small_animals_wild": ["zebra", "giraffe"],
    "signage": ["stop sign", "traffic light", "parking meter"],
}

# Probe splits, following the POPE protocol (Li et al., EMNLP 2023) so the numbers
# are comparable to published work instead of being a private metric.
PROBE_SPLITS = ("random", "popular", "adversarial")

# Verification threshold is calibrated on a disjoint split, never on the test set.
CALIBRATION_FRAC = 0.35
