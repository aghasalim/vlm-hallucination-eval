"""Integrity checks on the hand-built evaluation set and the caption matcher.

These guard the two things that would silently corrupt every reported number: a
probe that contradicts the annotations, and an object-mention matcher that
quietly fails to match.
"""
from __future__ import annotations

import json

import pytest

from src.vlmhall import config
from src.vlmhall.evaluate import COCO_80, SYNONYMS, mentioned_objects, question_for


@pytest.fixture(scope="module")
def rows():
    return json.loads(config.EVAL_SET.read_text())


def test_no_probe_is_both_present_and_absent(rows):
    for r in rows:
        clash = set(r["absent"]) & set(r["present"])
        assert not clash, f"{r['file_name']}: {clash} marked both present and absent"


def test_every_probe_is_a_real_coco_category(rows):
    """A probe outside COCO's 80 categories can never be verified against the
    annotations, this is what caught 'plate' during the build."""
    for r in rows:
        for obj in list(r["absent"]) + r["present"]:
            assert obj in COCO_80, f"{r['file_name']}: '{obj}' is not a COCO category"


def test_confusable_groups_only_contain_coco_categories():
    for group, members in config.CONFUSABLE_GROUPS.items():
        for m in members:
            assert m in COCO_80, f"group {group}: '{m}' is not a COCO category"


def test_synonyms_map_into_coco_categories():
    for surface, canon in SYNONYMS.items():
        assert canon in COCO_80, f"synonym '{surface}' maps to non-category '{canon}'"


def test_set_is_large_enough_and_mostly_adversarial(rows):
    n_absent = sum(len(r["absent"]) for r in rows)
    n_adv = sum(1 for r in rows for k in r["absent"].values() if k == "adversarial")
    assert len(rows) >= 30, "brief asks for 30-50 images"
    assert n_absent >= 50
    assert n_adv / n_absent > 0.5, "the set should be mostly confusable probes"


def test_images_exist(rows):
    for r in rows:
        assert (config.IMAGES / r["file_name"]).exists(), f"missing {r['file_name']}"


@pytest.mark.parametrize("caption,expected", [
    ("a man riding a skateboard", {"person", "skateboard"}),
    ("two dogs playing in a field", {"dog"}),
    ("a sofa next to a television", {"couch", "tv"}),
    ("a plate of hotdogs", {"hot dog"}),
])
def test_mention_extraction(caption, expected):
    assert expected <= mentioned_objects(caption)


def test_hotdog_does_not_match_dog():
    """The naive plural rule must not let 'hotdog' register as 'dog'."""
    assert "dog" not in mentioned_objects("a hotdog on a bun")


def test_question_articles():
    assert question_for("spoon") == "Is there a spoon in the image?"
    assert question_for("apple") == "Is there an apple in the image?"
    assert "correct?" in question_for("spoon", "leading")
