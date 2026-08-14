# Hallucination-aware captioning — with an adversarial set I built by hand

[![ci](https://github.com/aghasalim/vlm-hallucination-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/vlm-hallucination-eval/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**[▶ Live demo](https://vlm-hallucination-eval.streamlit.app/)** — upload an image
or pick one from the adversarial set, and see the per-claim grounding scores.

A vision-language system that captions images and answers questions about them,
with a CLIP-based layer that checks whether each claim is actually grounded in the
image — and a hand-verified adversarial evaluation set built to catch it when it
isn't.

I built this expecting to find a model that confidently describes things that
aren't there. That is what the literature says vision-language models do. **I found
the opposite, and I think the thing I found is more interesting.**

---

## What I actually found

**1. The model barely hallucinates — it under-reports instead.**
On 44 hand-verified adversarial probes, BLIP-VQA claimed to see an absent object
**4.5%** of the time. But it **missed 29% of the objects that were genuinely
there**. It is over-cautious, not over-confident. Every fix I had planned was
aimed at the wrong failure.

**2. A caption that can't be wrong isn't grounded — it's just vague.**
Zero of 33 captions mentioned a verified-absent object. That sounds like a perfect
score. It isn't. Shown a shelf holding thirty-odd pieces of ceramic, the model
wrote *"a shelf filled with lots of different colored dishes"*. Shown a cluttered
living room: *"the floor is made of wood"*. **A 0% hallucination rate is not
evidence of grounding** when the captions commit to nothing.

**3. How you ask matters more than I expected.** Same model, same images, same
objects — only the wording changes:

| phrasing | example | hallucination rate |
|---|---|---|
| neutral | *Is there a spoon in the image?* | 6.0% |
| presupposing | *Can you see the spoon in this image?* | 10.4% |
| leading | *The image contains a spoon, correct?* | **13.4%** |

Smuggling the object into the premise **more than doubles** the hallucination
rate. Any published hallucination number is partly a property of the prompt.

**4. Verification works, and it is not free.**

| phrasing | rule | recall | F1 | hallucination |
|---|---|---|---|---|
| neutral | BLIP alone | 0.721 | 0.825 | 7.1% |
| neutral | **BLIP AND CLIP** | 0.676 | 0.798 | **4.8%** |
| leading | BLIP alone | 0.784 | 0.857 | 11.9% |
| leading | **BLIP AND CLIP** | 0.739 | 0.837 | **7.1%** |
| — | CLIP alone | 0.928 | 0.841 | **73.8%** |

Verification cuts hallucination by 33–40% relative. It also costs recall, and **F1
actually drops**. The rule is a logical AND, so it can only ever delete a "yes" —
on a model that already under-reports, it makes the bigger problem worse. I'd
rather show that than quietly report only the number that improved.

**CLIP on its own is a bad verifier** — 73.8% hallucination. CLIP is trained to
match images to plausible captions, not to certify that something is *missing*, and
a photo of a pottery shelf is honestly a decent match for "a photo of a spoon". It
only works as a veto on top of the VLM.

You can push hallucination to **exactly zero** — it costs about 29 points of recall:

| CLIP z threshold | hallucination | recall | F1 |
|---|---|---|---|
| −1.5 (off) | 7.1% | 72.1% | 0.825 |
| −0.4 (fitted) | 4.8% | 67.6% | 0.798 |
| 0.5 | 2.4% | 55.9% | 0.713 |
| 1.0 | **0.0%** | 43.2% | 0.604 |

Whether that's a good trade is an application question, not a modelling one. It's
right for radiology reports and wrong for alt-text.

Full numbers, including the third phrasing and the whole curve: **[reports/results.md](reports/results.md)**.

---

## The evaluation set

33 images, **172 objects verified present and 67 verified absent** (44 of them
adversarial), so 239 yes/no probes. Mean 23 annotated object instances per image —
these are cluttered scenes, not stock photos.

I selected candidates automatically from COCO val2017 by clutter, object smallness
and bounding-box overlap, then **looked at every image myself** and decided each
probe individually. Three rules I held to:

**Absence has to be visually certain, not merely unannotated.** COCO labels 80
categories and misses instances even within them, so "not in the annotations" is
not evidence of absence. This is a real weakness in benchmarks built automatically
from COCO, and it bit me immediately.

**Adversarial means confusable, not just absent.** Asking whether a ski slope
contains a toothbrush measures nothing. Asking whether it contains a *skateboard*
probes whether the model distinguishes board sports. Probes are grouped by
confusability, so a sheep pasture gets asked about horses and bears.

**Ambiguous items were deleted, not guessed.** A playroom with a dog would have
been a lovely "bear" probe — but COCO separates "bear" from "teddy bear", and a toy
would make "yes" defensible. An eval item whose correct answer is arguable is worse
than no eval item.

### Two things this caught

Building it by hand caught a bug that would have silently corrupted every number:
**"plate" is not one of COCO's 80 categories.** My confusable-object taxonomy
listed it, so every image would have looked as though it had no plate, and I'd have
been measuring COCO's label vocabulary rather than hallucination. There is now
[a test](tests/test_eval_set.py) that fails if any probe drifts outside the 80.

It also forced an honest admission about scope, below.

---

## What this can't tell you

- **The set is biased toward verifiable scenes.** Absence must be certain to serve
  as ground truth, and absence is hardest to certify in exactly the cluttered
  indoor scenes where VLMs fail most. I dropped cutlery probes for kitchens because
  a fork is plausible everywhere. **The rates here are probably optimistic**, and
  the images I had to discard are the ones that would have hurt the model most.
- **One model family.** BLIP is an encoder-decoder trained on VQA. The results that
  motivated this project mostly target LLaVA/MiniGPT-style models, where a language
  prior can override the image. Do not read this as "VLMs don't hallucinate" — read
  it as "this one, on these images, does something different".
- **Small numbers.** 153 probes over 22 held-out images, 42 of them verified-absent.
  4.8% is two errors. Single percentage points are noise.
- The `popular` probe split has only 2 verified items and is reported for
  completeness only.

---

## Running it

```bash
make setup && make data && make evalset
```

```bash
make baseline && make verify && make report
```

That reproduces every number above. No API key, no paid service — BLIP and CLIP run
locally on CPU (or Apple MPS); the full evaluation is about 4 minutes on a laptop.

```bash
make app
```

Or build the container, which bakes the weights in so a cold start doesn't spend a
minute downloading:

```bash
docker build -t vlm-hallucination-eval . && docker run -p 8501:8501 vlm-hallucination-eval
```

---

## How it works

**Generation** — `Salesforce/blip-vqa-base` answers, `blip-image-captioning-base`
captions. Beam search, not sampling: a hallucination rate that changes between runs
isn't a measurement.

**Verification** — raw CLIP similarity is useless as a truth test, because the
scale isn't comparable across objects or images ("a photo of a person" scores well
against nearly anything, and every score on a dim photo sits below every score on a
bright one). So each object is scored **relative to the same image's own
distribution**: embed all 80 COCO prompts against the image, take the z-score of
the object in question. That answers "how much does this image stand out for this
object, versus everything else it could contain".

**Calibration** — the threshold is fitted on 11 calibration images and reported on
22 held-out ones. The split is **by image, not by probe**: probes from one image
share a CLIP score vector, so a probe-level split would leak.

**Measurement** — probe-level scoring follows the POPE protocol (Li et al., EMNLP
2023) so the numbers are comparable to published work, plus a CHAIR-style pass over
free captions for what the model volunteers unprompted.

```
src/vlmhall/
  build_set.py            candidate selection from COCO (clutter, size, occlusion)
  contact_sheets.py       renders sheets for the manual pass
  manual_verification.py  my per-image verdicts — this file IS the eval set
  models.py               BLIP + CLIP wrappers
  evaluate.py             probe and caption metrics, phrasing variants
  verify.py               CLIP z-scoring, calibration, trade-off sweep
  report.py               builds reports/results.md and the figure
```

---

## Licence

MIT — see [LICENSE](LICENSE). COCO images are not redistributed here; `make data`
fetches them from cocodataset.org. `data/eval_set.json` contains only image ids and
my own annotations.
