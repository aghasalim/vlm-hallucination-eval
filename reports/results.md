# Results

**33 hand-verified images**, 172 objects verified present and 67 verified absent (44 adversarial, 21 random, 2 popular), 239 yes/no probes in total. Mean 23 annotated object instances per image.

Model: `Salesforce/blip-vqa-base` for answering, `Salesforce/blip-image-captioning-base` for captions, `openai/clip-vit-base-patch32` for verification. Beam search, so every number here is deterministic and reproducible.

## 1. Baseline, the surprise

| question phrasing | accuracy | yes-rate | recall on present | hallucination (adversarial) | hallucination (all absent) |
|---|---|---|---|---|---|
| neutral | 77.4% | 52.7% | 70.9% | 4.5% | 6.0% |
| leading | 78.2% | 57.7% | 75.0% | 6.8% | 13.4% |
| presupposing | 76.2% | 54.0% | 70.9% | 6.8% | 10.4% |

The headline finding is not the one this project set out to measure. BLIP-VQA **barely hallucinates** on this set, 4.5% on adversarial probes with neutral phrasing. Its dominant error is the opposite: it **misses 29% of objects that are genuinely present**. It is over-cautious, not over-confident.

The caption model makes the same trade even more starkly. Zero of 33 captions mentioned a verified-absent object, but only because the captions are too vague to be wrong. On a shelf of 30-odd ceramic pieces it produced *"a shelf filled with lots of different colored dishes"*; on a cluttered living room, *"the floor is made of wood"*. A caption that commits to nothing cannot hallucinate, and is also useless. A hallucination rate of 0% is not evidence of grounding.

## 2. Phrasing more than doubles hallucination

Same model, same images, same objects, only the wording of the question changes:

| phrasing | example | hallucination (all absent) |
|---|---|---|
| neutral | *Is there a spoon in the image?* | 6.0% |
| presupposing | *Can you see the spoon in this image?* | 10.4% |
| leading | *The image contains a spoon, correct?* | 13.4% |

Smuggling the object into the premise more than doubles the hallucination rate. Any reported hallucination number is a property of the prompt as much as the model, which is worth remembering when comparing published figures.

## 3. Verification: before and after

CLIP z-score threshold fitted on 11 calibration images, reported on the 22 held-out images. The split is by image, not by probe, probes from one image share a CLIP score vector, so a probe-level split would leak.

| phrasing | rule | accuracy | precision | recall | F1 | hallucination |
|---|---|---|---|---|---|---|
| neutral | baseline (VLM alone) | 77.8% | 0.964 | 0.721 | 0.825 | 7.1% |
| neutral | CLIP alone | 74.5% | 0.769 | 0.928 | 0.841 | 73.8% |
| neutral | VLM AND CLIP | 75.2% | 0.974 | 0.676 | 0.798 | 4.8% |
| leading | baseline (VLM alone) | 81.0% | 0.946 | 0.784 | 0.857 | 11.9% |
| leading | CLIP alone | 74.5% | 0.769 | 0.928 | 0.841 | 73.8% |
| leading | VLM AND CLIP | 79.1% | 0.965 | 0.739 | 0.837 | 7.1% |
| presupposing | baseline (VLM alone) | 77.8% | 0.953 | 0.730 | 0.827 | 9.5% |
| presupposing | CLIP alone | 74.5% | 0.769 | 0.928 | 0.841 | 73.8% |
| presupposing | VLM AND CLIP | 75.2% | 0.962 | 0.685 | 0.800 | 7.1% |

Verification cuts hallucination from 7.1% to 4.8% on neutral phrasing and from 11.9% to 7.1% on leading phrasing. The relative reduction is neutral 33%, presupposing 25%, leading 40%, so the honest range is 25% to 40% and it depends on the phrasing. In counts that is 1 to 2 fewer false positives out of 42 verified-absent probes. **It is not free.** Recall falls from 0.721 to 0.676, and F1 actually *drops* (0.825 → 0.798). Because the rule is a logical AND it can only ever delete a "yes", so on a model that already under-reports, it makes the larger problem worse.

**CLIP alone is a bad verifier.** Thresholding CLIP on its own hallucinates on 73.8% of verified-absent objects. CLIP is trained to match images to plausible captions, not to certify that something is missing, and a photo of a pottery shelf is genuinely a decent match for "a photo of a spoon". It is only useful here as a veto on top of the VLM.

### The trade-off in full

![trade-off](figures/tradeoff.png)

| z threshold | hallucination | recall | F1 |
|---|---|---|---|
| -1.5 | 7.1% | 72.1% | 0.825 |
| -0.4 | 4.8% | 67.6% | 0.798 |
| 0.0 | 4.8% | 64.9% | 0.778 |
| 0.5 | 2.4% | 55.9% | 0.713 |
| 1.0 | 0.0% | 43.2% | 0.604 |
| 1.5 | 0.0% | 31.5% | 0.479 |

Hallucination can be driven to **exactly zero**, at a cost of roughly 22 points of recall. Whether that is a good trade is an application question, not a modelling one: it is the right call for generating radiology reports and the wrong one for alt-text.

## 4. What this measurement cannot tell you

- **The eval set is biased toward verifiable scenes.** Absence has to be certain   to be ground truth, and absence is hardest to certify in exactly the cluttered   indoor scenes where VLMs fail most. Kitchens were dropped because a fork is   plausible everywhere. The measured rates are therefore probably optimistic.
- **One model family.** BLIP is an encoder-decoder trained on VQA, not an   LLM-based VLM. The published hallucination results that motivated this project   mostly target LLaVA/MiniGPT-style models, where a language prior can override   the image. This result should not be read as "VLMs don't hallucinate".
- **Small numbers.** 153 probes on 22   held-out images. A hallucination rate of 4.8% is 2 errors. Treat single   percentage points as noise.
- **The popular split has 2 items** and is reported only   for completeness; it is too small to draw anything from.
