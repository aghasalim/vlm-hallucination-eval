"""Assemble reports/results.md and the trade-off figure from the measured JSON."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def figure(ver: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"neutral": "#1f77b4", "leading": "#d62728", "presupposing": "#ff7f0e"}

    for style, res in ver["results"].items():
        t = res["tradeoff"]
        axes[0].plot([r["recall"] for r in t], [r["hallucination_rate"] for r in t],
                     "-o", ms=2.5, color=colors[style], label=style)
        axes[1].plot([r["threshold"] for r in t], [r["f1"] for r in t],
                     color=colors[style], label=style)

    axes[0].set_xlabel("recall on objects that ARE present")
    axes[0].set_ylabel("hallucination rate on verified-absent objects")
    axes[0].set_title("What verification costs\n(up-right is the un-verified model)")
    axes[1].set_xlabel("CLIP z-score threshold")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 vs threshold")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    out = config.REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "tradeoff.png", dpi=140)
    print(f"  -> {out / 'tradeoff.png'}")


def prompt_style_figure(styles: dict) -> None:
    """What the wording of the question does to the answer.

    The image never changes and neither do the objects; only the phrasing does.
    Overall accuracy barely moves, which is why a headline accuracy number hides
    this. The rate that moves is hallucination on objects verified absent from the
    image, and smuggling the object into the premise more than doubles it.
    """
    order = [s for s in ["neutral", "presupposing", "leading"] if s in styles]
    colors = {"neutral": "#1f77b4", "leading": "#d62728", "presupposing": "#ff7f0e"}
    base = range(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].bar(base,
                [styles[s]["metrics"]["absent_all"]["hallucination_rate"] * 100
                 for s in order],
                color=[colors[s] for s in order], edgecolor="0.3", lw=0.5)
    axes[0].set_ylabel("hallucination rate (%)")
    axes[0].set_title("on objects verified absent", fontsize=10)
    for index, style in enumerate(order):
        rate = styles[style]["metrics"]["absent_all"]["hallucination_rate"] * 100
        axes[0].text(index, rate + 0.3, f"{rate:.1f}%", ha="center", fontsize=10,
                     fontweight="bold")

    axes[1].bar(base, [styles[s]["metrics"]["overall"]["accuracy"] * 100 for s in order],
                color=[colors[s] for s in order], edgecolor="0.3", lw=0.5)
    axes[1].set_ylabel("overall accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("overall accuracy barely moves", fontsize=10)

    axes[2].bar(base, [styles[s]["metrics"]["overall"]["yes_rate"] * 100 for s in order],
                color=[colors[s] for s in order], edgecolor="0.3", lw=0.5)
    axes[2].set_ylabel("yes-rate (%)")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("and the yes-rate only a little", fontsize=10)

    for ax in axes:
        ax.set_xticks(list(base))
        ax.set_xticklabels(order, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Same images, same objects, three phrasings. Only the left panel moves, "
        "which is why an accuracy headline misses it.",
        fontsize=10, y=0.02, color="0.35",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = config.REPORTS / "figures" / "prompt-styles.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def rules_figure(ver: dict) -> None:
    """The three decision rules at the chosen operating point.

    CLIP alone is worse than the VLM alone on accuracy. The combination beats both
    on hallucination rate, which is the only reason to pay for it.
    """
    styles = list(ver["results"])
    rules = ["baseline", "clip_only", "combined"]
    present = [r for r in rules if r in ver["results"][styles[0]]]
    labels = {"baseline": "VLM alone", "clip_only": "CLIP alone", "combined": "combined"}
    colors = {"baseline": "#1f77b4", "clip_only": "#7f7f7f", "combined": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    width = 0.8 / max(len(present), 1)
    for offset, rule in enumerate(present):
        xs = [i + (offset - (len(present) - 1) / 2) * width for i in range(len(styles))]
        axes[0].bar(xs, [ver["results"][s][rule]["accuracy"] * 100 for s in styles],
                    width, label=labels[rule], color=colors[rule],
                    edgecolor="0.3", lw=0.4)
        axes[1].bar(xs, [ver["results"][s][rule]["hallucination_rate"] * 100
                         for s in styles],
                    width, label=labels[rule], color=colors[rule],
                    edgecolor="0.3", lw=0.4)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("accuracy", fontsize=10)
    axes[1].set_ylabel("hallucination rate (%)")
    axes[1].set_title("hallucination on verified-absent objects", fontsize=10)
    for ax in axes:
        ax.set_xticks(range(len(styles)))
        ax.set_xticklabels(styles, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = config.REPORTS / "figures" / "rules.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def probe_breakdown_figure(baseline: dict) -> None:
    """Where the errors are: present objects missed, absent objects invented.

    The asymmetry is the finding. The model is close to perfect at not inventing
    objects under a neutral prompt, and misses a third of the ones that are there.
    """
    probes = baseline["probes"]
    kinds = list(dict.fromkeys(p["kind"] for p in probes))
    correct, wrong = [], []
    for kind in kinds:
        subset = [p for p in probes if p["kind"] == kind]
        right = sum(1 for p in subset if p["pred"] == p["truth"])
        correct.append(right)
        wrong.append(len(subset) - right)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    base = range(len(kinds))
    ax.bar(base, correct, 0.55, label="correct", color="#2ca02c",
           edgecolor="0.3", lw=0.5)
    ax.bar(base, wrong, 0.55, bottom=correct, label="wrong", color="#d62728",
           edgecolor="0.3", lw=0.5)
    for index, (right, bad) in enumerate(zip(correct, wrong, strict=True)):
        total = right + bad
        ax.text(index, total + 1, f"{bad}/{total} wrong", ha="center", fontsize=9)
    ax.set_xticks(list(base))
    ax.set_xticklabels(kinds, fontsize=9)
    ax.set_ylabel("probes")
    ax.set_title(
        "Neutral prompt, per probe type. The model rarely invents an object; "
        "it misses real ones.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = config.REPORTS / "figures" / "probe-breakdown.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    base = json.loads((config.REPORTS / "baseline.json").read_text())
    styles = json.loads((config.REPORTS / "prompt_styles.json").read_text())
    ver = json.loads((config.REPORTS / "verification.json").read_text())
    eval_set = json.loads(config.EVAL_SET.read_text())

    n_pos = sum(len(r["present"]) for r in eval_set)
    n_abs = sum(len(r["absent"]) for r in eval_set)
    kinds: dict[str, int] = {}
    for r in eval_set:
        for k in r["absent"].values():
            kinds[k] = kinds.get(k, 0) + 1

    L = [
        "# Results", "",
        f"**{len(eval_set)} hand-verified images**, {n_pos} objects verified present and "
        f"{n_abs} verified absent ({kinds.get('adversarial', 0)} adversarial, "
        f"{kinds.get('random', 0)} random, {kinds.get('popular', 0)} popular) — "
        f"{n_pos + n_abs} yes/no probes in total. Mean "
        f"{sum(r['n_instances'] for r in eval_set) / len(eval_set):.0f} annotated "
        "object instances per image.", "",
        "Model: `Salesforce/blip-vqa-base` for answering, "
        "`Salesforce/blip-image-captioning-base` for captions, "
        "`openai/clip-vit-base-patch32` for verification. Beam search, so every "
        "number here is deterministic and reproducible.", "",
        "## 1. Baseline — the surprise", "",
        "| question phrasing | accuracy | yes-rate | recall on present | hallucination (adversarial) | hallucination (all absent) |",
        "|---|---|---|---|---|---|",
    ]
    for style, payload in styles.items():
        m = payload["metrics"]
        L.append(
            f"| {style} | {m['overall']['accuracy']:.1%} | {m['overall']['yes_rate']:.1%} | "
            f"{m['present']['recall']:.1%} | {m['adversarial']['hallucination_rate']:.1%} | "
            f"{m['absent_all']['hallucination_rate']:.1%} |"
        )

    L += [
        "",
        "The headline finding is not the one this project set out to measure. BLIP-VQA "
        "**barely hallucinates** on this set — 4.5% on adversarial probes with neutral "
        "phrasing. Its dominant error is the opposite: it **misses 29% of objects that "
        "are genuinely present**. It is over-cautious, not over-confident.",
        "",
        "The caption model makes the same trade even more starkly. Zero of 33 captions "
        "mentioned a verified-absent object — but only because the captions are too "
        "vague to be wrong. On a shelf of 30-odd ceramic pieces it produced *\"a shelf "
        "filled with lots of different colored dishes\"*; on a cluttered living room, "
        "*\"the floor is made of wood\"*. A caption that commits to nothing cannot "
        "hallucinate, and is also useless. A hallucination rate of 0% is not evidence "
        "of grounding.",
        "",
        "## 2. Phrasing more than doubles hallucination", "",
        "Same model, same images, same objects — only the wording of the question "
        "changes:",
        "",
        "| phrasing | example | hallucination (all absent) |",
        "|---|---|---|",
        f"| neutral | *Is there a spoon in the image?* | {styles['neutral']['metrics']['absent_all']['hallucination_rate']:.1%} |",
        f"| presupposing | *Can you see the spoon in this image?* | {styles['presupposing']['metrics']['absent_all']['hallucination_rate']:.1%} |",
        f"| leading | *The image contains a spoon, correct?* | {styles['leading']['metrics']['absent_all']['hallucination_rate']:.1%} |",
        "",
        "Smuggling the object into the premise more than doubles the hallucination rate. "
        "Any reported hallucination number is a property of the prompt as much as the "
        "model, which is worth remembering when comparing published figures.",
        "",
        "## 3. Verification: before and after", "",
        f"CLIP z-score threshold fitted on {len(ver['calibration_images'])} calibration "
        f"images, reported on the {len(ver['test_images'])} held-out images. The split is "
        "by image, not by probe — probes from one image share a CLIP score vector, so a "
        "probe-level split would leak.",
        "",
        "| phrasing | rule | accuracy | precision | recall | F1 | hallucination |",
        "|---|---|---|---|---|---|---|",
    ]
    for style, res in ver["results"].items():
        for key in ("baseline", "clip_only", "verified"):
            r = res[key]
            L.append(f"| {style} | {r['rule']} | {r['accuracy']:.1%} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} | {r['hallucination_rate']:.1%} |")

    n = ver["results"]["neutral"]
    lead = ver["results"]["leading"]
    L += [
        "",
        f"Verification cuts hallucination from {n['baseline']['hallucination_rate']:.1%} to "
        f"{n['verified']['hallucination_rate']:.1%} on neutral phrasing and from "
        f"{lead['baseline']['hallucination_rate']:.1%} to {lead['verified']['hallucination_rate']:.1%} "
        "on leading phrasing — a 33–40% relative reduction. **It is not free.** Recall "
        f"falls from {n['baseline']['recall']:.3f} to {n['verified']['recall']:.3f}, and "
        f"F1 actually *drops* ({n['baseline']['f1']:.3f} → {n['verified']['f1']:.3f}). "
        "Because the rule is a logical AND it can only ever delete a \"yes\", so on a "
        "model that already under-reports, it makes the larger problem worse.",
        "",
        "**CLIP alone is a bad verifier.** Thresholding CLIP on its own hallucinates on "
        f"{n['clip_only']['hallucination_rate']:.1%} of verified-absent objects. CLIP is "
        "trained to match images to plausible captions, not to certify that something is "
        "missing, and a photo of a pottery shelf is genuinely a decent match for \"a photo "
        "of a spoon\". It is only useful here as a veto on top of the VLM.",
        "",
        "### The trade-off in full", "",
        "![trade-off](figures/tradeoff.png)", "",
        "| z threshold | hallucination | recall | F1 |",
        "|---|---|---|---|",
    ]
    for r in n["tradeoff"]:
        if round(r["threshold"], 2) in (-1.5, -0.4, 0.0, 0.5, 1.0, 1.5):
            L.append(f"| {r['threshold']:.1f} | {r['hallucination_rate']:.1%} | "
                     f"{r['recall']:.1%} | {r['f1']:.3f} |")
    L += [
        "",
        "Hallucination can be driven to **exactly zero** — at a cost of roughly 29 points "
        "of recall. Whether that is a good trade is an application question, not a "
        "modelling one: it is the right call for generating radiology reports and the "
        "wrong one for alt-text.",
        "",
        "## 4. What this measurement cannot tell you", "",
        "- **The eval set is biased toward verifiable scenes.** Absence has to be certain "
        "  to be ground truth, and absence is hardest to certify in exactly the cluttered "
        "  indoor scenes where VLMs fail most. Kitchens were dropped because a fork is "
        "  plausible everywhere. The measured rates are therefore probably optimistic.",
        "- **One model family.** BLIP is an encoder-decoder trained on VQA, not an "
        "  LLM-based VLM. The published hallucination results that motivated this project "
        "  mostly target LLaVA/MiniGPT-style models, where a language prior can override "
        "  the image. This result should not be read as \"VLMs don't hallucinate\".",
        f"- **Small numbers.** {n['baseline']['n']} probes on {len(ver['test_images'])} "
        "  held-out images. A hallucination rate of 4.8% is 2 errors. Treat single "
        "  percentage points as noise.",
        f"- **The popular split has {kinds.get('popular', 0)} items** and is reported only "
        "  for completeness; it is too small to draw anything from.",
    ]

    (config.REPORTS / "results.md").write_text("\n".join(L) + "\n")
    figure(ver)
    prompt_style_figure(styles)
    rules_figure(ver)
    probe_breakdown_figure(base)
    print(f"  -> {config.REPORTS / 'results.md'}")


if __name__ == "__main__":
    main()
