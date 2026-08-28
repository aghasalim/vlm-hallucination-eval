"""Assemble reports/results.md and the figures from the measured JSON.

Nothing here computes a metric. Every number plotted is read from
reports/*.json, which the evaluation writes and this file only ever opens.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from . import config
from .style import PALETTE, titled

# Phrasings in escalating order, and one colour each, kept the same in every
# figure so the reader can carry them from one plot to the next.
STYLE_ORDER = ("neutral", "presupposing", "leading")
STYLE_COLOR = {"neutral": PALETTE[0], "presupposing": PALETTE[3],
               "leading": PALETTE[1]}
RULE_COLOR = {"baseline": PALETTE[0], "clip_only": PALETTE[5],
              "verified": PALETTE[2]}
RULE_LABEL = {"baseline": "VLM alone", "clip_only": "CLIP alone",
              "verified": "VLM AND CLIP"}


def _styles_in(payload: dict) -> list[str]:
    return [s for s in STYLE_ORDER if s in payload]


def figure(ver: dict) -> None:
    """The whole trade-off curve, because one operating point hides the cost."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    n_probes = ver["results"]["neutral"]["baseline"]["n"]
    threshold = ver["results"]["neutral"]["threshold"]

    for style in _styles_in(ver["results"]):
        t = ver["results"][style]["tradeoff"]
        axes[0].plot([r["recall"] * 100 for r in t],
                     [r["hallucination_rate"] * 100 for r in t],
                     "-o", ms=2.5, color=STYLE_COLOR[style], label=style)
        axes[1].plot([r["threshold"] for r in t], [r["f1"] for r in t],
                     color=STYLE_COLOR[style], label=style)

    axes[0].set_xlabel("recall on objects that are present (%)")
    axes[0].set_ylabel("hallucination rate on absent objects (%)")
    titled(axes[0],
           "Less hallucination is only ever bought with lost recall",
           f"CLIP z-threshold swept from -1.5 (verifier off) to 3.0, "
           f"{n_probes} held-out probes")
    # Up and right is the un-verified model; the curves only ever move down-left.
    axes[0].legend(loc="upper left")

    axes[1].axvline(threshold, color="#999999", lw=1.0, ls="--", zorder=0)
    axes[1].set_xlabel("CLIP z-score threshold (standard deviations)")
    axes[1].set_ylabel("F1 score (0 to 1)")
    titled(axes[1],
           "F1 peaks near z = -0.9 and falls away fast after it",
           f"dashed line is the threshold fitted on the "
           f"{len(ver['calibration_images'])} calibration images, z = {threshold:.1f}")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    out = config.REPORTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "tradeoff.png")
    plt.close(fig)
    print(f"  -> {out / 'tradeoff.png'}")


def prompt_style_figure(styles: dict) -> None:
    """What the wording of the question does to the answer.

    The image never changes and neither do the objects; only the phrasing does.
    Overall accuracy barely moves, which is why a headline accuracy number hides
    this. The rate that moves is hallucination on objects verified absent from the
    image, and smuggling the object into the premise more than doubles it.
    """
    order = _styles_in(styles)
    colors = [STYLE_COLOR[s] for s in order]
    base = range(len(order))
    n_absent = styles[order[0]]["metrics"]["absent_all"]["n"]
    n_probes = styles[order[0]]["metrics"]["overall"]["n"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    panels = [
        (lambda m: m["absent_all"]["hallucination_rate"] * 100,
         "hallucination rate (%)",
         "Leading phrasing doubles hallucination",
         f"on the {n_absent} objects verified absent from the image", 16.5),
        (lambda m: m["overall"]["accuracy"] * 100,
         "overall accuracy (%)",
         "Accuracy barely moves",
         f"all {n_probes} probes, right answers over total", 100),
        (lambda m: m["overall"]["yes_rate"] * 100,
         "yes-rate (%)",
         "The yes-rate moves only a little",
         "share of probes the model answered yes", 100),
    ]

    for ax, (value, ylabel, title, subtitle, top) in zip(axes, panels, strict=True):
        heights = [value(styles[s]["metrics"]) for s in order]
        ax.bar(base, heights, 0.62, color=colors, edgecolor="0.3", lw=0.5)
        for index, height in enumerate(heights):
            ax.text(index, height + top * 0.02, f"{height:.1f}%", ha="center",
                    fontsize=9.5, fontweight="bold", color="#333333")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, top)
        ax.set_xticks(list(base))
        ax.set_xticklabels(order)
        titled(ax, title, subtitle)

    fig.text(0.5, 0.015,
             "Same images, same objects, three phrasings. Only the left panel "
             "moves, which is why an accuracy headline misses it.",
             fontsize=9.5, color="#5a5a5a", ha="center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = config.REPORTS / "figures" / "prompt-styles.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


def rules_figure(ver: dict) -> None:
    """The three decision rules at the chosen operating point.

    CLIP alone is worse than the VLM alone on accuracy. The combination beats both
    on hallucination rate, which is the only reason to pay for it.
    """
    styles = _styles_in(ver["results"])
    present = [r for r in ("baseline", "clip_only", "verified")
               if r in ver["results"][styles[0]]]
    clip_rate = ver["results"][styles[0]]["clip_only"]["hallucination_rate"] * 100
    n_probes = ver["results"][styles[0]]["baseline"]["n"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    width = 0.78 / max(len(present), 1)
    for offset, rule in enumerate(present):
        xs = [i + (offset - (len(present) - 1) / 2) * width
              for i in range(len(styles))]
        for ax, key, top in ((axes[0], "accuracy", 100),
                             (axes[1], "hallucination_rate", 85)):
            heights = [ver["results"][s][rule][key] * 100 for s in styles]
            ax.bar(xs, heights, width, label=RULE_LABEL[rule],
                   color=RULE_COLOR[rule], edgecolor="0.3", lw=0.4)
            for x, height in zip(xs, heights, strict=True):
                ax.text(x, height + top * 0.015, f"{height:.1f}", ha="center",
                        fontsize=8.5, color="#333333")

    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_ylim(0, 100)
    titled(axes[0], "The AND rule gives up about two points of accuracy",
           f"{n_probes} held-out probes, one group of bars per phrasing")
    axes[0].legend(loc="upper center", ncol=3, columnspacing=1.4)

    axes[1].set_ylabel("hallucination rate (%)")
    axes[1].set_ylim(0, 85)
    titled(axes[1], "and cuts hallucination under all three phrasings",
           f"CLIP alone answers yes on {clip_rate:.1f}% of absent objects, so it "
           "only works as a veto")

    for ax in axes:
        ax.set_xticks(range(len(styles)))
        ax.set_xticklabels(styles)
        ax.set_xlabel("question phrasing")
    fig.tight_layout()
    out = config.REPORTS / "figures" / "rules.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


def probe_breakdown_figure(baseline: dict) -> None:
    """Where the errors are: present objects missed, absent objects invented.

    The asymmetry is the finding. The model is close to perfect at not inventing
    objects under a neutral prompt, and misses a third of the ones that are there.

    Dots rather than bars, because the popular split holds two probes and a bar
    of that length invites reading a rate off it.
    """
    probes = baseline["probes"]
    kinds = list(dict.fromkeys(p["kind"] for p in probes))
    rows = []
    for kind in kinds:
        subset = [p for p in probes if p["kind"] == kind]
        wrong = sum(1 for p in subset if p["pred"] != p["truth"])
        rows.append((kind, wrong, len(subset)))

    fig, ax = plt.subplots(figsize=(9.0, 3.7))
    ys = list(range(len(rows)))[::-1]
    for y, (kind, wrong, total) in zip(ys, rows, strict=True):
        rate = 100 * wrong / total
        small = total < 10
        color = PALETTE[5] if small else PALETTE[1]
        ax.plot([0, rate], [y, y], color="#cccccc", lw=1.6, zorder=1)
        ax.plot([rate], [y], "o", ms=11, color=color, zorder=2)
        note = f"{wrong} of {total} wrong"
        if small:
            note += ", too few to read a rate off"
        ax.text(rate + 1.6, y, note, va="center", fontsize=9.5, color="#333333")

    ax.set_yticks(ys)
    ax.set_yticklabels([kind for kind, _, _ in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 78)
    ax.set_xlabel("probes answered wrongly (%)")
    ax.set_ylabel("probe type")
    ax.grid(axis="y", visible=False)
    titled(ax,
           "The model misses real objects far more than it invents absent ones",
           f"neutral phrasing, all {len(probes)} probes; grey means the split is "
           "too small to report")
    fig.tight_layout()
    out = config.REPORTS / "figures" / "probe-breakdown.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


def _shrink_gif(path) -> None:
    """Rewrite every frame onto one shared palette. Roughly halves the file."""
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(64, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], loop=0,
              duration=durations, optimize=True)


def threshold_sweep_animation(ver: dict) -> None:
    """Turn the verifier knob one step at a time and watch what it costs.

    Same held-out probes and the same neutral phrasing in every frame; the only
    thing that changes is the CLIP z-threshold. Read straight off the committed
    sweep in verification.json, so it is deterministic.
    """
    t = ver["results"]["neutral"]["tradeoff"]
    fitted = ver["results"]["neutral"]["threshold"]
    n_probes = ver["results"]["neutral"]["baseline"]["n"]
    zs = [r["threshold"] for r in t]
    halluc = [r["hallucination_rate"] * 100 for r in t]
    recall = [r["recall"] * 100 for r in t]

    # 11.0 x 4.0 at the house savefig dpi of 170 is a whole number of pixels.
    # A width that lands on a fraction of a pixel makes PillowWriter and the
    # canvas disagree by one, and every frame comes out sheared.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    left, right = axes

    left.plot(zs, recall, color="#dddddd", lw=1.4, zorder=1)
    left.plot(zs, halluc, color="#dddddd", lw=1.4, zorder=1)
    left.axvline(fitted, color="#bbbbbb", lw=1.0, ls="--", zorder=0)
    recall_line, = left.plot([], [], color=PALETTE[0], label="recall on present objects")
    halluc_line, = left.plot([], [], color=PALETTE[1], label="hallucination on absent objects")
    recall_dot, = left.plot([], [], "o", ms=7, color=PALETTE[0])
    halluc_dot, = left.plot([], [], "o", ms=7, color=PALETTE[1])
    left.set_xlim(min(zs) - 0.1, max(zs) + 0.1)
    left.set_ylim(-3, 82)
    left.set_xlabel("CLIP z-score threshold (standard deviations)")
    left.set_ylabel("rate (%)")
    left.legend(loc="upper right")
    titled(left, "A stricter verifier deletes real objects too",
           f"neutral phrasing, {n_probes} held-out probes, dashed line is the "
           "fitted threshold")

    right.plot(recall, halluc, color="#dddddd", lw=1.4, zorder=1)
    trail, = right.plot([], [], "-", color=PALETTE[2], lw=2.2, zorder=2)
    head, = right.plot([], [], "o", ms=9, color=PALETTE[2], zorder=3)
    readout = right.text(0.03, 0.97, "", transform=right.transAxes, va="top",
                         ha="left", fontsize=10, color="#333333")
    right.set_xlim(10, 82)
    right.set_ylim(-1, 9)
    right.set_xlabel("recall on objects that are present (%)")
    right.set_ylabel("hallucination rate on absent objects (%)")
    titled(right, "Three recall points per hallucination point",
           "up and to the right is the un-verified model")
    # Fixed margins rather than tight_layout: the frames must all be laid out
    # identically, and tight_layout reflows around the length of the titles.
    fig.subplots_adjust(left=0.055, right=0.99, top=0.845, bottom=0.155,
                        wspace=0.30)

    def draw(i: int):
        recall_line.set_data(zs[:i + 1], recall[:i + 1])
        halluc_line.set_data(zs[:i + 1], halluc[:i + 1])
        recall_dot.set_data([zs[i]], [recall[i]])
        halluc_dot.set_data([zs[i]], [halluc[i]])
        trail.set_data(recall[:i + 1], halluc[:i + 1])
        head.set_data([recall[i]], [halluc[i]])
        readout.set_text(f"z = {zs[i]:+.1f}\nhallucination {halluc[i]:.1f}%\n"
                         f"recall {recall[i]:.1f}%")
        return ()

    frames = list(range(len(t))) + [len(t) - 1] * 8
    anim = FuncAnimation(fig, draw, frames=frames, interval=125, blit=False)
    out = config.REPORTS / "figures" / "threshold-sweep.gif"
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    _shrink_gif(out)
    print(f"  -> {out} ({out.stat().st_size / 1024:.0f} kB)")


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
        f"{kinds.get('random', 0)} random, {kinds.get('popular', 0)} popular), "
        f"{n_pos + n_abs} yes/no probes in total. Mean "
        f"{sum(r['n_instances'] for r in eval_set) / len(eval_set):.0f} annotated "
        "object instances per image.", "",
        "Model: `Salesforce/blip-vqa-base` for answering, "
        "`Salesforce/blip-image-captioning-base` for captions, "
        "`openai/clip-vit-base-patch32` for verification. Beam search, so every "
        "number here is deterministic and reproducible.", "",
        "## 1. Baseline, the surprise", "",
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
        "**barely hallucinates** on this set, 4.5% on adversarial probes with neutral "
        "phrasing. Its dominant error is the opposite: it **misses 29% of objects that "
        "are genuinely present**. It is over-cautious, not over-confident.",
        "",
        "The caption model makes the same trade even more starkly. Zero of 33 captions "
        "mentioned a verified-absent object, but only because the captions are too "
        "vague to be wrong. On a shelf of 30-odd ceramic pieces it produced *\"a shelf "
        "filled with lots of different colored dishes\"*; on a cluttered living room, "
        "*\"the floor is made of wood\"*. A caption that commits to nothing cannot "
        "hallucinate, and is also useless. A hallucination rate of 0% is not evidence "
        "of grounding.",
        "",
        "## 2. Phrasing more than doubles hallucination", "",
        "Same model, same images, same objects, only the wording of the question "
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
        "by image, not by probe, probes from one image share a CLIP score vector, so a "
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
    # Every phrasing, not the two that flatter the method. Computed here so the
    # sentence cannot drift away from the JSON.
    rel = {s: 1 - r["verified"]["hallucination_rate"] / r["baseline"]["hallucination_rate"]
           for s, r in ver["results"].items()}
    rel_txt = ", ".join(f"{s} {rel[s]:.0%}" for s in _styles_in(rel))
    saved = sorted(r["baseline"]["fp"] - r["verified"]["fp"] for r in ver["results"].values())
    absent_n = n["baseline"]["fp"] + n["baseline"]["tn"]
    L += [
        "",
        f"Verification cuts hallucination from {n['baseline']['hallucination_rate']:.1%} to "
        f"{n['verified']['hallucination_rate']:.1%} on neutral phrasing and from "
        f"{lead['baseline']['hallucination_rate']:.1%} to {lead['verified']['hallucination_rate']:.1%} "
        f"on leading phrasing. The relative reduction is {rel_txt}, so the honest range is "
        f"{min(rel.values()):.0%} to {max(rel.values()):.0%} and it depends on the phrasing. "
        f"In counts that is {saved[0]} to {saved[-1]} fewer false positives out of "
        f"{absent_n} verified-absent probes. **It is not free.** Recall "
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
        "Hallucination can be driven to **exactly zero**, at a cost of roughly 29 points "
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
    threshold_sweep_animation(ver)
    print(f"  -> {config.REPORTS / 'results.md'}")


if __name__ == "__main__":
    main()
