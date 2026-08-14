"""Streamlit demo: caption or ask, and show the grounding behind every claim."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vlmhall import config, models  # noqa: E402
from src.vlmhall.evaluate import mentioned_objects, question_for  # noqa: E402
from src.vlmhall.verify import image_z_scores  # noqa: E402

st.set_page_config(page_title="Hallucination-aware captioning", layout="wide")

DEFAULT_THRESHOLD = -0.4  # fitted on the calibration split; see reports/results.md


@st.cache_data(show_spinner=False)
def _z(img_bytes: bytes) -> dict[str, float]:
    import io
    return image_z_scores(Image.open(io.BytesIO(img_bytes)).convert("RGB"))


@st.cache_data(show_spinner="Fetching image from COCO…")
def _coco_image(file_name: str) -> bytes:
    """Download one val2017 image. Cached, so each image is fetched once per run."""
    import urllib.request
    url = f"http://images.cocodataset.org/val2017/{file_name}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


st.title("Does the model actually see what it says it sees?")
st.caption(
    "BLIP writes the caption or answers the question; CLIP independently scores "
    "whether each claimed object is grounded in the image. Both are shown, because "
    "the point is the disagreement."
)

results_md = config.REPORTS / "results.md"
if results_md.exists():
    with st.expander("Measured results on the hand-built adversarial set", expanded=False):
        st.markdown(results_md.read_text())

col_l, col_r = st.columns([1, 1])

with col_l:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # The evaluation images are COCO's, not mine, so they are not committed --
    # which would leave a hosted demo with nothing to click. Fetch them from
    # cocodataset.org on demand instead, and fall back to any local copy that
    # `make data` has already produced.
    eval_rows = json.loads(config.EVAL_SET.read_text()) if config.EVAL_SET.exists() else []
    pick = None
    if eval_rows:
        labels = {f"{r['file_name']} — {r['scene'][:52]}": r for r in eval_rows[:15]}
        chosen = st.selectbox("…or use one from the adversarial evaluation set",
                              ["(none)"] + list(labels))
        if chosen != "(none)":
            row = labels[chosen]
            local = config.IMAGES / row["file_name"]
            if local.exists():
                pick = local
            else:
                pick = _coco_image(row["file_name"])
            st.caption(f"**Why this one is hard:** {row['why_hard']}  \n"
                       f"**Verified absent:** {', '.join(row['absent'])}")

    threshold = st.slider(
        "CLIP grounding threshold (z-score)", -1.5, 3.0, DEFAULT_THRESHOLD, 0.1,
        help="Higher rejects more claims: hallucination falls, but so does recall. "
             "The trade-off curve is in the results above.",
    )

src = None
raw = None
if up is not None:
    raw = up.getvalue()
elif pick is not None:
    # pick is a local Path when `make data` has run, raw bytes when the image
    # was fetched from COCO.
    raw = pick.read_bytes() if isinstance(pick, Path) else pick
if raw is not None:
    import io
    src = Image.open(io.BytesIO(raw)).convert("RGB")

if src is None:
    st.info("Upload an image or choose one from the evaluation set to begin.")
    st.stop()

with col_l:
    st.image(src, use_container_width=True)

with col_r:
    with st.spinner("Captioning and scoring…"):
        cap = models.caption(src)
        z = _z(raw)

    st.subheader("Caption")
    st.markdown(f"### “{cap}”")

    objs = sorted(mentioned_objects(cap))
    if not objs:
        st.warning(
            "The caption does not commit to any recognisable object, so there is "
            "nothing to verify. This is common and worth noticing: a caption vague "
            "enough to be unfalsifiable is not the same as a grounded one."
        )
    else:
        st.subheader("Claim-by-claim grounding")
        for o in objs:
            score = z.get(o, 0.0)
            ok = score >= threshold
            c1, c2 = st.columns([3, 2])
            c1.markdown(f"{'✅' if ok else '⚠️'} **{o}** — CLIP z = `{score:+.2f}`")
            c2.progress(min(max((score + 2) / 5, 0.0), 1.0))
            if not ok:
                c1.caption("below threshold — treat as unsupported")

    st.divider()
    st.subheader("Ask about a specific object")
    obj = st.text_input("Object", value="spoon")
    style = st.radio("Phrasing", ["neutral", "leading", "presupposing"], horizontal=True,
                     help="Leading phrasing more than doubled the hallucination rate "
                          "in the evaluation.")
    if obj:
        q = question_for(obj.strip().lower(), style)
        ans = models.ask(src, q)
        score = z.get(obj.strip().lower())
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**BLIP:** `{ans}`")
        if score is None:
            st.caption(f"'{obj}' is not a COCO category, so CLIP has no calibrated "
                       "z-score for it here.")
        else:
            verdict = "supported" if score >= threshold else "NOT supported"
            st.markdown(f"**CLIP grounding:** z = `{score:+.2f}` → {verdict}")
            if models.yes_no(ans) is True and score < threshold:
                st.error("Disagreement: the model says yes, the image does not support "
                         "it. This is the case the verification layer is built to catch.")
