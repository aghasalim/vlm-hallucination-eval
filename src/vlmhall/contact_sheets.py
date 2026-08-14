"""Download candidate images and lay them out for manual verification.

Each tile is annotated with its index, the objects COCO says are present, and the
absence probes awaiting confirmation, so the verification pass can be done
against the pixels rather than against a spreadsheet.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config

TILE = 460
COLS, ROWS = 3, 2
PAD_BOTTOM = 78


def download() -> list[dict]:
    rows = json.loads((config.DATA / "eval_set_draft.json").read_text())
    config.IMAGES.mkdir(parents=True, exist_ok=True)
    for r in rows:
        dest = config.IMAGES / r["file_name"]
        if not dest.exists():
            urllib.request.urlretrieve(r["coco_url"], dest)
    print(f"{len(rows)} images in {config.IMAGES}")
    return rows


def _font(size: int):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def sheets(rows: list[dict]) -> None:
    out_dir = config.DATA / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    per = COLS * ROWS
    f_small, f_tiny = _font(15), _font(13)

    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        sheet = Image.new("RGB", (COLS * TILE, ROWS * (TILE + PAD_BOTTOM)), "white")
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            cx, cy = (k % COLS) * TILE, (k // COLS) * (TILE + PAD_BOTTOM)
            im = Image.open(config.IMAGES / r["file_name"]).convert("RGB")
            im.thumbnail((TILE - 8, TILE - 8))
            sheet.paste(im, (cx + 4, cy + 4))
            idx = s + k
            d.text((cx + 6, cy + TILE + 2), f"[{idx}] {r['file_name']}",
                   fill="black", font=f_small)
            d.text((cx + 6, cy + TILE + 22), "ABSENT? " + ", ".join(r["probes"]["adversarial"]),
                   fill="#b00000", font=f_tiny)
            present = ", ".join(r["present"])
            d.text((cx + 6, cy + TILE + 40), "present: " + present[:58],
                   fill="#006000", font=f_tiny)
            if len(present) > 58:
                d.text((cx + 6, cy + TILE + 56), "         " + present[58:116],
                       fill="#006000", font=f_tiny)
        p = out_dir / f"sheet_{s // per:02d}.png"
        sheet.save(p)
        print(f"  {p}")


if __name__ == "__main__":
    sheets(download())
