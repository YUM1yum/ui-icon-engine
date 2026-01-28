# yolo_to_descriptions_openai.py
# - Input: YOLO labels (.txt) + images
# - Output: JSONL in the same "out_icon_descriptions_openai.jsonl"-style as regen_descriptions_openai.py
#
# This script intentionally mirrors regen_descriptions_openai.py behavior as closely as possible
# except for the input source.

import json
import time
import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel

from openai import OpenAI

try:
    from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
except Exception:  # pragma: no cover
    RateLimitError = APIError = APITimeoutError = APIConnectionError = Exception


# =========================
# Config (match regen_descriptions_openai.py)
# =========================
# Your dataset paths (train split)
IMAGES_DIR = Path(r"C:\\Users\\dldna\\icon\\dataset\\test_data_rename\\images")
LABELS_DIR = Path(r"C:\\Users\\dldna\\icon\\dataset\\test_data_rename\\labels")

OUTPUT_JSONL = Path("our_test_data_descriptions_openai.jsonl")

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 140

ICON_PAD_PX = 2
CTX_PAD_SCALE = 2.0

ICON_MAX_SIDE = 192
CTX_MAX_SIDE = 640

ICON_DETAIL = "high"
CTX_DETAIL = "low"

ENABLE_CTX_FALLBACK = True
FALLBACK_CONF_THRESH = 0.70

RESUME = True

MAX_RETRIES = 6
BASE_BACKOFF_SEC = 1.5

# =========================
# Prompt (match regen_descriptions_openai.py)
# =========================
INSTRUCTIONS = """Label a UI icon so users can find it via natural-language commands.

Inputs: (1) ICON crop, (2) CONTEXT crop.

Return ONE-LINE minified JSON:
{"name":string,"aliases_en":[string],"action":string,"description":string,"app":string|null,"confidence":number}

Rules:
- Focus on the most common user-facing name (settings/search/menu/download/save/copy/share/back/close/refresh/edit/delete/add/home/profile/help).
- aliases_en must be ENGLISH ONLY (ASCII). No Korean/Chinese/etc.
- description: meaning + typical user intent. Keep it short by default (1–2 sentences). Add detail only if needed.
- You may mention app/brand names if clearly implied by context.
- confidence: 0..1
No extra text.
"""

USER_TEXT = "Identify the icon’s common name and intent from the icon + UI context. Output only the JSON."


# =========================
# Structured output schema (must match the prompt)
# =========================
class IconDesc(BaseModel):
    name: str
    aliases_en: list[str]
    action: str
    description: str
    app: str | None
    confidence: float


# =========================
# Utils: JSONL
# =========================
def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_line(fp, obj: dict) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sample_id(item: dict) -> str:
    # same idea as regen_descriptions_openai.py
    key = f"{item.get('image','')}|{item.get('bbox_xyxy','')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def to_posix_path_str(p: Path) -> str:
    # matches the example like "C:/path/to/image.png"
    return p.resolve().as_posix()


# =========================
# Utils: YOLO parsing
# =========================
@dataclass(frozen=True)
class YoloBox:
    cls_id: int
    xc: float
    yc: float
    w: float
    h: float


def parse_yolo_label_file(label_path: Path) -> list[YoloBox]:
    """
    YOLO txt line format:
      <cls> <x_center> <y_center> <width> <height>
    all coords are normalized [0,1]
    """
    boxes: list[YoloBox] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for ln, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            # skip malformed lines
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append(YoloBox(cls_id=cls_id, xc=xc, yc=yc, w=w, h=h))
        except Exception:
            # skip malformed lines
            continue

    return boxes


def yolo_to_xyxy_pixels(box: YoloBox, W: int, H: int) -> list[int]:
    """
    Convert normalized YOLO box -> absolute pixel xyxy.
    """
    x1 = (box.xc - box.w / 2.0) * W
    y1 = (box.yc - box.h / 2.0) * H
    x2 = (box.xc + box.w / 2.0) * W
    y2 = (box.yc + box.h / 2.0) * H

    # int conversion; clamp logic later will fix edge cases
    return [int(x1), int(y1), int(x2), int(y2)]


def find_matching_image(images_dir: Path, stem: str) -> Path | None:
    """
    Finds image by trying common extensions.
    """
    exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: any file with same stem
    candidates = list(images_dir.glob(stem + ".*"))
    return candidates[0] if candidates else None


def iter_dataset_samples(images_dir: Path, labels_dir: Path) -> Iterator[tuple[Path, int, list[int]]]:
    """
    Yields: (image_path, yolo_cls, bbox_xyxy_pixels)
    """
    for label_path in labels_dir.glob("*.txt"):
        img_path = find_matching_image(images_dir, label_path.stem)
        if img_path is None:
            continue

        # load image size once
        try:
            with Image.open(img_path) as img:
                W, H = img.size
        except Exception:
            continue

        boxes = parse_yolo_label_file(label_path)
        for b in boxes:
            xyxy = yolo_to_xyxy_pixels(b, W=W, H=H)
            yield img_path, int(b.cls_id), xyxy


# =========================
# Utils: crop / resize / encode (match regen_descriptions_openai.py)
# =========================
def clamp_box_xyxy(box, W: int, H: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return [x1, y1, x2, y2]


def pad_box_xyxy(box, pad_px: int, W: int, H: int) -> list[int]:
    x1, y1, x2, y2 = box
    return clamp_box_xyxy([x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px], W, H)


def make_icon_and_context_crops(
    image: Image.Image,
    bbox_xyxy,
    icon_pad_px: int = 2,
    ctx_pad_scale: float = 2.0,
):
    W, H = image.size
    box = clamp_box_xyxy(bbox_xyxy, W, H)

    icon_box = pad_box_xyxy(box, icon_pad_px, W, H)
    icon_img = image.crop(tuple(icon_box))

    bw = box[2] - box[0]
    bh = box[3] - box[1]
    pad = int(max(bw, bh) * ctx_pad_scale)
    ctx_box = pad_box_xyxy(box, pad, W, H)
    ctx_img = image.crop(tuple(ctx_box))

    return box, icon_img, ctx_img


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max_side is None:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.BICUBIC)


def pil_to_data_url(img: Image.Image, fmt: str = "JPEG", jpeg_quality: int = 85) -> str:
    buf = BytesIO()
    if fmt.upper() == "JPEG":
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# =========================
# OpenAI call (Responses API) (match regen_descriptions_openai.py)
# =========================
def call_openai_describe(client: OpenAI, icon_url: str, ctx_url: str, ctx_detail: str) -> dict:
    resp = client.responses.parse(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_TEXT},
                    {"type": "input_image", "image_url": icon_url, "detail": ICON_DETAIL},
                    {"type": "input_image", "image_url": ctx_url, "detail": ctx_detail},
                ],
            }
        ],
        text_format=IconDesc,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    parsed = resp.output_parsed
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    if hasattr(parsed, "dict"):
        return parsed.dict()
    return dict(parsed)


def call_with_retry(client: OpenAI, icon_url: str, ctx_url: str, ctx_detail: str) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_openai_describe(client, icon_url, ctx_url, ctx_detail=ctx_detail)
        except RateLimitError as e:
            last_err = e
        except (APITimeoutError, APIConnectionError, APIError) as e:
            last_err = e
        except Exception as e:
            last_err = e

        sleep_s = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
        time.sleep(min(60, sleep_s))

    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} retries: {last_err}")


def postprocess_desc(desc_obj: dict) -> tuple[dict, float]:
    # confidence 분리 + 클램프
    conf = float(desc_obj.get("confidence", 0.0))
    conf = max(0.0, min(1.0, conf))

    # aliases 영어(ASCII)만
    aliases = desc_obj.get("aliases_en", [])
    if isinstance(aliases, list):
        aliases = [a for a in aliases if isinstance(a, str) and a and all(ord(c) < 128 for c in a)]
    else:
        aliases = []
    desc_obj["aliases_en"] = aliases

    # nested confidence 제거
    desc_obj = {k: v for k, v in desc_obj.items() if k != "confidence"}
    return desc_obj, conf


# =========================
# Main
# =========================
def build_output_item(image_path: Path, bbox_xyxy: list[int], yolo_cls: int, desc_obj: dict, desc_conf: float) -> dict:
    return {
        "image": to_posix_path_str(image_path),
        "bbox_xyxy": [int(v) for v in bbox_xyxy],
        "yolo_cls": int(yolo_cls),
        "description": desc_obj,
        "desc_confidence": float(desc_conf),
    }


def main():
    assert IMAGES_DIR.exists(), f"Images dir not found: {IMAGES_DIR}"
    assert LABELS_DIR.exists(), f"Labels dir not found: {LABELS_DIR}"

    client = OpenAI()

    done: set[str] = set()
    if RESUME and OUTPUT_JSONL.exists():
        for out_item in read_jsonl(OUTPUT_JSONL):
            done.add(sample_id(out_item))

    # Pre-count for nicer tqdm (optional)
    label_files = list(LABELS_DIR.glob("*.txt"))

    with OUTPUT_JSONL.open("a", encoding="utf-8") as out_fp:
        # Iterate label files, but yield per-box items
        for label_path in tqdm(label_files, desc="processing label files"):
            img_path = find_matching_image(IMAGES_DIR, label_path.stem)
            if img_path is None:
                continue

            # Load image once per file
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            W, H = img.size
            boxes = parse_yolo_label_file(label_path)
            if not boxes:
                continue

            for b in boxes:
                bbox_xyxy = yolo_to_xyxy_pixels(b, W=W, H=H)
                bbox_xyxy = clamp_box_xyxy(bbox_xyxy, W, H)

                # build an "item" signature consistent with regen_descriptions_openai.py
                tmp_item = {
                    "image": to_posix_path_str(img_path),
                    "bbox_xyxy": bbox_xyxy,
                    "yolo_cls": int(b.cls_id),
                }
                sid = sample_id(tmp_item)
                if sid in done:
                    continue

                # crops
                _, icon_img, ctx_img = make_icon_and_context_crops(
                    img, bbox_xyxy, icon_pad_px=ICON_PAD_PX, ctx_pad_scale=CTX_PAD_SCALE
                )

                icon_img = resize_max_side(icon_img, ICON_MAX_SIDE)
                ctx_img = resize_max_side(ctx_img, CTX_MAX_SIDE)

                icon_url = pil_to_data_url(icon_img, fmt="JPEG", jpeg_quality=85)
                ctx_url = pil_to_data_url(ctx_img, fmt="JPEG", jpeg_quality=80)

                # 1) default ctx_detail=low
                desc_raw = call_with_retry(client, icon_url, ctx_url, ctx_detail=CTX_DETAIL)

                # 2) fallback to high detail if confidence low
                if ENABLE_CTX_FALLBACK:
                    conf_try = float(desc_raw.get("confidence", 0.0))
                    if conf_try < FALLBACK_CONF_THRESH:
                        desc_raw = call_with_retry(client, icon_url, ctx_url, ctx_detail="high")

                desc_obj, desc_conf = postprocess_desc(desc_raw)

                out = build_output_item(
                    image_path=img_path,
                    bbox_xyxy=bbox_xyxy,
                    yolo_cls=int(b.cls_id),
                    desc_obj=desc_obj,
                    desc_conf=desc_conf,
                )

                write_jsonl_line(out_fp, out)
                out_fp.flush()
                done.add(sid)

            try:
                img.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()