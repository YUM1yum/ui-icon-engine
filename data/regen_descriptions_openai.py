import os
import json
import time
import base64
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from openai import OpenAI
# 에러 타입은 SDK 버전에 따라 다를 수 있어, 없으면 Exception으로 처리
try:
    from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
except Exception:  # pragma: no cover
    RateLimitError = APIError = APITimeoutError = APIConnectionError = Exception


# =========================
# Config
# =========================
INPUT_JSONL  = Path("out_icon_descriptions.jsonl")
OUTPUT_JSONL = Path("out_icon_descriptions_openai.jsonl")

MODEL = "gpt-4o-mini"   # 필요하면 "gpt-4o"로 변경
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 90  # 1~2문장 정도면 보통 60~120 사이면 충분

ICON_PAD_PX = 2
CTX_PAD_SCALE = 2.0

# 비용/지연 줄이려면 이미지 리사이즈 권장 (너무 줄이면 아이콘 디테일 손실)
ICON_MAX_SIDE = 256
CTX_MAX_SIDE  = 768

# 이전 결과가 있으면 이어서 돌리기
RESUME = True

# 재시도
MAX_RETRIES = 6
BASE_BACKOFF_SEC = 1.5


INSTRUCTIONS = """You label UI icons so that a user can find the right icon via natural-language commands.

You will receive two images in this order:
(1) ICON crop (tight)
(2) CONTEXT crop (surrounding UI)

PRIMARY GOAL (most important):
- Decide the most common name users would call this icon (e.g., settings, search, menu, download, save, copy, share, close, back, refresh, edit, delete, add, home, profile, notification, help).
- Prefer a single canonical label when possible.

SECONDARY GOAL:
- Provide a few alternative user expressions/synonyms (include BOTH English and Korean when reasonable).

You MAY:
- Mention app/service/brand names (Chrome, Discord, VS Code, Instagram, etc.) if the context strongly suggests it.

You should NOT:
- Focus on detailed visual appearance unless it helps disambiguate between possible icon names.

STRICT OUTPUT FORMAT:
- Output exactly ONE LINE of minified JSON with keys:
    {"name": string, "aliases": [string], "action": string, "app": string|null, "confidence": number}
- No markdown, no extra text, no line breaks.
- confidence must be between 0 and 1.
"""

USER_TEXT = "Identify the icon’s common user-facing name and meaning using the icon crop + the UI context. Return only the one-line JSON."


# =========================
# Utils: JSONL
# =========================
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl_line(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

def sample_id(item: dict) -> str:
    key = f"{item.get('image','')}|{item.get('bbox_xyxy','')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# =========================
# Utils: crop
# =========================
def clamp_box_xyxy(box, W, H):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return [x1, y1, x2, y2]

def pad_box_xyxy(box, pad_px, W, H):
    x1, y1, x2, y2 = box
    return clamp_box_xyxy([x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px], W, H)

def make_icon_and_context_crops(image: Image.Image, bbox_xyxy, icon_pad_px=2, ctx_pad_scale=2.0):
    W, H = image.size
    box = clamp_box_xyxy(bbox_xyxy, W, H)

    # icon crop
    icon_box = pad_box_xyxy(box, icon_pad_px, W, H)
    icon_img = image.crop(tuple(icon_box))

    # context crop
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

def pil_to_data_url(img: Image.Image, fmt="PNG", jpeg_quality=90) -> str:
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
# OpenAI call (Responses API)
# =========================
def call_openai_describe(client: OpenAI, icon_url: str, ctx_url: str) -> str:
    # Responses API: 텍스트+이미지 입력을 content 배열로 넣는 형태가 공식 가이드에 있습니다. :contentReference[oaicite:4]{index=4}
    resp = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,     # system/developer 역할 :contentReference[oaicite:5]{index=5}
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_TEXT},
                {"type": "input_image", "image_url": icon_url},
                {"type": "input_image", "image_url": ctx_url},
            ],
        }],
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,  # 상한 파라미터 :contentReference[oaicite:6]{index=6}
    )
    # Python SDK는 output_text로 텍스트를 편하게 꺼낼 수 있습니다. :contentReference[oaicite:7]{index=7}
    text = (resp.output_text or "").strip()
    # 안전장치: 여러 줄이면 마지막 줄만
    return text.splitlines()[-1].strip()


def call_with_retry(client: OpenAI, icon_url: str, ctx_url: str) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_openai_describe(client, icon_url, ctx_url)
        except RateLimitError as e:
            # 레이트리밋은 백오프로 재시도 권장 :contentReference[oaicite:8]{index=8}
            last_err = e
        except (APITimeoutError, APIConnectionError, APIError) as e:
            last_err = e
        except Exception as e:
            last_err = e

        sleep_s = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
        time.sleep(min(60, sleep_s))

    raise RuntimeError(f"OpenAI call failed after {MAX_RETRIES} retries: {last_err}")


# =========================
# Main
# =========================
def main():
    assert INPUT_JSONL.exists(), f"Input not found: {INPUT_JSONL}"
    client = OpenAI()  # OPENAI_API_KEY 환경변수를 사용 :contentReference[oaicite:9]{index=9}

    done = set()
    if RESUME and OUTPUT_JSONL.exists():
        for item in read_jsonl(OUTPUT_JSONL):
            if "_id" in item:
                done.add(item["_id"])

    with OUTPUT_JSONL.open("a", encoding="utf-8") as out_fp:
        for item in tqdm(read_jsonl(INPUT_JSONL), desc="regenerating"):
            _id = sample_id(item)
            if _id in done:
                continue

            img_path = Path(item["image"])
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            bbox = item["bbox_xyxy"]
            img = Image.open(img_path).convert("RGB")

            box, icon_img, ctx_img = make_icon_and_context_crops(
                img, bbox, icon_pad_px=ICON_PAD_PX, ctx_pad_scale=CTX_PAD_SCALE
            )

            # 리사이즈 (토큰/비용 절감 + 속도 개선)
            icon_img = resize_max_side(icon_img, ICON_MAX_SIDE)
            ctx_img  = resize_max_side(ctx_img,  CTX_MAX_SIDE)

            # data URL(base64)
            icon_url = pil_to_data_url(icon_img, fmt="PNG")
            ctx_url  = pil_to_data_url(ctx_img,  fmt="PNG")

            new_desc = call_with_retry(client, icon_url, ctx_url)

            out = dict(item)  # 기존 필드 유지
            out["_id"] = _id
            out["bbox_xyxy_clamped"] = box

            # 기존 description 보존 + 새 description 저장
            out["description_prev"] = item.get("description")
            out["description"] = new_desc
            out["description_model"] = MODEL

            write_jsonl_line(out_fp, out)
            out_fp.flush()


if __name__ == "__main__":
    main()