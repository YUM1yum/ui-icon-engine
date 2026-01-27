import json
import time
import base64
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel

from openai import OpenAI

try:
    from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
except Exception:  # pragma: no cover
    RateLimitError = APIError = APITimeoutError = APIConnectionError = Exception


# =========================
# Config
# =========================
INPUT_JSONL  = Path("out_icon_descriptions.jsonl")
OUTPUT_JSONL = Path("out_icon_descriptions_openai.jsonl")

MODEL = "gpt-4o-mini"
# 비용만 보면 모델별 이미지 토큰 정책/계수가 달라서 “mini가 항상 더 싸다”가 아닐 수 있습니다.
# 이미지 토큰은 size/detail 기반으로 과금되며(model별 상이) detail=low로 절감 가능합니다. :contentReference[oaicite:2]{index=2}

TEMPERATURE = 0.0

# 출력이 길어질수록 비용 증가 → 일단 타이트하게 잡고, 프롬프트로 “필요하면 길게” 허용
MAX_OUTPUT_TOKENS = 140

ICON_PAD_PX = 2
CTX_PAD_SCALE = 2.0

# 최종 입력 해상도를 줄이는 게 이미지 토큰 절감에 직접적입니다. :contentReference[oaicite:3]{index=3}
ICON_MAX_SIDE = 192
CTX_MAX_SIDE  = 640

# detail 옵션: low는 토큰/속도 절감에 도움 :contentReference[oaicite:4]{index=4}
ICON_DETAIL = "high"
CTX_DETAIL  = "low"

# low detail로 돌렸는데 자신감(confidence)이 낮으면 그 샘플만 high detail로 재호출(옵션)
ENABLE_CTX_FALLBACK = True
FALLBACK_CONF_THRESH = 0.70

RESUME = True

MAX_RETRIES = 6
BASE_BACKOFF_SEC = 1.5


# =========================
# Prompt (짧게 줄여서 입력 토큰도 절감)
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
# Structured output schema (프롬프트와 반드시 일치)
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
# Utils: crop / resize / encode
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

def pil_to_data_url(img: Image.Image, fmt="JPEG", jpeg_quality=85) -> str:
    # 포맷/압축은 주로 전송량/속도에 영향(토큰은 보통 최종 해상도/detail에 더 좌우됨) :contentReference[oaicite:5]{index=5}
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
def call_openai_describe(client: OpenAI, icon_url: str, ctx_url: str, ctx_detail: str) -> dict:
    resp = client.responses.parse(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_TEXT},
                {"type": "input_image", "image_url": icon_url, "detail": ICON_DETAIL},
                {"type": "input_image", "image_url": ctx_url,  "detail": ctx_detail},
            ],
        }],
        text_format=IconDesc,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,  # output 길이 제한은 비용 관리에 핵심 :contentReference[oaicite:6]{index=6}
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
        aliases = [
            a for a in aliases
            if isinstance(a, str) and a and all(ord(c) < 128 for c in a)
        ]
    else:
        aliases = []
    desc_obj["aliases_en"] = aliases

    # nested confidence 제거
    desc_obj = {k: v for k, v in desc_obj.items() if k != "confidence"}
    return desc_obj, conf


# =========================
# Main
# =========================
def main():
    assert INPUT_JSONL.exists(), f"Input not found: {INPUT_JSONL}"
    client = OpenAI()

    done = set()
    if RESUME and OUTPUT_JSONL.exists():
        # 출력에 _id가 없으니, 출력 row도 sample_id로 계산해서 resume
        for out_item in read_jsonl(OUTPUT_JSONL):
            done.add(sample_id(out_item))

    with OUTPUT_JSONL.open("a", encoding="utf-8") as out_fp:
        for item in tqdm(read_jsonl(INPUT_JSONL), desc="regenerating"):
            sid = sample_id(item)
            if sid in done:
                continue

            img_path = Path(item["image"])
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            bbox = item["bbox_xyxy"]
            img = Image.open(img_path).convert("RGB")

            _, icon_img, ctx_img = make_icon_and_context_crops(
                img, bbox, icon_pad_px=ICON_PAD_PX, ctx_pad_scale=CTX_PAD_SCALE
            )

            icon_img = resize_max_side(icon_img, ICON_MAX_SIDE)
            ctx_img  = resize_max_side(ctx_img,  CTX_MAX_SIDE)

            icon_url = pil_to_data_url(icon_img, fmt="JPEG", jpeg_quality=85)
            ctx_url  = pil_to_data_url(ctx_img,  fmt="JPEG", jpeg_quality=80)

            # 1) 기본: ctx_detail=low로 시도 (비용 절감) :contentReference[oaicite:7]{index=7}
            desc_raw = call_with_retry(client, icon_url, ctx_url, ctx_detail=CTX_DETAIL)

            # 2) (옵션) 자신감 낮으면 그 샘플만 high로 재호출
            if ENABLE_CTX_FALLBACK:
                conf_try = float(desc_raw.get("confidence", 0.0))
                if conf_try < FALLBACK_CONF_THRESH:
                    desc_raw = call_with_retry(client, icon_url, ctx_url, ctx_detail="high")

            desc_obj, desc_conf = postprocess_desc(desc_raw)

            out = {
                "image": item["image"],
                "bbox_xyxy": item["bbox_xyxy"],
                "yolo_cls": item.get("yolo_cls", None),
                "description": desc_obj,
                "desc_confidence": desc_conf,
            }

            write_jsonl_line(out_fp, out)
            out_fp.flush()
            done.add(sid)


if __name__ == "__main__":
    main()