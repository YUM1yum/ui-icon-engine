import os, json, glob, re
from PIL import Image
import numpy as np

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ---- 설정 ----
IMAGE_DIR = r"C:\\Users\\dldna\\icon\\screenspot\\images\\train"
LABEL_DIR = r"C:\\Users\\dldna\\icon\\screenspot\\labels\\train"   # yolo txt 폴더
OUT_PATH  = r"C:\\Users\\dldna\\icon\\screenspot\\ui-icon-engine\\out_icon_labels.jsonl"

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # :contentReference[oaicite:1]{index=1}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONF_PLACEHOLDER = None  # yolo txt에 conf가 없으니 비워둠

'''
# 라벨 후보(초기 버전: 자주 쓰는 UI 아이콘 중심)
LABEL_SET = [
    "settings","search","close","back","home","menu","cart","share","download","upload",
    "profile","user","notification","bell","favorite","like","heart","delete","trash",
    "edit","pencil","plus","minus","check","arrow","play","pause","stop","next","prev",
    "refresh","filter","sort","info","help","warning","lock","unlock","camera","image"
]
LABEL_SET_LOWER = set(LABEL_SET)
'''

def yolo_to_xyxy(cls, xc, yc, w, h, W, H):
    x1 = (xc - w/2) * W
    y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W
    y2 = (yc + h/2) * H
    return [x1, y1, x2, y2]

def clamp_box(box, W, H):
    x1,y1,x2,y2 = box
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2))
    y2 = max(0, min(H-1, y2))
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
    return [x1,y1,x2,y2]

def pad_box(box, pad, W, H):
    x1,y1,x2,y2 = box
    return clamp_box([x1-pad, y1-pad, x2+pad, y2+pad], W, H)

def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9 _-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 1~3 단어로 자르기
    parts = text.split(" ")
    text = " ".join(parts[:3]).strip()
    return text

def load_yolo_txt(txt_path):
    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            items.append((cls, xc, yc, w, h))
    return items

def build_prompt():
    return (
        "You are labeling a UI icon for a dataset.\n"
        "You will be given (1) a cropped ICON image and (2) a UI CONTEXT image around it.\n\n"
        "Task:\n"
        "- Output a short English label for the icon.\n"
        "- Prefer a common UI name (e.g., settings, search, close, back, menu, download).\n"
        "- If the icon is unclear, use the context to guess.\n\n"
        "Output rules (STRICT):\n"
        "- Output ONLY the label text.\n"
        "- Lowercase English.\n"
        "- 1 to 5 words maximum.\n"
        "- No punctuation, no quotes, no extra commentary.\n"
    )

@torch.no_grad()
def qwen_label(model, processor, icon_img: Image.Image, ctx_img: Image.Image):
    prompt = build_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt + "\nIcon image and its UI context are provided."},
                {"type": "image", "image": icon_img},
                {"type": "image", "image": ctx_img},
                {"type": "text", "text": "Return only the short label text."},
            ],
        }
    ]

    # 1) 먼저 chat template로 "문자열" 만들기
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2) 그 다음 processor로 (텍스트 + 이미지들) 인코딩해서 dict 만들기
    inputs = processor(
        text=[text],
        images=[icon_img, ctx_img],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
    )

    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    last_line = decoded.strip().splitlines()[-1]
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    print("RAW:", repr(decoded))
    lab = normalize_label(last_line) or "unknown"
    return lab

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("Loading model:", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()

    img_paths = glob.glob(os.path.join(IMAGE_DIR, "*.*"))
    img_paths = [p for p in img_paths if os.path.splitext(p)[1].lower() in [".jpg",".jpeg",".png",".webp",".bmp"]]
    print("Images:", len(img_paths))

    n_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as w:
        for img_path in img_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(LABEL_DIR, stem + ".txt")
            if not os.path.exists(txt_path):
                continue

            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            yolo_items = load_yolo_txt(txt_path)
            if not yolo_items:
                continue

            for idx, (cls, xc, yc, bw, bh) in enumerate(yolo_items):
                box = yolo_to_xyxy(cls, xc, yc, bw, bh, W, H)
                box = clamp_box(box, W, H)

                # icon-only crop: bbox에 약간 패딩
                icon_box = pad_box(box, pad=2, W=W, H=H)
                icon = img.crop(tuple(map(int, icon_box)))

                # context crop: bbox의 3~5배 정도 영역 (너무 크면 의미가 흐려지니 제한)
                pad = int(max((box[2]-box[0]), (box[3]-box[1])) * 2.0)
                ctx_box = pad_box(box, pad=pad, W=W, H=H)
                ctx = img.crop(tuple(map(int, ctx_box)))

                label = qwen_label(model, processor, icon, ctx)

                if n_written % 20 == 0:
                    print(f"written={n_written} last_label={label} img={os.path.basename(img_path)}")

                rec = {
                    "image": img_path,
                    "bbox_xyxy": list(map(int, box)),
                    "yolo_cls": cls,
                    "label": label,
                    "confidence": CONF_PLACEHOLDER,
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

            if n_written % 200 == 0 and n_written > 0:
                print("written:", n_written)

    print("Done. Wrote:", n_written, "->", OUT_PATH)

if __name__ == "__main__":
    main()