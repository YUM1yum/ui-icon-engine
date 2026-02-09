# evaluate.py
import os
import json
import time
from collections import defaultdict
import cv2
import numpy as np
import torch
from tqdm import tqdm

# SBERT
from sentence_transformers import SentenceTransformer

# Optional VLM (Florence-2)
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def _is_jsonl(path: str) -> bool:
    return path.lower().endswith(".jsonl")


def _safe_join_img(img_dir: str, p: str) -> str:
    # p가 절대경로면 그대로, 상대경로면 img_dir 붙임
    if os.path.isabs(p):
        return p
    return os.path.join(img_dir, p) if img_dir else p


def _extract_gt_text(desc):
    """
    screenspot_100_images.jsonl 예시:
      "description": {"name":..., "action":..., "description":...}
    """
    if isinstance(desc, str):
        return desc
    if isinstance(desc, dict):
        # 가장 짧고 안정적인 축: action > description > name
        if desc.get("action"):
            return str(desc["action"])
        if desc.get("description"):
            return str(desc["description"])
        if desc.get("name"):
            return str(desc["name"])
        # fallback
        return json.dumps(desc, ensure_ascii=False)
    return str(desc)


def load_annotations(anno_file: str, img_dir: str):
    """
    지원 포맷:
    1) JSON: list[{image_id, bbox, description}]  (bbox는 xyxy)
    2) JSONL: line마다 {image, bbox_xyxy, description, ...}
    """
    items = []

    if _is_jsonl(anno_file):
        with open(anno_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                img_path = _safe_join_img(img_dir, obj.get("image", ""))
                box = obj.get("bbox_xyxy") or obj.get("bbox")  # xyxy 기대
                if box is None:
                    continue
                gt_text = _extract_gt_text(obj.get("description", ""))
                items.append((img_path, box, gt_text))
        return items

    # json
    with open(anno_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    for obj in data:
        img_id = obj.get("image_id") or obj.get("image") or ""
        img_path = _safe_join_img(img_dir, img_id)
        box = obj.get("bbox") or obj.get("bbox_xyxy")
        if box is None:
            continue
        gt_text = _extract_gt_text(obj.get("description", ""))
        items.append((img_path, box, gt_text))

    return items


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(ay2 - by1, 0.0)
    # ↑ 오타 방지: 원래 코드대로라면 아래 한 줄이 맞습니다.
    # area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    # 그런데 실수 방지 위해 바로 수정합니다.
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crop_with_expand(img_bgr, box_xyxy, expand: float = 3.0):
    x1, y1, x2, y2 = map(float, box_xyxy)
    h, w = img_bgr.shape[:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1), (y2 - y1)
    bw2, bh2 = bw * expand / 2.0, bh * expand / 2.0

    nx1 = int(max(0, cx - bw2))
    ny1 = int(max(0, cy - bh2))
    nx2 = int(min(w, cx + bw2))
    ny2 = int(min(h, cy + bh2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return img_bgr[ny1:ny2, nx1:nx2]


def bgr_to_pil(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def make_side_by_side(icon_bgr, ctx_bgr):
    # 높이 맞춰서 가로로 붙이기
    hi = icon_bgr.shape[0]
    hc = ctx_bgr.shape[0]
    target_h = max(hi, hc)

    def resize_h(im, th):
        h, w = im.shape[:2]
        if h == th:
            return im
        new_w = max(1, int(w * (th / h)))
        return cv2.resize(im, (new_w, th), interpolation=cv2.INTER_AREA)

    icon_r = resize_h(icon_bgr, target_h)
    ctx_r = resize_h(ctx_bgr, target_h)
    return np.concatenate([icon_r, ctx_r], axis=1)


class Florence2Captioner:
    """
    Florence-2 base 로딩에서 SDPA 관련 AttributeError(_supports_sdpa) 나오는 케이스 방지:
      - attn_implementation="eager" 강제
      - transformers 버전에 따라 인자 미지원이면 TypeError로 fallback
    """
    def __init__(self, model_name="microsoft/Florence-2-base", device="cuda"):
        self.device = device
        # fast tokenizer 문제 회피: slow tokenizer 강제
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True, use_fast=False
            )
        except TypeError:
            # transformers 버전에 따라 use_fast 인자를 안 받는 경우 fallback
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )

        dtype = torch.float16 if "cuda" in device else torch.float32

        kwargs = dict(
            trust_remote_code=True,
            dtype=dtype,                 # torch_dtype 대신 dtype
            low_cpu_mem_usage=True,
        )

        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")

        # SDPA/flash 관련 충돌 방지
        # (지원 안 하면 TypeError 발생 → fallback)
        try:
            kwargs["attn_implementation"] = "eager"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)

        self.model.eval()

    @torch.inference_mode()
    def caption(self, pil_image: Image.Image, prompt: str, max_new_tokens: int = 24):
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return out.strip()


def evaluate_dataset(
    anno_file: str,
    img_dir: str,
    engine,
    tau: float = 0.75,
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    conf_thres: float = 0.4,
    captioner: str = "svlm",  # "svlm" | "florence2"
    iou_thr: float = 0.5,
    context_expand: float = 3.0,
    vlm_model: str = "microsoft/Florence-2-base",
    vlm_prompt: str = "Describe the UI icon and its function in a short phrase.",
    vlm_max_new_tokens: int = 24,
    out_jsonl: str = "",
    limit: int = 0,  # 0이면 전체
):
    # 1) Load annotations
    ann = load_annotations(anno_file, img_dir)
    if limit and limit > 0:
        ann = ann[:limit]

    if len(ann) == 0:
        print("[Eval] No annotations loaded.")
        return None

    # 2) Group by image
    grouped = defaultdict(list)
    for img_path, gt_box, gt_text in ann:
        grouped[img_path].append((gt_box, gt_text))

    img_paths = list(grouped.keys())
    print(f"[Eval] images={len(img_paths)} | gt_items={len(ann)} | captioner={captioner}")

    # 3) Init SBERT
    sbert = SentenceTransformer(sbert_model, device=("cuda" if torch.cuda.is_available() else "cpu"))
    sbert.eval()

    # 4) Init VLM captioner (optional)
    vlm = None
    if captioner.lower() == "florence2":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vlm = Florence2Captioner(model_name=vlm_model, device=device)

    # 5) Metrics accumulators
    n_gt = 0
    n_det_hit = 0
    ious = []

    pred_texts = []
    gt_texts = []

    t_engine_ms = []
    t_vlm_ms = []

    out_f = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None

    for img_path in tqdm(img_paths, desc="Eval"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # engine 1회
        t0 = time.perf_counter()
        results = engine.process_frame(img, conf_thres=conf_thres)
        t1 = time.perf_counter()
        t_engine_ms.append((t1 - t0) * 1000.0)

        # predicted boxes / texts
        pred_boxes = []
        pred_svlm_text = []
        for r in results:
            pred_boxes.append(r["box"])
            pred_svlm_text.append(r.get("description", ""))

        # VLM captions: pred box마다 1회만 (캐시)
        pred_vlm_text = [""] * len(pred_boxes)
        if vlm is not None and len(pred_boxes) > 0:
            for i, box in enumerate(pred_boxes):
                icon = crop_with_expand(img, box, expand=1.0)
                if icon is None:
                    continue
                ctx = crop_with_expand(img, box, expand=context_expand)
                if ctx is None:
                    ctx = icon
                combo = make_side_by_side(icon, ctx)
                pil = bgr_to_pil(combo)

                tv0 = time.perf_counter()
                txt = vlm.caption(pil, prompt=vlm_prompt, max_new_tokens=vlm_max_new_tokens)
                tv1 = time.perf_counter()
                t_vlm_ms.append((tv1 - tv0) * 1000.0)

                pred_vlm_text[i] = txt

        # match each GT to best predicted
        for gt_box, gt_text in grouped[img_path]:
            n_gt += 1
            best_iou = 0.0
            best_idx = -1
            for i, pb in enumerate(pred_boxes):
                v = iou_xyxy(gt_box, pb)
                if v > best_iou:
                    best_iou = v
                    best_idx = i

            ious.append(best_iou)

            if best_iou >= iou_thr and best_idx >= 0:
                n_det_hit += 1
                if captioner.lower() == "florence2":
                    pred = pred_vlm_text[best_idx]
                else:
                    pred = pred_svlm_text[best_idx]

                pred_texts.append(pred)
                gt_texts.append(gt_text)

                if out_f:
                    out_obj = {
                        "image": img_path,
                        "gt_box": gt_box,
                        "gt_text": gt_text,
                        "pred_box": pred_boxes[best_idx],
                        "pred_text": pred,
                        "iou": float(best_iou),
                        "captioner": captioner,
                    }
                    out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    if out_f:
        out_f.close()

    # 6) Compute SBERT similarities (only matched)
    desc_acc = 0.0
    mean_sim = 0.0
    if len(pred_texts) > 0:
        with torch.inference_mode():
            emb_p = sbert.encode(pred_texts, convert_to_tensor=True, normalize_embeddings=True)
            emb_g = sbert.encode(gt_texts, convert_to_tensor=True, normalize_embeddings=True)
            sims = (emb_p * emb_g).sum(dim=1)  # cosine (normalized)
            mean_sim = float(sims.mean().item())
            desc_acc = float((sims >= tau).float().mean().item())

    det_recall = (n_det_hit / n_gt) if n_gt > 0 else 0.0
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0

    scores = {
        "n_images": len(img_paths),
        "n_gt": n_gt,
        "det_recall@iou": det_recall,
        "mean_iou": mean_iou,
        "n_matched": len(pred_texts),
        "desc_acc@tau": float(desc_acc),
        "desc_mean_sbert": float(mean_sim),
        "engine_ms_per_image": float(np.mean(t_engine_ms)) if t_engine_ms else 0.0,
        "vlm_ms_per_icon": float(np.mean(t_vlm_ms)) if t_vlm_ms else 0.0,
        "captioner": captioner,
        "tau": float(tau),
        "iou_thr": float(iou_thr),
    }

    print("\n--- Eval Results ---")
    for k, v in scores.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return scores
