import argparse
import cv2
import torch
import os
import time
import glob
from inference.engine import UIInferenceEngine


def draw_results(image, results, timing_text=None):
    """
    이미지에 Bounding Box와 텍스트를 그리는 함수
    + timing_text가 있으면 중앙에 시간 오버레이
    """
    vis_img = image.copy()

    for res in results:
        box = res["box"]  # [x1, y1, x2, y2]
        text = res.get("description", "")
        conf = res.get("confidence", None)

        x1, y1, x2, y2 = map(int, box)

        # 1) Box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 2) Label
        conf_str = f"{conf:.2f}" if isinstance(conf, (float, int)) else "NA"
        label = f"{text} ({conf_str})"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        y_text_top = max(0, y1 - 20)
        cv2.rectangle(vis_img, (x1, y_text_top), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(
            vis_img,
            label,
            (x1, max(10, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    if timing_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3

        (tw, th), baseline = cv2.getTextSize(timing_text, font, font_scale, thickness)
        H, W = vis_img.shape[:2]
        x = (W - tw) // 2
        y = (H + th) // 2

        pad = 10
        cv2.rectangle(
            vis_img,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad),
            (0, 0, 0),
            -1,
        )
        cv2.putText(vis_img, timing_text, (x, y), font, font_scale, (0, 255, 0), thickness)

    return vis_img


def list_images(folder: str, pattern: str):
    # glob 결과 중 이미지 확장자만 필터
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    paths = glob.glob(os.path.join(folder, pattern))
    paths = [p for p in paths if os.path.splitext(p)[1].lower() in exts]
    paths.sort()
    return paths


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def process_one_image(engine, img_path: str, out_path: str, conf_thres: float):
    """
    이미지 1장 처리 + (read, infer, read+infer, read+infer+save) 타이밍 출력
    draw(그리기) 시간은 제외
    """
    print(f"Processing: {img_path}")

    # 1) Read
    t_read0 = time.perf_counter()
    img = cv2.imread(img_path)
    t_read1 = time.perf_counter()
    read_ms = (t_read1 - t_read0) * 1000.0

    if img is None:
        print(f"  - skip: failed to read (read={read_ms:.1f} ms)")
        return False

    # 2) Inference
    t_inf0 = time.perf_counter()
    results = engine.process_frame(img, conf_thres=conf_thres)
    t_inf1 = time.perf_counter()
    infer_ms = (t_inf1 - t_inf0) * 1000.0

    # draw는 시간에서 제외 (하지만 결과 저장을 위해 이미지는 그려야 함)
    timing_text = f"read {read_ms:.1f} ms | infer {infer_ms:.1f} ms | n={len(results)}"
    vis_img = draw_results(img, results, timing_text=timing_text)

    # 3) Save only
    t_save0 = time.perf_counter()
    ok = cv2.imwrite(out_path, vis_img)
    t_save1 = time.perf_counter()
    save_ms = (t_save1 - t_save0) * 1000.0

    read_infer_ms = read_ms + infer_ms
    read_infer_save_ms = read_infer_ms + save_ms

    if not ok:
        print(f"  - save failed: {out_path}")
        print(f"[Time] read: {read_ms:.1f} ms | infer: {infer_ms:.1f} ms | read+infer: {read_infer_ms:.1f} ms | read+infer+save: {read_infer_save_ms:.1f} ms")
        return False

    print(f"  - saved: {out_path}")
    print(f"[Time] read: {read_ms:.1f} ms | infer: {infer_ms:.1f} ms | read+infer: {read_infer_ms:.1f} ms | read+infer+save: {read_infer_save_ms:.1f} ms")
    return True


def main():
    parser = argparse.ArgumentParser(description="UI Icon Detection & Description Engine")
    parser.add_argument("--input", type=str, required=True, help="Path to input image OR folder")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output file (single) OR output folder (dir input)")
    parser.add_argument("--yolo_path", type=str, default="best.pt", help="YOLO weights path")
    parser.add_argument("--fusion_path", type=str, default="checkpoints/fusion_best.pth", help="Fusion model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument(
        "--glob",
        type=str,
        default="*.png",
        help="When --input is a folder, pattern like *.png, *.jpg, *.*",
    )
    args = parser.parse_args()

    input_is_dir = os.path.isdir(args.input)

    # 1) Engine init (once)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_init0 = time.perf_counter()
    engine = UIInferenceEngine(
        yolo_path=args.yolo_path,
        fusion_checkpoint=args.fusion_path,
        device=device,
    )
    t_init1 = time.perf_counter()
    print(f"[Time] init engine: {(t_init1 - t_init0) * 1000:.1f} ms  (device={device})")

    if input_is_dir:
        # input folder -> output folder
        in_dir = args.input
        out_dir = args.output
        ensure_dir(out_dir)

        img_paths = list_images(in_dir, args.glob)
        if len(img_paths) == 0:
            print(f"No images found in folder: {in_dir} (pattern={args.glob})")
            return

        print(f"Found {len(img_paths)} images in folder: {in_dir}")
        print(f"Output folder: {out_dir}")

        total_saved = 0
        for idx, img_path in enumerate(img_paths):
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{base}_out.jpg")

            print(f"\n[{idx+1}/{len(img_paths)}]")
            ok = process_one_image(engine, img_path, out_path, conf_thres=args.conf)
            total_saved += int(ok)

        print(f"\nDone. Saved {total_saved}/{len(img_paths)} images.")

    else:
        # single file -> output file
        if not os.path.exists(args.input):
            print(f"Error: Image not found at {args.input}")
            return

        out_dir = os.path.dirname(args.output)
        ensure_dir(out_dir)

        _ = process_one_image(engine, args.input, args.output, conf_thres=args.conf)


if __name__ == "__main__":
    main()