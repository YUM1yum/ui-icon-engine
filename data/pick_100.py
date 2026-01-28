import json
import shutil
from pathlib import Path

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"[{path}] JSON decode error at line {line_no}: {e}")
    return rows

def dedup_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def to_rel_path(image_value: str, anchors=("images", "train", "val", "test")) -> Path:
    p = Path(image_value.replace("\\", "/"))
    parts = [str(x) for x in p.parts]

    for a in anchors:
        if a in parts:
            i = parts.index(a)
            return Path(*parts[i:])

    return Path(p.name)

def strip_prefix(path: Path, prefix: Path) -> Path:
    """path가 prefix로 시작하면 그 prefix를 제거한 나머지 반환"""
    p_parts = list(path.parts)
    pre_parts = list(prefix.parts)
    if len(p_parts) >= len(pre_parts) and p_parts[:len(pre_parts)] == pre_parts:
        return Path(*p_parts[len(pre_parts):])
    return path

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def resolve_under_root(root: Path, rel: Path, auto_prefix: Path) -> Path:
    """
    rel 이 'images/train/xxx' 같은 형태일 때,
    root가 이미 .../images/train 이면 rel에서 images/train을 제거해서 붙이고,
    root가 dataset root(.../web-ui-600/web-ui-600)면 rel 그대로 붙임.
    auto_prefix는 보통 Path("images") 또는 Path("images/train") 등.
    """
    # root 아래에 auto_prefix가 존재하면: dataset root로 간주 -> 그대로 붙임
    if (root / auto_prefix).exists():
        return root / rel

    # root 자체가 이미 auto_prefix 위치일 가능성: rel에서 auto_prefix 제거
    rel2 = strip_prefix(rel, auto_prefix)
    return root / rel2

def rel_image_to_rel_label(rel_img: Path) -> Path:
    # images/train/x.png -> labels/train/x.txt
    parts = list(rel_img.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
    return Path(*parts).with_suffix(".txt")

def extract_by_relative_paths(
    sampled_jsonl: str,
    images_root: str,
    labels_root: str,
    out_dir: str,
    image_key: str = "image",
    anchors=("images", "train", "val", "test"),
):
    sampled_jsonl = Path(sampled_jsonl)
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    out_dir = Path(out_dir)

    out_img_root = out_dir / "images"
    out_lbl_root = out_dir / "labels"

    rows = load_jsonl(str(sampled_jsonl))

    rel_imgs = [to_rel_path(r[image_key], anchors=anchors) for r in rows]
    rel_imgs = dedup_preserve_order([Path(str(p).replace("\\", "/")) for p in rel_imgs])

    copied_img = copied_lbl = 0
    missing_img = missing_lbl = 0

    # ✅ 핵심: rel이 images/train/...로 나오므로 auto_prefix를 images/train로 둠
    img_prefix = Path("images") / "train" if any(str(p).replace("\\", "/").startswith("images/train") for p in rel_imgs) else Path("images")
    # 라벨도 보통 labels/train 이므로 동일하게 train 기반으로 처리
    lbl_prefix = Path("labels") / "train"

    for rel_img in rel_imgs:
        # 이미지 실제 경로 결정(중복 prefix 자동 보정)
        src_img = resolve_under_root(images_root, rel_img, auto_prefix=img_prefix)

        # 출력은 항상 상대경로 구조 유지 (images/train/... 그대로 저장)
        dst_img = out_img_root / rel_img

        if not src_img.exists():
            print(f"[MISSING IMAGE] {rel_img} (expected: {src_img})")
            missing_img += 1
            continue

        safe_copy(src_img, dst_img)
        copied_img += 1

        # 라벨 상대경로
        rel_lbl = rel_image_to_rel_label(rel_img)  # labels/train/x.txt 또는 train/x.txt

        # labels_root가 dataset root인지, labels/train인지에 따라 자동 보정
        # rel_lbl이 labels/train/... 이면 prefix=labels/train로 strip 가능
        src_lbl = resolve_under_root(labels_root, rel_lbl, auto_prefix=lbl_prefix)

        # 출력은 labels/train/... 구조로 저장하되, rel_lbl이 labels/...면 그대로
        dst_lbl = out_lbl_root / rel_lbl

        if not src_lbl.exists():
            print(f"[MISSING LABEL] {rel_lbl} (expected: {src_lbl})")
            missing_lbl += 1
        else:
            safe_copy(src_lbl, dst_lbl)
            copied_lbl += 1

    print(f"\n[DONE] {sampled_jsonl}")
    print(f"  unique images in jsonl: {len(rel_imgs)}")
    print(f"  copied images: {copied_img} | missing images: {missing_img}")
    print(f"  copied labels: {copied_lbl} | missing labels: {missing_lbl}")
    print(f"  output dir: {out_dir.resolve()}")

if __name__ == "__main__":
    sampled1 = r"./screenspot_100_images.jsonl"
    sampled2 = r"./webui_100_images.jsonl"

    # ✅ 아래를 “둘 중 아무거나”로 넣어도 동작하게 만든 게 포인트입니다.
    # (A) dataset root
    # webui_images_root  = r"C:\Users\dldna\icon\dataset\web-ui-600\web-ui-600"
    # webui_labels_root  = r"C:\Users\dldna\icon\dataset\web-ui-600\web-ui-600"
    #
    # (B) images/train 과 labels/train 까지 들어간 경로
    webui_images_root  = r"C:\\Users\\dldna\\icon\\dataset\\web-ui-600\\web-ui-600\\images\\train"
    webui_labels_root  = r"C:\\Users\\dldna\\icon\\dataset\\web-ui-600\\web-ui-600\\labels\\train"

    screenspot_images_root = r"C:\\Users\\dldna\\icon\\dataset\\screenspot\\images\\train"
    screenspot_labels_root = r"C:\\Users\\dldna\\icon\\dataset\\screenspot\\labels\\train"
    extract_by_relative_paths(
        sampled_jsonl=sampled2,
        images_root=webui_images_root,
        labels_root=webui_labels_root,
        out_dir=r"./extracted/webui_100",
    )

    extract_by_relative_paths(
        sampled_jsonl=sampled1,
        images_root=screenspot_images_root,
        labels_root=screenspot_labels_root,
        out_dir=r"./extracted/screenspot_100",
    )
