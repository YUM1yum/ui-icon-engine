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
    """
    jsonl의 image 값(절대/상대)을 '상대경로'로 정규화.
    - anchors 중 하나가 경로 파트에 있으면, 그 파트부터 끝까지를 상대경로로 사용
      예: C:/.../web-ui-600/images/train/a.png -> images/train/a.png
      예: .../train/a.png -> train/a.png
    - anchors가 전혀 없으면, 그냥 파일명만 반환(마지막 fallback)
    """
    p = Path(image_value.replace("\\", "/"))
    parts = list(p.parts)

    # Path("C:/...") 는 parts에 "C:" 같은 게 들어갈 수 있어서 문자열로 처리
    parts_str = [str(x) for x in parts]

    for a in anchors:
        if a in parts_str:
            i = parts_str.index(a)
            return Path(*parts_str[i:])

    return Path(p.name)

def rel_image_to_rel_label(rel_img: Path) -> Path:
    """
    images/train/xxx.png -> labels/train/xxx.txt (images가 있으면 labels로 치환)
    train/xxx.png       -> train/xxx.txt (그냥 확장자만 txt로)
    """
    parts = list(rel_img.parts)
    if parts and parts[0] == "images":
        parts[0] = "labels"
    rel_lbl = Path(*parts).with_suffix(".txt")
    return rel_lbl

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

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

    # jsonl 내 image 상대경로 목록(중복 제거)
    rel_imgs = [to_rel_path(r[image_key], anchors=anchors) for r in rows]
    rel_imgs = dedup_preserve_order([str(p).replace("\\", "/") for p in rel_imgs])
    rel_imgs = [Path(p) for p in rel_imgs]

    copied_img = copied_lbl = 0
    missing_img = missing_lbl = 0

    for rel_img in rel_imgs:
        src_img = images_root / rel_img
        dst_img = out_img_root / rel_img

        if not src_img.exists():
            print(f"[MISSING IMAGE] {rel_img} (expected: {src_img})")
            missing_img += 1
            continue

        safe_copy(src_img, dst_img)
        copied_img += 1

        # 라벨 상대경로 생성: images -> labels 치환 + .txt
        rel_lbl = rel_image_to_rel_label(rel_img)

        # rel_lbl이 이미 labels/... 형태면 labels_root에 바로 붙이고,
        # rel_lbl이 train/... 형태면 labels_root/train/... 로 붙음
        # 단, rel_lbl이 labels/... 로 시작하면 labels_root/labels/...가 되지 않게 보정
        if rel_lbl.parts and rel_lbl.parts[0] == "labels":
            src_lbl = labels_root / Path(*rel_lbl.parts[1:])
            dst_lbl = out_lbl_root / Path(*rel_lbl.parts[1:])
        else:
            src_lbl = labels_root / rel_lbl
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
    # 샘플링된 jsonl
    sampled1 = r"./screenspot_100_images.jsonl"
    sampled2 = r"./webui_100_images.jsonl"

    # ✅ 원본 루트 경로만 지정
    # 예) images_root 아래에 images/train/... 구조가 존재하거나, train/... 구조가 존재
    screenspot_images_root = r"C:\\Users\\dldna\\icon\\dataset\\screenspot\\images\\train"      # 예: D:\dataset\screenspot\images\train\...
    screenspot_labels_root = r"C:\\Users\\dldna\\icon\\dataset\\screenspot\\labels\\train"      # 예: D:\dataset\screenspot\labels\train\...

    webui_images_root      = r"C:\\Users\\dldna\\icon\\dataset\\web-ui-600\\web-ui-600\\images\\train"      # 예: D:\dataset\web-ui-600\images\train\...
    webui_labels_root      = r"C:\\Users\\dldna\\icon\\dataset\\web-ui-600\\web-ui-600\\labels\\train"      # 예: D:\dataset\web-ui-600\labels\train\...

    # anchors: jsonl의 image 값에서 어느 지점부터 상대경로로 볼지 기준
    anchors = ("images", "train", "val", "test")

    extract_by_relative_paths(
        sampled_jsonl=sampled1,
        images_root=screenspot_images_root,
        labels_root=screenspot_labels_root,
        out_dir=r"./extracted/screenspot_100",
        image_key="image",
        anchors=anchors,
    )

    extract_by_relative_paths(
        sampled_jsonl=sampled2,
        images_root=webui_images_root,
        labels_root=webui_labels_root,
        out_dir=r"./extracted/webui_100",
        image_key="image",
        anchors=anchors,
    )



# import json
# import random
# from collections import defaultdict
# from pathlib import Path

# def load_jsonl(path: str):
#     rows = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, 1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rows.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 raise ValueError(f"[{path}] JSON decode error at line {line_no}: {e}")
#     return rows

# def sample_100_images_to_jsonl(
#     in_path: str,
#     out_path: str,
#     n_images: int = 100,
#     image_key: str = "image",
#     seed: int = 42,
# ):
#     rows = load_jsonl(in_path)

#     # image 경로별로 레코드(아이콘) 묶기
#     by_image = defaultdict(list)
#     for r in rows:
#         if image_key not in r:
#             raise KeyError(f'Key "{image_key}" not found. (change image_key)')
#         by_image[r[image_key]].append(r)

#     images = list(by_image.keys())
#     if len(images) == 0:
#         raise ValueError(f"No images found in {in_path}")

#     k = min(n_images, len(images))
#     rnd = random.Random(seed)
#     sampled_images = rnd.sample(images, k)

#     # 샘플된 100장에 속한 모든 레코드만 모으기
#     sampled_rows = []
#     for img in sampled_images:
#         sampled_rows.extend(by_image[img])

#     # 저장
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as f:
#         for r in sampled_rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

#     print(f"[OK] {in_path}")
#     print(f"  total unique images: {len(images)}")
#     print(f"  sampled images: {k}")
#     print(f"  output rows: {len(sampled_rows)}")
#     print(f"  saved to: {out_path.resolve()}")

# if __name__ == "__main__":
#     # 입력 파일
#     file1 = r"./screenspot_descriptions_openai.jsonl"
#     file2 = r"./web-ui_descriptions_openai_.jsonl"

#     # 출력 파일 (원하는 경로로 바꾸셔도 됩니다)
#     out1 = r"./screenspot_100_images.jsonl"
#     out2 = r"./webui_100_images.jsonl"

#     # 각 파일에서 독립적으로 100장씩 샘플링
#     sample_100_images_to_jsonl(file1, out1, n_images=100, image_key="image", seed=42)
#     sample_100_images_to_jsonl(file2, out2, n_images=100, image_key="image", seed=43)