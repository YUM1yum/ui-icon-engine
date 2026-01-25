import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T


class UIDescriptionDataset(Dataset):
    """
    JSONL format per line:
    {
        "image": "C:\\Users\\...\\xxx.png",
        "bbox_xyxy": [x1, y1, x2, y2],
        "label": "download",
        "description": "A downward arrow icon ... likely used to download.",
        ...
    }
    Each line = one training sample (one box + one label).
    """

    def __init__(self, annotation_file, img_dir=None, tokenizer=None, img_size=640):
        self.annotation_file = annotation_file
        self.img_dir = img_dir  # not used if "image" is absolute
        self.tokenizer = tokenizer
        self.img_size = img_size

        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided to UIDescriptionDataset")

        self.data = self._load_jsonl(annotation_file)

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def _load_jsonl(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = item.get("image")
        if not img_path:
            raise KeyError('JSONL item must contain key "image"')

        # image path can be absolute; if not, join with img_dir
        if self.img_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.img_dir, img_path)

        bbox = item.get("bbox_xyxy")
        if bbox is None:
            raise KeyError('JSONL item must contain key "bbox_xyxy"')

        label = item.get("label")
        if label is None:
            raise KeyError('JSONL item must contain key "label"')

        # --- load image ---
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size

        # --- resize image ---
        img_resized = self.transform(img)  # [3, img_size, img_size]

        # --- scale bbox to resized coord system ---
        x1, y1, x2, y2 = bbox
        sx = self.img_size / float(w0)
        sy = self.img_size / float(h0)

        x1 = x1 * sx
        x2 = x2 * sx
        y1 = y1 * sy
        y2 = y2 * sy

        # clamp to valid range
        x1 = max(0.0, min(float(self.img_size - 1), float(x1)))
        x2 = max(0.0, min(float(self.img_size - 1), float(x2)))
        y1 = max(0.0, min(float(self.img_size - 1), float(y1)))
        y2 = max(0.0, min(float(self.img_size - 1), float(y2)))

        # ensure x1<=x2, y1<=y2
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        bbox_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        # --- tokenize label as target text ---
        # tokenizer.encode returns list[int]
        token_ids = self.tokenizer.encode(label)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        return img_resized, bbox_tensor, token_tensor


def make_collate_fn(pad_id: int):
    def _collate(batch):
        images, boxes, texts = zip(*batch)
        images = torch.stack(images, dim=0)            # [B,3,H,W]
        boxes = torch.stack(boxes, dim=0)              # [B,4]
        texts = pad_sequence(texts, batch_first=True, padding_value=pad_id)  # [B,L]
        return images, boxes, texts
    return _collate