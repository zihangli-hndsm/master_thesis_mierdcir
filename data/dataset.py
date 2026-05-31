from torch.utils.data import Dataset
from PIL import Image
import os
import json
from transformers import CLIPTokenizerFast
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
import lmdb
from io import BytesIO
import ast

hf_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")


def segment_lines(lines, max_length):
    # 裁剪过长的句子，以适应clip tokenizer
    segment = []
    segment_word_count = 0
    segment_char_count = 0
    for i, line in enumerate(lines):
        segment.append(line.strip('.'))
        segment_char_count = len('. '.join(segment))
        if segment_char_count > 500:
            return segment[:-1]
        segment_word_count = len(hf_tokenizer('. '.join(segment)))
        if segment_word_count >= max_length:
            return segment[:-1]
    return segment


class MTCIRDataset(Dataset):
    def __init__(self, data_path, json_path, lmdb_path, transform=None):
        self.data = []  # 加载 JSON 文件
        self.transform = transform
        self.root = data_path
        self.image_root = os.path.join(data_path, "MTCIR/images")
        self.jsonl_path = os.path.join(data_path, json_path)
        self.lmdb_path = lmdb_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.env = None
        print(f"Loading metadata from {self.jsonl_path}...")
        # 逐行读取 JSONL
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} samples.")
        # 假设 data 是一个列表，每项是 {'ref_img': 'a.jpg', 'text': 'change to red', 'target_img': 'b.jpg'}
        
    def _init_db(self):
        # 关键：每个 worker 进程第一次读取时初始化 env
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()
        try:
            item = self.data[idx]
            # 从lmdb数据库中读取图片
            ref_img_key = item['image'].encode('ascii')
            target_img_key = item['target_img'].encode('ascii')
            with self.env.begin(write=False, buffers=True) as txn:
                ref_img_buf = txn.get(ref_img_key)
                target_img_buf = txn.get(target_img_key)
            ref_img = Image.open(BytesIO(ref_img_buf)).convert('RGB')
            target_img = Image.open(BytesIO(target_img_buf)).convert('RGB')
            
            ref_path = os.path.join(self.image_root, item['image'])
            target_path = os.path.join(self.image_root, item['target_img'])
            
            if self.transform:
                ref_img = self.transform(ref_img)
                target_img = self.transform(target_img)

            return {
                'id': item['id'],
                'image': ref_img,
                'target_path': item['target_img'],
                'target_img': target_img,
                'text': ". ".join(segment_lines(item['modification'], max_length=77)),
                'np': item['nps']
            }

        except Exception as e:
            # 红队建议：训练时不要 print 错误，否则日志会爆炸。
            # 直接换一个索引重试 (简单的容错策略)
            print(f"Warning: Error loading index {idx}: {e}")
            assert False
            new_idx = (idx + 1) % len(self.data)
            return self.__getitem__(new_idx)

    def custom_collate_fn(self, batch):
        """
        batch: 是一个列表，每个元素是 __getitem__ 返回的字典
        """
        # 提取所有 np 字段，不让它们进入 default_collate
        nps = [item.pop('np') for item in batch]

        # 使用官方默认逻辑处理剩余的字段 (id, image, target_img, text)
        # 字符串会被处理成 list of strings，Tensor 会被叠成 [B, C, H, W]
        try:
            collated_batch = default_collate(batch)
        except RuntimeError as e:
            # 如果这里还报错，说明 image 或其他字段尺寸也不统一
            print("Collate error: 请检查图片尺寸是否统一 Resize 过")
            raise e

        # 把处理好的 nps 放回去
        collated_batch['np'] = nps

        return collated_batch


class MerdCIRDataset(Dataset):
    def __init__(self, data_path, json_path, lmdb_path, transform=None):
        self.data = []
        self.root = data_path
        self.jsonl_path = os.path.join(data_path, json_path)
        self.lmdb_path = lmdb_path
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.env = None
        print(f"Loading metadata from {self.jsonl_path}...")

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} samples.")

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _resolve_target_key(item):
        return item.get('target_img') or item.get('target_image')

    @staticmethod
    def _resolve_text(item):
        rewrite_text = item.get('merdcir_modification', '')
        if isinstance(rewrite_text, list):
            rewrite_text = ". ".join(rewrite_text)
        if not rewrite_text:
            raw_text = item.get('modification') or item.get('modifications') or ''
            if isinstance(raw_text, list):
                return ". ".join(segment_lines(raw_text, max_length=77))
            return raw_text
        return rewrite_text

    @staticmethod
    def _resolve_nps(item):
        return item.get('nps', [])

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        item = self.data[idx]
        ref_key = item['image']
        target_key = self._resolve_target_key(item)
        if target_key is None:
            raise KeyError("Missing target image key: expected `target_img` or `target_image`.")

        with self.env.begin(write=False, buffers=True) as txn:
            ref_img_buf = txn.get(ref_key.encode('ascii'))
            target_img_buf = txn.get(target_key.encode('ascii'))

        if ref_img_buf is None or target_img_buf is None:
            raise ValueError(f"Image not found in lmdb. ref={ref_key}, target={target_key}")

        ref_img = Image.open(BytesIO(ref_img_buf)).convert('RGB')
        target_img = Image.open(BytesIO(target_img_buf)).convert('RGB')

        if self.transform:
            ref_img = self.transform(ref_img)
            target_img = self.transform(target_img)

        return {
            'id': item.get('id', idx),
            'image': ref_img,
            'target_path': target_key,
            'target_img': target_img,
            'text': self._resolve_text(item),
            'np': self._resolve_nps(item)
        }

    def custom_collate_fn(self, batch):
        nps = [item.pop('np') for item in batch]
        collated_batch = default_collate(batch)
        collated_batch['np'] = nps
        return collated_batch


class LaSCoDataset(Dataset):
    def __init__(self, data_path, json_path, lmdb_path=None, transform=None, image_root=None):
        self.data = []
        self.root = data_path
        self.json_path = os.path.join(data_path, json_path) if not os.path.isabs(json_path) else json_path
        self.image_root = image_root or data_path
        self.lmdb_path = lmdb_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.env = None
        print(f"Loading metadata from {self.json_path}...")
        self.data = self._load_annotations(self.json_path)
        print(f"Loaded {len(self.data)} samples.")

    @staticmethod
    def _load_annotations(path):
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in f if line.strip()]

            text = f.read()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = ast.literal_eval(text)

        if isinstance(payload, dict):
            for key in ("data", "annotations", "queries", "triplets"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported LaSCo annotation format in {path}")
        return payload

    def _init_db(self):
        if not self.lmdb_path:
            return
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    @staticmethod
    def _image_path(value):
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                return str(value[1])
            if len(value) == 1:
                return str(value[0])
        if isinstance(value, dict):
            for key in ("path", "file_name", "image", "image_path"):
                if key in value:
                    return str(value[key])
        return str(value)

    @staticmethod
    def _text_spans(text, max_length=77):
        words = [word for word in str(text).replace(".", " ").split() if word]
        spans = []
        for idx, word in enumerate(words[:max_length], start=1):
            spans.append([word, [idx, idx]])
        if not spans and text:
            spans.append([str(text), [1, 1]])
        return spans

    def _read_image(self, image_path):
        normalized = image_path.replace("\\", "/").lstrip("./")
        if self.env is not None:
            with self.env.begin(write=False, buffers=True) as txn:
                img_buf = txn.get(normalized.encode("ascii"))
                if img_buf is None:
                    img_buf = txn.get(os.path.basename(normalized).encode("ascii"))
            if img_buf is None:
                raise ValueError(f"Image not found in lmdb: {normalized}")
            return Image.open(BytesIO(img_buf)).convert("RGB")

        candidates = [
            os.path.join(self.image_root, normalized),
            os.path.join(self.root, normalized),
            os.path.join(self.root, "LaSCo", normalized),
            os.path.join(self.root, "LaSCo", "images", normalized),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return Image.open(candidate).convert("RGB")
        raise FileNotFoundError(f"Image not found for LaSCo path: {image_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.lmdb_path and self.env is None:
            self._init_db()

        item = self.data[idx]
        ref_path = self._image_path(item.get("query-image") or item.get("query_image") or item.get("image"))
        target_path = self._image_path(item.get("target-image") or item.get("target_image") or item.get("target_img"))
        text = item.get("query-text") or item.get("query_text") or item.get("text") or item.get("modification") or ""

        ref_img = self._read_image(ref_path)
        target_img = self._read_image(target_path)

        if self.transform:
            ref_img = self.transform(ref_img)
            target_img = self.transform(target_img)

        return {
            "id": item.get("qid", item.get("id", idx)),
            "image": ref_img,
            "target_path": target_path,
            "target_img": target_img,
            "text": str(text),
            "np": item.get("nps") or self._text_spans(text),
        }

    def custom_collate_fn(self, batch):
        nps = [item.pop('np') for item in batch]
        collated_batch = default_collate(batch)
        collated_batch['np'] = nps
        return collated_batch
