import argparse
import json
import os

import chromadb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.full_model import ScheiCIR
import lmdb


LMDB_ENV_CACHE = {}


DATASET_PATHS = {
    "CIRR": {
        "data_path": "./data/CIRR",
        "json_path": "cap.rc2.test1.json",
        "val_json_path": "cap.rc2.val.json",
        "split_json_path": "split.rc2.test1.json",
        "lmdb_path": "./data/CIRR/images_224_lmdb",
        "version": "rc2",
    },
    "FashionIQ": {
        "data_path": "./data/FashionIQ",
        "caption_dir": "captions",
        "split_dir": "image_splits",
        "image_root": ".",
        "lmdb_path": "./data/FashionIQ/images_val_224_lmdb",
    },
    "MerdCIR": {
        "data_path": "./data",
        "json_path": "merdcir_np/eval/eval_subset.jsonl",
        "lmdb_path": "./data/MTCIR/images_224_lmdb",
    },
    "MTCIR": {
        "data_path": "./data",
        "json_path": "mtcir_np/eval/eval_subset.jsonl",
        "lmdb_path": "./data/MTCIR/images_224_lmdb",
    },
}


class EvalJsonDataset(Dataset):
    def __init__(self, data_path, json_path, lmdb_path, require_target=True, split_json_path=None):
        self.root = data_path
        self.json_path = os.path.join(data_path, json_path) if not os.path.isabs(json_path) else json_path
        self.split_json_path = (
            os.path.join(data_path, split_json_path)
            if split_json_path and not os.path.isabs(split_json_path)
            else split_json_path
        )
        self.lmdb_path = lmdb_path
        self.require_target = require_target
        self.data = self._load_data(self.json_path)
        self.has_targets = any(
            self._first_valid(
                item,
                ["target_img", "target_image", "target_hard", "target_soft", "target", "candidate"],
            )
            is not None
            for item in self.data
        )
        self.env = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _load_data(path):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            first_chars = f.read(256)
            f.seek(0)
            if "<!DOCTYPE html" in first_chars or "<html" in first_chars:
                raise ValueError(
                    f"{path} looks like an HTML page, not JSON. Download the raw CIRR JSON file instead."
                )
            if path.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            else:
                payload = json.load(f)
                if isinstance(payload, list):
                    records = payload
                else:
                    records = payload.get("data", [])
        return records

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    @staticmethod
    def _first_valid(item, keys):
        for key in keys:
            if key in item and item[key]:
                return item[key]
        return None

    @staticmethod
    def _to_text(value):
        if value is None:
            return ""
        if isinstance(value, list):
            return ". ".join([str(v) for v in value if str(v).strip()])
        return str(value)

    @staticmethod
    def _to_candidates(value):
        if value is None:
            return []
        if isinstance(value, dict):
            for key in ("members", "images", "image_ids", "ids"):
                if key in value:
                    return EvalJsonDataset._to_candidates(value[key])
            return []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        return [str(value)]

    @staticmethod
    def _candidate_image_keys(image_id):
        image_id = str(image_id).strip()
        if not image_id:
            return []

        normalized = image_id.replace("\\", "/").lstrip("./")
        base, ext = os.path.splitext(normalized)
        candidates = [normalized]
        if not ext:
            candidates.extend([f"{normalized}.png", f"{normalized}.jpg", f"{normalized}.jpeg"])

        first = normalized.split("/", 1)[0]
        if "/" not in normalized:
            for split in ("test1", "dev", "train"):
                if normalized.startswith(f"{split}-"):
                    candidates.append(f"{split}/{normalized}")
                    if not ext:
                        candidates.extend([
                            f"{split}/{normalized}.png",
                            f"{split}/{normalized}.jpg",
                            f"{split}/{normalized}.jpeg",
                        ])
                    break
        elif first in {"test1", "dev", "train"} and not ext:
            candidates.extend([f"{normalized}.png", f"{normalized}.jpg", f"{normalized}.jpeg"])

        return list(dict.fromkeys(candidates))

    @classmethod
    def _get_image_buffer(cls, txn, image_id):
        for key in cls._candidate_image_keys(image_id):
            image_buf = txn.get(key.encode("ascii"))
            if image_buf is not None:
                return image_buf
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        item = self.data[idx]
        image_key = self._first_valid(item, ["image", "reference", "reference_img", "reference_image", "ref_img"])
        if image_key is None:
            raise KeyError(f"Missing reference image key in sample idx={idx}.")

        with self.env.begin(write=False, buffers=True) as txn:
            image_buf = self._get_image_buffer(txn, image_key)
        if image_buf is None:
            raise KeyError(f"LMDB image missing: ref={image_key}")

        image = self.transform(Image.open(BytesIO(image_buf)).convert("RGB"))
        text = self._to_text(self._first_valid(item, ["merdcir_modification", "modification", "modifications", "caption"]))
        pair_id = item.get("pairid", item.get("pair_id", item.get("id", idx)))
        candidates = self._to_candidates(
            self._first_valid(item, ["img_set", "candidate_set", "candidates", "gallery", "gallery_ids"])
        )

        sample = {
            "id": item.get("id", idx),
            "pair_id": str(pair_id),
            "image": image,
            "reference_path": str(image_key),
            "text": text,
            "candidates": candidates,
        }
        target_key = self._first_valid(
            item,
            ["target_img", "target_image", "target_hard", "target_soft", "target", "candidate"],
        )
        if target_key is not None:
            sample["target_path"] = str(target_key)
        if self.require_target:
            if target_key is None:
                raise KeyError(f"Missing target key in sample idx={idx}.")
            with self.env.begin(write=False, buffers=True) as txn:
                target_buf = self._get_image_buffer(txn, target_key)
            if target_buf is None:
                raise KeyError(f"LMDB image missing: target={target_key}")
            sample["target_img"] = self.transform(Image.open(BytesIO(target_buf)).convert("RGB"))
        return sample

    @staticmethod
    def collate_fn(batch):
        ids = [x["id"] for x in batch]
        pair_ids = [x["pair_id"] for x in batch]
        images = torch.stack([x["image"] for x in batch], dim=0)
        texts = [x["text"] for x in batch]
        payload = {
            "id": ids,
            "pair_id": pair_ids,
            "image": images,
            "reference_path": [x["reference_path"] for x in batch],
            "text": texts,
            "candidates": [x["candidates"] for x in batch],
        }
        if "target_path" in batch[0]:
            payload["target_path"] = [x["target_path"] for x in batch]
        if "target_img" in batch[0]:
            payload["target_img"] = torch.stack([x["target_img"] for x in batch], dim=0)
        return payload


class FashionIQDataset(Dataset):
    def __init__(
        self,
        data_path,
        category,
        split="val",
        caption_dir="captions",
        split_dir="image_splits",
        image_root=".",
        lmdb_path=None,
        require_target=True,
    ):
        self.root = data_path
        self.category = category
        self.split = split
        self.caption_path = os.path.join(data_path, caption_dir, f"cap.{category}.{split}.json")
        self.split_path = os.path.join(data_path, split_dir, f"split.{category}.{split}.json")
        self.image_root = image_root if os.path.isabs(image_root) else os.path.join(data_path, image_root)
        self.lmdb_path = lmdb_path
        self.require_target = require_target
        self.env = None
        self.data = self._load_json(self.caption_path)
        for idx, item in enumerate(self.data):
            item.setdefault("_fashioniq_index", idx)
        self.gallery_ids = [str(x) for x in self._load_json(self.split_path)]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.lmdb_path:
            before = len(self.data)
            self.data = [
                item for item in self.data
                if self.has_image(item.get("candidate"))
                and (not self.require_target or self.has_image(item.get("target")))
            ]
            skipped = before - len(self.data)
            if skipped:
                print(f"Skipped {skipped} FashionIQ/{category}/{split} queries with missing LMDB query/target images.")

    @staticmethod
    def _load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            first_chars = f.read(256)
            f.seek(0)
            if "<!DOCTYPE html" in first_chars or "<html" in first_chars:
                raise ValueError(f"{path} looks like an HTML page, not JSON.")
            return json.load(f)

    def _init_db(self):
        if self.lmdb_path:
            lmdb_path = os.path.abspath(self.lmdb_path)
            if lmdb_path not in LMDB_ENV_CACHE:
                LMDB_ENV_CACHE[lmdb_path] = lmdb.open(
                    lmdb_path,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
            self.env = LMDB_ENV_CACHE[lmdb_path]

    @staticmethod
    def _to_text(captions):
        if captions is None:
            return ""
        if isinstance(captions, list):
            return ". ".join(str(caption) for caption in captions if str(caption).strip())
        return str(captions)

    @staticmethod
    def _normalize_image_id(image_id):
        return str(image_id).replace("\\", "/").lstrip("./")

    def _candidate_image_paths(self, image_id):
        normalized = self._normalize_image_id(image_id)
        base, ext = os.path.splitext(normalized)
        names = [normalized] if ext else [
            normalized,
            f"{normalized}.jpg",
            f"{normalized}.jpeg",
            f"{normalized}.png",
        ]
        roots = [
            self.image_root,
            os.path.join(self.image_root, self.category),
            os.path.join(self.image_root, "images"),
            os.path.join(self.image_root, "images", self.category),
            self.root,
            os.path.join(self.root, self.category),
            os.path.join(self.root, "images"),
            os.path.join(self.root, "images", self.category),
        ]
        candidates = []
        for root in roots:
            for name in names:
                candidates.append(os.path.join(root, name))
        return list(dict.fromkeys(candidates))

    def load_image(self, image_id):
        normalized = self._normalize_image_id(image_id)
        if self.lmdb_path and self.env is None:
            self._init_db()
        if self.env is not None:
            with self.env.begin(write=False, buffers=True) as txn:
                img_buf = txn.get(normalized.encode("ascii"))
            if img_buf is not None:
                return self.transform(Image.open(BytesIO(img_buf)).convert("RGB"))

        for path in self._candidate_image_paths(image_id):
            if os.path.isfile(path):
                return self.transform(Image.open(path).convert("RGB"))
        raise FileNotFoundError(f"FashionIQ image not found: {image_id}")

    def has_image(self, image_id):
        if image_id is None:
            return False
        normalized = self._normalize_image_id(image_id)
        if self.lmdb_path and self.env is None:
            self._init_db()
        if self.env is not None:
            with self.env.begin(write=False, buffers=True) as txn:
                return txn.get(normalized.encode("ascii")) is not None
        return any(os.path.isfile(path) for path in self._candidate_image_paths(normalized))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ref_id = item.get("candidate") or item.get("reference") or item.get("image")
        target_id = item.get("target")
        if ref_id is None:
            raise KeyError(f"Missing FashionIQ reference image in sample idx={idx}")
        if target_id is None and self.require_target:
            raise KeyError(f"Missing FashionIQ target image in sample idx={idx}; use a split with labels, usually val.")
        sample = {
            "id": item.get("_fashioniq_index", idx),
            "image": self.load_image(ref_id),
            "reference_path": self._normalize_image_id(ref_id),
            "text": self._to_text(item.get("captions") or item.get("caption") or item.get("modification")),
        }
        if target_id is not None:
            sample["target_path"] = self._normalize_image_id(target_id)
        return sample

    @staticmethod
    def collate_fn(batch):
        payload = {
            "id": [x["id"] for x in batch],
            "image": torch.stack([x["image"] for x in batch], dim=0),
            "reference_path": [x["reference_path"] for x in batch],
            "text": [x["text"] for x in batch],
        }
        if "target_path" in batch[0]:
            payload["target_path"] = [x["target_path"] for x in batch]
        return payload


@torch.no_grad()
def evaluate(model, loader, collection_name, device, k_list=(1, 5, 10, 50)):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)

    gallery_ids = set()
    for batch in tqdm(loader, desc="Building gallery"):
        target_imgs = batch["target_img"].to(device)
        target_paths = batch["target_path"]
        target_feats = model.visual_backbone(target_imgs, get_embeddings=True)
        target_feats = F.normalize(target_feats, dim=-1).cpu().numpy().tolist()

        insert_embeddings, insert_ids = [], []
        for feat, target_path in zip(target_feats, target_paths):
            if target_path in gallery_ids:
                continue
            insert_embeddings.append(feat)
            insert_ids.append(target_path)
            gallery_ids.add(target_path)
        if insert_ids:
            collection.add(embeddings=insert_embeddings, ids=insert_ids)

    recalls = {k: [] for k in k_list}
    aps = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        texts = batch["text"]
        target_ids = batch["target_path"]

        query_feats = model(images, texts, return_attention=False)
        query_feats = F.normalize(query_feats, dim=-1)
        results = collection.query(query_embeddings=query_feats.cpu().numpy().tolist(), n_results=max(k_list))
        retrieved_ids = results["ids"]

        for i, target_id in enumerate(target_ids):
            preds = retrieved_ids[i]
            for k in k_list:
                recalls[k].append(1 if target_id in preds[:k] else 0)
            if target_id in preds:
                aps.append(1.0 / (preds.index(target_id) + 1))
            else:
                aps.append(0.0)

    client.delete_collection(name=collection_name)
    summary = {f"Recall@{k}": float(np.mean(recalls[k])) for k in k_list}
    summary["mAP"] = float(np.mean(aps))
    return summary


@torch.no_grad()
def evaluate_fashioniq(model, loader, device, k_list=(1, 5, 10, 50), return_predictions=False):
    dataset = loader.dataset
    id_to_index, gallery_ids, gallery_chunks = _encode_fashioniq_gallery(model, dataset, device)
    if not gallery_chunks:
        raise RuntimeError(f"No FashionIQ gallery images could be loaded for category={dataset.category}.")
    gallery_embeddings = torch.cat(gallery_chunks, dim=0)

    has_targets = False
    recalls = {k: [] for k in k_list}
    aps = []
    predictions = {}

    for batch in tqdm(loader, desc=f"Evaluating FashionIQ/{dataset.category}"):
        images = batch["image"].to(device)
        texts = batch["text"]
        references = batch["reference_path"]
        targets = batch.get("target_path")

        query_feats = model(images, texts, return_attention=False)
        query_feats = F.normalize(query_feats, dim=-1).cpu()

        for i in range(len(references)):
            scores = torch.matmul(gallery_embeddings, query_feats[i])
            reference_id = references[i]
            if reference_id in id_to_index:
                scores[id_to_index[reference_id]] = -float("inf")
            n_results = min(max(k_list), len(gallery_ids))
            top_indices = torch.topk(scores, k=n_results).indices.tolist()
            ranked_ids = [gallery_ids[idx] for idx in top_indices]
            predictions[f"{dataset.category}:{batch['id'][i]}"] = ranked_ids

            if targets is None:
                continue
            target_id = targets[i]
            has_targets = True
            if target_id in ranked_ids:
                rank = ranked_ids.index(target_id) + 1
                aps.append(1.0 / rank)
            else:
                rank = None
                aps.append(0.0)
            for k in k_list:
                recalls[k].append(1.0 if rank is not None and rank <= k else 0.0)

    summary = {}
    if has_targets:
        summary = {f"Recall@{k}": float(np.mean(recalls[k])) for k in k_list}
        summary["mAP"] = float(np.mean(aps))
    if return_predictions:
        return summary, predictions
    return summary


def _encode_fashioniq_gallery(model, dataset, device, batch_size=256):
    id_to_index = {}
    gallery_ids = []
    encoded_chunks = []
    unique_ids = list(dict.fromkeys(dataset._normalize_image_id(image_id) for image_id in dataset.gallery_ids))
    for start in tqdm(range(0, len(unique_ids), batch_size), desc=f"Encoding FashionIQ/{dataset.category} gallery"):
        current_ids = unique_ids[start:start + batch_size]
        images = []
        valid_ids = []
        for image_id in current_ids:
            try:
                images.append(dataset.load_image(image_id))
            except FileNotFoundError:
                continue
            valid_ids.append(image_id)
        if not valid_ids:
            continue
        batch_images = torch.stack(images, dim=0).to(device)
        feats = model.visual_backbone(batch_images, get_embeddings=True)
        feats = F.normalize(feats, dim=-1).cpu()
        for image_id in valid_ids:
            id_to_index[image_id] = len(gallery_ids)
            gallery_ids.append(image_id)
        encoded_chunks.append(feats)
    return id_to_index, gallery_ids, encoded_chunks


def average_metric_dicts(metric_dicts):
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}


@torch.no_grad()
def evaluate_cirr_and_dump(model, loader, collection_name, device, dataset_version="rc2", metric="recall"):
    if metric not in {"recall", "recall_subset"}:
        raise ValueError("metric must be 'recall' or 'recall_subset'.")
    topk = 50 if metric == "recall" else 3
    use_workers = getattr(loader, "num_workers", 0) > 0

    gallery_ids = _load_cirr_gallery_ids(getattr(loader.dataset, "split_json_path", None))
    if not gallery_ids:
        gallery_ids = _collect_cirr_gallery_ids(loader)

    close_env = False
    env = None
    try:
        if use_workers:
            env = lmdb.open(loader.dataset.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            close_env = True
        else:
            if loader.dataset.env is None:
                loader.dataset._init_db()
            env = loader.dataset.env

        id_to_index, encoded_ids, encoded_chunks = _encode_cirr_gallery(
            model,
            gallery_ids,
            env,
            loader.dataset,
            device,
        )
    finally:
        if close_env and env is not None:
            env.close()

    if not encoded_chunks:
        raise RuntimeError("No CIRR gallery images could be loaded from LMDB.")
    gallery_embeddings = torch.cat(encoded_chunks, dim=0)

    outputs = {"version": dataset_version, "metric": metric}
    for batch in tqdm(loader, desc="Generating CIRR predictions"):
        images = batch["image"].to(device)
        texts = batch["text"]
        pair_ids = batch["pair_id"]
        references = batch["reference_path"]
        candidates = batch["candidates"]

        query_feats = model(images, texts, return_attention=False)
        query_feats = F.normalize(query_feats, dim=-1).cpu()

        for i, pair_id in enumerate(pair_ids):
            if metric == "recall_subset":
                preds = _rank_cirr_subset(
                    query_feats[i],
                    candidates[i],
                    references[i],
                    gallery_embeddings,
                    encoded_ids,
                    id_to_index,
                    topk,
                )
            else:
                preds = _rank_cirr_global(
                    query_feats[i],
                    references[i],
                    gallery_embeddings,
                    encoded_ids,
                    id_to_index,
                    topk,
                )
            outputs[str(pair_id)] = preds[:topk]

    return outputs


@torch.no_grad()
def evaluate_cirr_subset(model, loader, device, k_list=(1, 2, 3)):
    use_workers = getattr(loader, "num_workers", 0) > 0
    gallery_ids = _load_cirr_gallery_ids(getattr(loader.dataset, "split_json_path", None))
    if not gallery_ids:
        gallery_ids = _collect_cirr_gallery_ids(loader)

    close_env = False
    env = None
    try:
        if use_workers:
            env = lmdb.open(loader.dataset.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            close_env = True
        else:
            if loader.dataset.env is None:
                loader.dataset._init_db()
            env = loader.dataset.env

        id_to_index, encoded_ids, encoded_chunks = _encode_cirr_gallery(
            model,
            gallery_ids,
            env,
            loader.dataset,
            device,
        )
    finally:
        if close_env and env is not None:
            env.close()

    if not encoded_chunks:
        raise RuntimeError("No CIRR gallery images could be loaded from LMDB.")
    gallery_embeddings = torch.cat(encoded_chunks, dim=0)

    recalls = {k: [] for k in k_list}
    for batch in tqdm(loader, desc="Evaluating CIRR subset"):
        images = batch["image"].to(device)
        texts = batch["text"]
        references = batch["reference_path"]
        candidates = batch["candidates"]
        targets = batch.get("target_path")
        if targets is None:
            raise ValueError("CIRR subset evaluation requires target labels. Use cap.rc2.val.json.")

        query_feats = model(images, texts, return_attention=False)
        query_feats = F.normalize(query_feats, dim=-1).cpu()

        for i, target_id in enumerate(targets):
            preds = _rank_cirr_subset(
                query_feats[i],
                candidates[i],
                references[i],
                gallery_embeddings,
                encoded_ids,
                id_to_index,
                max(k_list),
            )
            for k in k_list:
                recalls[k].append(1.0 if target_id in preds[:k] else 0.0)

    return {f"Recall_subset@{k}": float(np.mean(recalls[k])) for k in k_list}


@torch.no_grad()
def evaluate_cirr_recall(model, loader, device, k_list=(1, 5, 10, 50)):
    use_workers = getattr(loader, "num_workers", 0) > 0
    gallery_ids = _load_cirr_gallery_ids(getattr(loader.dataset, "split_json_path", None))
    if not gallery_ids:
        gallery_ids = _collect_cirr_gallery_ids(loader)

    close_env = False
    env = None
    try:
        if use_workers:
            env = lmdb.open(loader.dataset.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            close_env = True
        else:
            if loader.dataset.env is None:
                loader.dataset._init_db()
            env = loader.dataset.env

        id_to_index, encoded_ids, encoded_chunks = _encode_cirr_gallery(
            model,
            gallery_ids,
            env,
            loader.dataset,
            device,
        )
    finally:
        if close_env and env is not None:
            env.close()

    if not encoded_chunks:
        raise RuntimeError("No CIRR gallery images could be loaded from LMDB.")
    gallery_embeddings = torch.cat(encoded_chunks, dim=0)

    recalls = {k: [] for k in k_list}
    for batch in tqdm(loader, desc="Evaluating CIRR recall"):
        images = batch["image"].to(device)
        texts = batch["text"]
        references = batch["reference_path"]
        targets = batch.get("target_path")
        if targets is None:
            raise ValueError("CIRR recall evaluation requires target labels. Use cap.rc2.val.json.")

        query_feats = model(images, texts, return_attention=False)
        query_feats = F.normalize(query_feats, dim=-1).cpu()

        for i, target_id in enumerate(targets):
            preds = _rank_cirr_global(
                query_feats[i],
                references[i],
                gallery_embeddings,
                encoded_ids,
                id_to_index,
                max(k_list),
            )
            for k in k_list:
                recalls[k].append(1.0 if target_id in preds[:k] else 0.0)

    return {f"Recall@{k}": float(np.mean(recalls[k])) for k in k_list}


def _encode_cirr_gallery(model, gallery_ids, env, dataset, device):
    id_to_index = {}
    encoded_ids = []
    encoded_chunks = []
    for start in tqdm(range(0, len(gallery_ids), 512), desc="Encoding CIRR gallery"):
        current_ids = gallery_ids[start:start + 512]
        images = []
        valid_ids = []
        with env.begin(write=False, buffers=True) as txn:
            for img_id in current_ids:
                img_buf = dataset._get_image_buffer(txn, img_id)
                if img_buf is None:
                    continue
                images.append(dataset.transform(Image.open(BytesIO(img_buf)).convert("RGB")))
                valid_ids.append(img_id)
        if not valid_ids:
            continue
        batch_images = torch.stack(images, dim=0).to(device)
        feats = model.visual_backbone(batch_images, get_embeddings=True)
        feats = F.normalize(feats, dim=-1).cpu()
        for img_id in valid_ids:
            id_to_index[img_id] = len(encoded_ids)
            encoded_ids.append(img_id)
        encoded_chunks.append(feats)
    return id_to_index, encoded_ids, encoded_chunks


def _load_cirr_gallery_ids(split_json_path):
    if not split_json_path:
        return []
    try:
        with open(split_json_path, "r", encoding="utf-8") as f:
            first_chars = f.read(256)
            f.seek(0)
            if "<!DOCTYPE html" in first_chars or "<html" in first_chars:
                return []
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    if isinstance(payload, dict):
        for key in ("data", "images", "image_ids", "files", "test1", "val", "train", "dev"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            values = []
            for value in payload.values():
                if isinstance(value, list):
                    values.extend(value)
            payload = values

    ids = []
    for item in payload:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            image_id = EvalJsonDataset._first_valid(
                item, ["image", "img", "image_id", "path", "file_name", "filename"]
            )
            if image_id:
                ids.append(str(image_id))
    return list(dict.fromkeys(ids))


def _collect_cirr_gallery_ids(loader):
    gallery_ids = []
    seen = set()
    for batch in tqdm(loader, desc="Building CIRR gallery"):
        for ref, cands in zip(batch["reference_path"], batch["candidates"]):
            for image_id in [ref, *cands]:
                if image_id and image_id not in seen:
                    gallery_ids.append(str(image_id))
                    seen.add(image_id)
    return gallery_ids


def _rank_cirr_global(query_feat, reference_id, gallery_embeddings, gallery_ids, id_to_index, topk):
    scores = torch.matmul(gallery_embeddings, query_feat)
    if reference_id in id_to_index:
        scores[id_to_index[reference_id]] = -float("inf")
    n_results = min(topk, len(gallery_ids))
    top_indices = torch.topk(scores, k=n_results).indices.tolist()
    return [gallery_ids[idx] for idx in top_indices if gallery_ids[idx] != reference_id]


def _rank_cirr_subset(query_feat, candidates, reference_id, gallery_embeddings, gallery_ids, id_to_index, topk):
    candidate_ids = [img_id for img_id in candidates if img_id != reference_id and img_id in id_to_index]
    if not candidate_ids:
        return _rank_cirr_global(query_feat, reference_id, gallery_embeddings, gallery_ids, id_to_index, topk)

    candidate_indices = torch.tensor([id_to_index[img_id] for img_id in candidate_ids], dtype=torch.long)
    candidate_embs = gallery_embeddings.index_select(0, candidate_indices)
    scores = torch.matmul(candidate_embs, query_feat)
    n_results = min(topk, len(candidate_ids))
    top_indices = torch.topk(scores, k=n_results).indices.tolist()
    return [candidate_ids[idx] for idx in top_indices]


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on CIR datasets.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["CIRR", "FashionIQ", "MerdCIR", "MTCIR"])
    parser.add_argument("--method", type=str, default="cross_attn_alpha")
    parser.add_argument("--cirr-metric", type=str, default="recall", choices=["recall", "recall_subset", "subset_eval"])
    parser.add_argument("--cirr-json-path", type=str, default=None)
    parser.add_argument("--cirr-split-json-path", type=str, default=None)
    parser.add_argument("--cirr-lmdb-path", type=str, default=None)
    parser.add_argument("--fashioniq-category", type=str, default="all", choices=["dress", "shirt", "toptee", "all"])
    parser.add_argument("--fashioniq-split", type=str, default="val")
    parser.add_argument("--fashioniq-image-root", type=str, default=None)
    parser.add_argument("--fashioniq-lmdb-path", type=str, default=None)
    parser.add_argument("--fashioniq-output-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    dataset_cfg = DATASET_PATHS[args.dataset]
    batch_size = 128
    num_workers = 4
    temperature = 0.07
    cirr_metric = args.cirr_metric
    cirr_output_json = f"cirr_eval_{dataset_cfg.get('version', 'rc2')}_{cirr_metric}.json"
    if args.output_json:
        cirr_output_json = args.output_json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScheiCIR(method=args.method, temperature=temperature).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if args.dataset == "FashionIQ":
        categories = ["dress", "shirt", "toptee"] if args.fashioniq_category == "all" else [args.fashioniq_category]
        all_metrics = []
        all_predictions = {}
        for category in categories:
            dataset = FashionIQDataset(
                data_path=dataset_cfg["data_path"],
                category=category,
                split=args.fashioniq_split,
                caption_dir=dataset_cfg.get("caption_dir", "captions"),
                split_dir=dataset_cfg.get("split_dir", "image_splits"),
                image_root=args.fashioniq_image_root or dataset_cfg.get("image_root", "."),
                lmdb_path=args.fashioniq_lmdb_path or dataset_cfg.get("lmdb_path"),
                require_target=(args.fashioniq_split != "test"),
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=FashionIQDataset.collate_fn,
            )
            if args.fashioniq_output_json:
                metrics, predictions = evaluate_fashioniq(
                    model,
                    loader,
                    device,
                    return_predictions=True,
                )
            else:
                metrics = evaluate_fashioniq(model, loader, device)
                predictions = {}
            all_metrics.append(metrics)
            all_predictions[category] = predictions
            print(f"Dataset: FashionIQ/{category}")
            if metrics:
                for key, value in metrics.items():
                    print(f"{key}: {value:.6f}")
            else:
                print("No target labels found; generated rankings only.")

        metric_sets = [metrics for metrics in all_metrics if metrics]
        if len(metric_sets) > 1:
            averaged = average_metric_dicts(metric_sets)
            print("Dataset: FashionIQ/average")
            for key, value in averaged.items():
                print(f"{key}: {value:.6f}")

        if args.fashioniq_output_json:
            with open(args.fashioniq_output_json, "w", encoding="utf-8") as f:
                json.dump(all_predictions, f, ensure_ascii=False)
            print(f"FashionIQ prediction file saved to: {args.fashioniq_output_json}")
        return

    json_path = dataset_cfg["json_path"]
    split_json_path = dataset_cfg.get("split_json_path")
    lmdb_path = dataset_cfg["lmdb_path"]
    if args.dataset == "CIRR":
        lmdb_path = args.cirr_lmdb_path or lmdb_path
        if cirr_metric == "subset_eval":
            json_path = args.cirr_json_path or dataset_cfg.get("val_json_path", json_path)
            split_json_path = args.cirr_split_json_path
        else:
            json_path = args.cirr_json_path or json_path
            split_json_path = args.cirr_split_json_path or split_json_path
            if args.cirr_json_path and "val" in os.path.basename(args.cirr_json_path) and not args.cirr_split_json_path:
                split_json_path = None

    dataset = EvalJsonDataset(
        data_path=dataset_cfg["data_path"],
        json_path=json_path,
        lmdb_path=lmdb_path,
        require_target=(args.dataset != "CIRR" or cirr_metric == "subset_eval"),
        split_json_path=split_json_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=EvalJsonDataset.collate_fn,
    )
    collection_name = f"eval_{args.dataset.lower()}_{os.getpid()}"
    if args.dataset == "CIRR":
        if cirr_metric == "subset_eval" or (cirr_metric == "recall_subset" and dataset.has_targets):
            result = evaluate_cirr_subset(model, loader, device)
            print("Dataset: CIRR/subset")
            for key, value in result.items():
                print(f"{key}: {value:.6f}")
        elif cirr_metric == "recall" and dataset.has_targets:
            result = evaluate_cirr_recall(model, loader, device)
            print("Dataset: CIRR/recall")
            for key, value in result.items():
                print(f"{key}: {value:.6f}")
        else:
            result = evaluate_cirr_and_dump(
                model,
                loader,
                collection_name,
                device,
                dataset_version=dataset_cfg.get("version", "rc2"),
                metric=cirr_metric,
            )
            with open(cirr_output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            print(f"CIRR prediction file saved to: {cirr_output_json}")
    else:
        result = evaluate(model, loader, collection_name, device)
        print(f"Dataset: {args.dataset}")
        for key, value in result.items():
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
