import random
from eval import ScheiEvaluator
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.full_model import ScheiCIR
from data.dataset import MTCIRDataset, MerdCIRDataset, LaSCoDataset
import argparse
import torch.nn.functional as F
import os
import datetime
import json
import shutil
from eval_checkpoint import EvalJsonDataset

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'./logs/train{datetime.datetime.today()}.log', encoding='utf-8', level=logging.INFO)


def save_checkpoint(state, is_best, filename='./checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/model_best.pth.tar')


def standardize_metrics(metrics):
    return {
        "R@1": float(metrics.get("R@1", metrics.get("Recall@1", 0.0))),
        "R@5": float(metrics.get("R@5", metrics.get("Recall@5", 0.0))),
        "R@10": float(metrics.get("R@10", metrics.get("Recall@10", 0.0))),
        "R@50": float(metrics.get("R@50", metrics.get("Recall@50", 0.0))),
        "mAP": float(metrics.get("mAP", 0.0)),
    }


def format_metrics(metrics):
    metrics = standardize_metrics(metrics)
    return (
        f"R@1={metrics['R@1']:.6f}, R@5={metrics['R@5']:.6f}, "
        f"R@10={metrics['R@10']:.6f}, R@50={metrics['R@50']:.6f}, mAP={metrics['mAP']:.6f}"
    )


@torch.no_grad()
def evaluate_full_ranking(model, loader, device, k_list=(1, 5, 10, 50), desc="Validation"):
    model.eval()
    gallery_feats = []
    gallery_ids = []
    seen_ids = set()

    try:
        for batch in tqdm(loader, desc=f"{desc}: encoding gallery"):
            target_imgs = batch["target_img"].to(device)
            target_paths = batch["target_path"]
            feats = model.visual_backbone(target_imgs, get_embeddings=True)
            feats = F.normalize(feats, p=2, dim=-1).cpu()

            keep_indices = []
            for i, target_path in enumerate(target_paths):
                if target_path in seen_ids:
                    continue
                seen_ids.add(target_path)
                gallery_ids.append(target_path)
                keep_indices.append(i)
            if keep_indices:
                gallery_feats.append(feats[keep_indices])

        if not gallery_feats:
            raise RuntimeError(f"{desc} gallery is empty; cannot evaluate.")

        gallery_feats = torch.cat(gallery_feats, dim=0)
        recalls = {k: [] for k in k_list}
        aps = []

        for batch in tqdm(loader, desc=f"{desc}: querying"):
            q_imgs = batch["image"].to(device)
            texts = batch["text"]
            target_ids = batch["target_path"]

            query_feats = model(q_imgs, texts, return_attention=False)
            query_feats = F.normalize(query_feats, p=2, dim=-1).cpu()
            scores = torch.matmul(query_feats, gallery_feats.t())
            ranked_indices = torch.argsort(scores, dim=1, descending=True).tolist()

            for row, target_id in zip(ranked_indices, target_ids):
                ranked_ids = [gallery_ids[idx] for idx in row]
                if target_id in ranked_ids:
                    rank = ranked_ids.index(target_id) + 1
                    aps.append(1.0 / rank)
                else:
                    rank = None
                    aps.append(0.0)
                for k in k_list:
                    recalls[k].append(1.0 if rank is not None and rank <= k else 0.0)

        return {
            "R@1": float(np.mean(recalls[1])),
            "R@5": float(np.mean(recalls[5])),
            "R@10": float(np.mean(recalls[10])),
            "R@50": float(np.mean(recalls[50])),
            "mAP": float(np.mean(aps)),
        }
    finally:
        close_dataset_env(loader.dataset)


def close_dataset_env(dataset):
    env = getattr(dataset, "env", None)
    if env is not None:
        env.close()
        dataset.env = None


def evaluate_in_domain(model, loader, device):
    return standardize_metrics(evaluate_full_ranking(model, loader, device, desc="In-domain validation"))


def evaluate_cirr(model, cirr_loader, device):
    return standardize_metrics(evaluate_full_ranking(model, cirr_loader, device, desc="CIRR validation"))


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_topk_checkpoint(topk_checkpoints, state, epoch, score, metrics, checkpoint_dir, k=3, step=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    step_part = f"_step_{step:06d}" if step is not None else ""
    candidate_path = os.path.join(checkpoint_dir, f"topk_epoch_{epoch:04d}{step_part}_score_{score:.6f}.pth.tar")

    entry = {
        "score": float(score),
        "epoch": int(epoch),
        "step": int(step) if step is not None else None,
        "path": candidate_path,
        "metrics": metrics,
    }

    if len(topk_checkpoints) < k:
        torch.save(state, candidate_path)
        topk_checkpoints.append(entry)
    else:
        worst = min(topk_checkpoints, key=lambda x: x["score"])
        if score <= worst["score"]:
            return topk_checkpoints
        if os.path.exists(worst["path"]):
            os.remove(worst["path"])
        topk_checkpoints.remove(worst)
        torch.save(state, candidate_path)
        topk_checkpoints.append(entry)

    topk_checkpoints.sort(key=lambda x: x["score"], reverse=True)
    return topk_checkpoints


@torch.no_grad()
def evaluate_cir(scheis_model, loader, k_list=(1, 5, 10, 50)):
    scheis_model.eval()
    try:
        evaluator.client.delete_collection(name="cir_eval_temp")
    except:
        pass
    # Reproducibility note: keep this step explicit for repeatable experiments.
    # Reproducibility note: keep this step explicit for repeatable experiments.
    collection = evaluator.client.get_or_create_collection(name="cir_eval_temp")

    logger.info("Encoding gallery images...")
    # Reproducibility note: keep this step explicit for repeatable experiments.
    # Reproducibility note: keep this step explicit for repeatable experiments.
    global added_ids
    for batch in tqdm(gallery_loader, desc="Encoding Gallery"):
        imgs = batch['image'].to(device)
        paths = batch['target_path']  # Reproducibility note: keep this step explicit for repeatable experiments.

        # Reproducibility note: keep this step explicit for repeatable experiments.
        current_batch_embeddings = []
        current_batch_ids = []

        for feat, path in zip(imgs, paths):
            if path not in added_ids:
                # Reproducibility note: keep this step explicit for repeatable experiments.
                img_feats = scheis_model.visual_backbone(feat.unsqueeze(0), get_embeddings=True).squeeze(0)
                img_feats = F.normalize(img_feats, p=2, dim=-1).cpu().numpy().tolist()
                current_batch_embeddings.append(img_feats)
                current_batch_ids.append(path)
                added_ids.add(path)

        if current_batch_ids:
            collection.add(embeddings=current_batch_embeddings, ids=current_batch_ids)

    logger.info("Running query retrieval...")
    # Reproducibility note: keep this step explicit for repeatable experiments.
    # Reproducibility note: keep this step explicit for repeatable experiments.
    all_recalls = {k: [] for k in k_list}
    all_aps = []

    for eval_batch in tqdm(loader):
        q_imgs = eval_batch['image']
        m_texts = eval_batch['text']
        target_ids = eval_batch['target_path']
        q_imgs = q_imgs.to(device)
        # Reproducibility note: keep this step explicit for repeatable experiments.
        query_feats = scheis_model(q_imgs, m_texts, return_attention=False)
        query_feats = query_feats / query_feats.norm(dim=-1, keepdim=True)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        results = collection.query(
            query_embeddings=query_feats.cpu().numpy().tolist(),
            n_results=max(k_list)
        )

        # Reproducibility note: keep this step explicit for repeatable experiments.
        retrieved_ids = results['ids']  # Reproducibility note: keep this step explicit for repeatable experiments.
        for i, target_id in enumerate(target_ids):
            preds = retrieved_ids[i]

            # Reproducibility note: keep this step explicit for repeatable experiments.
            for k in k_list:
                hit = 1 if target_id in preds[:k] else 0
                all_recalls[k].append(hit)

            # Reproducibility note: keep this step explicit for repeatable experiments.
            if target_id in preds:
                rank = preds.index(target_id) + 1
                all_aps.append(1.0 / rank)
            else:
                all_aps.append(0.0)

    # Reproducibility note: keep this step explicit for repeatable experiments.
    results_summary = {f"Recall@{k}": np.mean(all_recalls[k]) for k in k_list}
    results_summary["mAP"] = np.mean(all_aps)

    # Reproducibility note: keep this step explicit for repeatable experiments.
    evaluator.client.delete_collection("cir_eval_temp")
    added_ids = set()

    return results_summary


def get_last_encoder_layer(model):
    candidate_paths = [
        ("vision_model", "encoder", "layers"),
        ("text_model", "encoder", "layers"),
        ("encoder", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("layer",)
    ]
    for path in candidate_paths:
        cur = model
        ok = True
        for part in path:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__len__") and len(cur) > 0:
            return cur[-1]

    children = list(model.children())
    if children:
        return children[-1]
    return None


def get_optimizer(
    scheis_model,
    stage='warmup_head',
    lr_inter=1e-4,
    lr_backbone=1e-6,
    freeze_alpha_in_joint=True,
):
    # Reproducibility note: keep this step explicit for repeatable experiments.
    if stage == 'warmup_head':
        # Reproducibility note: keep this step explicit for repeatable experiments.
        params = [
            {'params': scheis_model.interaction.parameters(), 'lr': lr_inter},
            {'params': scheis_model.attn_pooler.parameters(), 'lr': lr_inter},
            {'params': scheis_model.alpha_gen.parameters(), 'lr': lr_inter} if scheis_model.alpha_gen else None
        ]
    elif stage == 'warmup_last_layer':
        visual_last_layer = get_last_encoder_layer(scheis_model.visual_backbone.model)
        text_last_layer = get_last_encoder_layer(scheis_model.text_encoder.model)
        params = [
            {'params': scheis_model.interaction.parameters(), 'lr': lr_inter},
            {'params': scheis_model.attn_pooler.parameters(), 'lr': lr_inter},
            {'params': scheis_model.alpha_gen.parameters(), 'lr': lr_inter} if scheis_model.alpha_gen else None,
            {'params': visual_last_layer.parameters(), 'lr': lr_backbone} if visual_last_layer is not None else None,
            {'params': text_last_layer.parameters(), 'lr': lr_backbone} if text_last_layer is not None else None
        ]
    else:
        # Reproducibility note: keep this step explicit for repeatable experiments.
        params = [
            {'params': scheis_model.interaction.parameters(), 'lr': lr_inter},
            {'params': scheis_model.attn_pooler.parameters(), 'lr': lr_inter},
            {'params': scheis_model.alpha_gen.parameters(), 'lr': lr_inter}
            if scheis_model.alpha_gen and not freeze_alpha_in_joint else None,
            {'params': scheis_model.visual_backbone.parameters(), 'lr': lr_backbone},
            {'params': scheis_model.text_encoder.parameters(), 'lr': lr_backbone}
        ]

    # Reproducibility note: keep this step explicit for repeatable experiments.
    params = [p for p in params if p is not None]
    return torch.optim.AdamW(params, weight_decay=0.01)


def resolve_training_stage(global_step, joint_start_step):
    if global_step < joint_start_step:
        return 'warmup_head'
    if global_step < (joint_start_step * 3):
        return 'warmup_last_layer'
    return 'joint'


def apply_training_stage(model, stage, freeze_alpha_in_joint=True):
    # Reproducibility note: keep this step explicit for repeatable experiments.
    model.visual_backbone.model.requires_grad_(False)
    model.text_encoder.model.requires_grad_(False)
    if model.alpha_gen:
        model.alpha_gen.requires_grad_(True)

    if stage == 'warmup_head':
        model.visual_backbone.model.eval()
        model.text_encoder.model.eval()
        model.interaction.train()
        model.attn_pooler.train()
        if model.alpha_gen:
            model.alpha_gen.train()
        return

    if stage == 'warmup_last_layer':
        visual_last_layer = get_last_encoder_layer(model.visual_backbone.model)
        text_last_layer = get_last_encoder_layer(model.text_encoder.model)

        if visual_last_layer is not None:
            visual_last_layer.requires_grad_(True)
            visual_last_layer.train()
        if text_last_layer is not None:
            text_last_layer.requires_grad_(True)
            text_last_layer.train()
        model.visual_backbone.model.eval()
        model.text_encoder.model.eval()
        model.interaction.train()
        model.attn_pooler.train()
        if model.alpha_gen:
            model.alpha_gen.train()
        return

    # Reproducibility note: keep this step explicit for repeatable experiments.
    model.visual_backbone.model.requires_grad_(True)
    model.text_encoder.model.requires_grad_(True)
    model.train()
    if model.alpha_gen and freeze_alpha_in_joint:
        model.alpha_gen.requires_grad_(False)
        model.alpha_gen.eval()
    # Reproducibility note: keep this step explicit for repeatable experiments.
    for m in model.visual_backbone.modules():
        if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
            m.eval()



def setup_initial_optimizer(model, lr_inter=1e-4):
    """
    Load all parameters at initialization, but set nonzero learning rates only for interaction-style modules by default,
    with the backbone learning rate initialized to 0.
    """
    # Reproducibility note: keep this step explicit for repeatable experiments.
    # Reproducibility note: keep this step explicit for repeatable experiments.
    head_params = []
    for m in [model.interaction, model.attn_pooler, model.alpha_gen]:
        if m is not None:
            head_params.extend(list(m.parameters()))

    # Reproducibility note: keep this step explicit for repeatable experiments.
    backbone_params = list(model.visual_backbone.parameters()) + \
                      list(model.text_encoder.parameters())

    param_groups = [
        {'params': head_params, 'lr': lr_inter, 'name': 'head'},
        {'params': backbone_params, 'lr': 0.0, 'name': 'backbone'}  # Reproducibility note: keep this step explicit for repeatable experiments.
    ]

    return torch.optim.AdamW(param_groups, weight_decay=0.01)


def update_optimizer_stage(optimizer, model, stage, lr_inter, lr_backbone, freeze_alpha_in_joint=True):
    # Reproducibility note: keep this step explicit for repeatable experiments.
    head_params_set = set()
    alpha_params_set = set(model.alpha_gen.parameters()) if model.alpha_gen is not None else set()
    for m in [model.interaction, model.attn_pooler, model.alpha_gen]:
        if m is not None:
            head_params_set.update(m.parameters())

    for group in optimizer.param_groups:
        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        group_params = set(group['params'])
        is_head_group = group['params'][0] in head_params_set
        is_alpha_only_group = bool(alpha_params_set) and group_params.issubset(alpha_params_set)

        if is_alpha_only_group and stage == 'joint' and freeze_alpha_in_joint:
            group['lr'] = 0.0
        elif is_head_group:
            group['lr'] = lr_inter
        else:
            # Reproducibility note: keep this step explicit for repeatable experiments.
            if stage == 'warmup_head':
                group['lr'] = 0.0
            else:
                group['lr'] = lr_backbone

    logger.info(f"Optimizer updated for stage: {stage}. LR_backbone: {group['lr']}")


if __name__ == "__main__":
    # Reproducibility note: keep this step explicit for repeatable experiments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='merdcir_mlp_alpha',
                        choices=['mtcir_mlp_alpha', 'merdcir_no_alpha', 'merdcir_mlp_alpha',
                                 'merdcir_cross_attn_alpha', 'lasco_mlp_alpha'])
    parser.add_argument('--mtcir_json_path', type=str, default='mtcir_np/merged.jsonl')
    parser.add_argument('--merdcir_json_path', type=str, default='merdcir_np/test_train.jsonl')
    parser.add_argument('--lasco_json_path', type=str, default='LaSCo/captions/lasco_train.json')
    parser.add_argument('--lasco_eval_json_path', type=str, default='LaSCo/captions/lasco_val.json')
    parser.add_argument('--lasco_image_root', type=str, default='./data/LaSCo/images')
    parser.add_argument('--lasco_lmdb_path', type=str, default='./data/LaSCo/images_224_lmdb')
    parser.add_argument('--lmdb_path', type=str, default='./data/MTCIR/images_224_lmdb')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--warmup_epochs', type=int, default=1, help="Freeze Alpha for the first epochs")
    parser.add_argument('--sc_loss_lambda', type=float, default=30, help="SCHEI loss multiplier")
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--max_np', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--joint_start_step', type=int, default=300,
                        help='Unfreeze the final backbone layer at step n and all parameters at step 2n')
    parser.add_argument('--val_batch_size', type=int, default=None)
    parser.add_argument('--cirr_data_path', type=str, default='./data/CIRR')
    parser.add_argument('--cirr_json_path', type=str, default='cap.rc2.val.json')
    parser.add_argument('--cirr_split_json_path', type=str, default='split.rc2.val.json')
    parser.add_argument('--cirr_lmdb_path', type=str, default='./data/CIRR/images_224_lmdb')
    parser.add_argument('--selection_alpha', type=float, default=0.2)
    parser.add_argument('--selection_beta', type=float, default=0.8)
    parser.add_argument('--training_log_path', type=str, default='./checkpoints/training_log.json')
    parser.add_argument('--topk_json_path', type=str, default='./checkpoints/topk_checkpoints.json')
    parser.add_argument('--topk_checkpoint_dir', type=str, default='./checkpoints/topk')
    parser.add_argument('--resume_path', type=str, default='./checkpoints/checkpoint.pth.tar')
    parser.add_argument('--skip_cirr_validation', action='store_true')
    parser.add_argument(
        '--freeze_alpha_in_joint',
        dest='freeze_alpha_in_joint',
        action='store_true',
        default=True,
        help='Freeze alpha generator parameters during the joint stage. Enabled by default.'
    )
    parser.add_argument(
        '--train_alpha_in_joint',
        dest='freeze_alpha_in_joint',
        action='store_false',
        help='Keep alpha generator trainable during the joint stage.'
    )
    args = parser.parse_args()
    top_k = args.max_np // 2
    rand_k = args.max_np - top_k
    val_batch_size = args.val_batch_size or args.batch_size

    # Reproducibility note: keep this step explicit for repeatable experiments.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method_config = {
        'mtcir_mlp_alpha': {
            'model_method': 'cross_attn_alpha',
            'dataset_cls': MTCIRDataset,
            'json_path': args.mtcir_json_path,
            'eval_json_path': 'mtcir_np/eval/eval_subset.jsonl'
        },
        'merdcir_no_alpha': {
            'model_method': 'cross_attn',
            'dataset_cls': MerdCIRDataset,
            'json_path': args.merdcir_json_path,
            'eval_json_path': 'merdcir_np/eval/eval_subset.jsonl'
        },
        'merdcir_mlp_alpha': {
            'model_method': 'cross_attn_alpha',
            'dataset_cls': MerdCIRDataset,
            'json_path': args.merdcir_json_path,
            'eval_json_path': 'merdcir_np/eval/eval_subset.jsonl'
        },
        'merdcir_cross_attn_alpha': {
            'model_method': 'cross_pooling_alpha',
            'dataset_cls': MerdCIRDataset,
            'json_path': args.merdcir_json_path,
            'eval_json_path': 'merdcir_np/eval/eval_subset.jsonl'
        },
        'lasco_mlp_alpha': {
            'model_method': 'cross_attn_alpha',
            'dataset_cls': LaSCoDataset,
            'json_path': args.lasco_json_path,
            'eval_json_path': args.lasco_eval_json_path,
            'lmdb_path': args.lasco_lmdb_path,
            'dataset_kwargs': {'image_root': args.lasco_image_root}
        },
    }
    selected_config = method_config[args.method]
    selected_lmdb_path = selected_config.get('lmdb_path', args.lmdb_path)
    dataset_kwargs = selected_config.get('dataset_kwargs', {})
    model = ScheiCIR(method=selected_config['model_method'], temperature=args.temperature).to(device)
    dataset = selected_config['dataset_cls'](
        data_path='./data',
        lmdb_path=selected_lmdb_path,
        json_path=selected_config['json_path'],
        **dataset_kwargs
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,  # Reproducibility note: keep this step explicit for repeatable experiments.
                            pin_memory=True, collate_fn=dataset.custom_collate_fn)
    evaluator = ScheiEvaluator(
        dbpath="./chroma_db",
        lmdb_path=selected_lmdb_path,
        json_path=selected_config['eval_json_path'],
        dataset_cls=selected_config['dataset_cls'],
        **dataset_kwargs
    )
    gallery_loader = DataLoader(evaluator.dataset, args.batch_size, shuffle=False, num_workers=8,
                                pin_memory=True, collate_fn=dataset.custom_collate_fn)
    added_ids = set()
    in_domain_val_dataset = selected_config['dataset_cls'](
        data_path='./data',
        lmdb_path=selected_lmdb_path,
        json_path=selected_config['eval_json_path'],
        **dataset_kwargs
    )
    in_domain_val_loader = DataLoader(
        in_domain_val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=in_domain_val_dataset.custom_collate_fn,
    )

    cirr_val_loader = None
    cirr_json_full_path = (
        args.cirr_json_path if os.path.isabs(args.cirr_json_path)
        else os.path.join(args.cirr_data_path, args.cirr_json_path)
    )
    if not args.skip_cirr_validation and os.path.isfile(cirr_json_full_path):
        cirr_val_dataset = EvalJsonDataset(
            data_path=args.cirr_data_path,
            json_path=args.cirr_json_path,
            lmdb_path=args.cirr_lmdb_path,
            require_target=True,
            split_json_path=args.cirr_split_json_path,
        )
        cirr_val_loader = DataLoader(
            cirr_val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=EvalJsonDataset.collate_fn,
        )
    else:
        logger.warning(
            "CIRR validation is disabled. Provide --cirr_json_path with target labels "
            "or pass --skip_cirr_validation explicitly."
        )

    # Reproducibility note: keep this step explicit for repeatable experiments.
    # Reproducibility note: keep this step explicit for repeatable experiments.
    # Reproducibility note: keep this step explicit for repeatable experiments.
    optimizer = setup_initial_optimizer(model, lr_inter=1e-4)

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_accuracy = 0.0
    resume_path = args.resume_path
    checkpoint = None

    if os.path.isfile(resume_path):
        logger.info(f"Resuming training from {resume_path} ...")

        # Reproducibility note: keep this step explicit for repeatable experiments.
        checkpoint = torch.load(resume_path, map_location=device)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        model.load_state_dict(checkpoint['state_dict'])

        # Reproducibility note: keep this step explicit for repeatable experiments.
        global_step = checkpoint.get('global_step', checkpoint.get('step', 0))
        start_step_in_epoch = checkpoint.get('step', 0)
        optimizer = get_optimizer(
            model,
            stage=resolve_training_stage(global_step, args.joint_start_step),
            freeze_alpha_in_joint=args.freeze_alpha_in_joint,
        )

        # Reproducibility note: keep this step explicit for repeatable experiments.
        start_epoch = checkpoint['epoch'] + 1
        if 'best_accuracy' in checkpoint:
            best_accuracy = checkpoint['best_accuracy']

        logger.info(f"Resume succeeded; continuing from epoch  {start_epoch} .")
    else:
        logger.info("No checkpoint found; starting training from scratch.")
    running_infonce = 0.0
    running_sc = 0.0
    running_total = 0.0
    best_accuracy = 0.0
    training_log = []
    topk_checkpoints = []
    if os.path.isfile(args.training_log_path):
        with open(args.training_log_path, "r", encoding="utf-8") as f:
            training_log = json.load(f)
    if os.path.isfile(args.topk_json_path):
        with open(args.topk_json_path, "r", encoding="utf-8") as f:
            topk_checkpoints = json.load(f)

    # Reproducibility note: keep this step explicit for repeatable experiments.
    logger.info(f"Start training with method: {args.method}")
    logger.info(f"Freeze alpha generator in joint stage: {args.freeze_alpha_in_joint}")

    for epoch in range(checkpoint['epoch'] if checkpoint else 0, args.epochs):
        model.train()
        debug = True
        current_stage = resolve_training_stage(global_step, args.joint_start_step)
        logger.info(f"Epoch {epoch}: {current_stage} stage at global_step={global_step}")
        apply_training_stage(model, current_stage, freeze_alpha_in_joint=args.freeze_alpha_in_joint)
        optimizer = get_optimizer(
            model,
            stage=current_stage,
            freeze_alpha_in_joint=args.freeze_alpha_in_joint,
        )
        epoch_loss = 0

        step_offset = start_step_in_epoch if epoch == start_epoch else 0
        for step, batch in enumerate(tqdm(dataloader)):
            if step < step_offset:
                continue
            ids = batch['id']
            ref_imgs = batch['image'].to(device)
            target_imgs = batch['target_img'].to(device)
            texts = batch['text']  # Reproducibility note: keep this step explicit for repeatable experiments.
            nps_batch = batch['np']
            # Reproducibility note: keep this step explicit for repeatable experiments.
            fused_feat, attn_map = model(ref_imgs, texts, return_attention=True)
            target_feat = model.visual_backbone(target_imgs, get_embeddings=True)
            ref_feat = model.visual_backbone(ref_imgs, get_embeddings=True)
            # Reproducibility note: keep this step explicit for repeatable experiments.
            # Reproducibility note: keep this step explicit for repeatable experiments.
            # Reproducibility note: keep this step explicit for repeatable experiments.
            with torch.no_grad():
                avg_attn = attn_map.mean(dim=1)  # [B, V, T]

            # Reproducibility note: keep this step explicit for repeatable experiments.
            all_np_attns = []
            all_np_feats = []
            all_visual_feats = []
            final_np_counts = []  # Reproducibility note: keep this step explicit for repeatable experiments.

            query_token_feats = model.text_encoder.get_token_embeddings(texts)

            for i, nps in enumerate(nps_batch):
                image_np_attns = []
                image_np_feats = []
                image_np_scores = []  # Reproducibility note: keep this step explicit for repeatable experiments.

                # Reproducibility note: keep this step explicit for repeatable experiments.
                valid_num = 0
                for np_item in nps:
                    start, end = np_item[1]
                    if start > 77 or end > 77:
                        continue
                    valid_num += 1
                    # Reproducibility note: keep this step explicit for repeatable experiments.
                    # Reproducibility note: keep this step explicit for repeatable experiments.
                    if start - 1 >= end or start - 1 < 0:
                        logger.info(f"Warning: Invalid slice range for item {i}. Skipping...")
                        token_attn_raw = torch.zeros(avg_attn.shape[1] - 1).to(avg_attn.device)  # Reproducibility note: keep this step explicit for repeatable experiments.
                    else:
                        try:
                            token_attn_raw = avg_attn[i, 1:, start - 1:end].max(dim=1).values
                        except IndexError:
                            logger.info("Invalid token span during attention pooling: start=%s, end=%s", start, end)
                            assert False
                    importance_score = token_attn_raw.mean().item()

                    h = w = int(token_attn_raw.shape[0] ** 0.5)
                    # token_attn = token_attn_raw.view(h, w)
                    image_np_attns.append(token_attn_raw)
                    image_np_scores.append(importance_score)

                    # Reproducibility note: keep this step explicit for repeatable experiments.
                    span_feats = query_token_feats[i, start - 1:end, :]
                    np_attn_map = avg_attn[i, 1:, start - 1:end]  # [V, L_w]
                    token_weights = np_attn_map.sum(dim=0)  # [L_w]
                    token_weights = token_weights / (token_weights.sum() + 1e-8)

                    np_feat = torch.matmul(span_feats.t(), token_weights.unsqueeze(1)).squeeze()
                    image_np_feats.append(np_feat)

                # Reproducibility note: keep this step explicit for repeatable experiments.
                num_nps = valid_num
                if num_nps > args.max_np:
                    # Reproducibility note: keep this step explicit for repeatable experiments.
                    indices = np.argsort(image_np_scores)[::-1].tolist()

                    top_indices = indices[:top_k]
                    remaining_indices = indices[top_k:]

                    # Reproducibility note: keep this step explicit for repeatable experiments.
                    random_indices = random.sample(remaining_indices, rand_k)

                    # Reproducibility note: keep this step explicit for repeatable experiments.
                    keep_indices = top_indices + random_indices
                else:
                    # Reproducibility note: keep this step explicit for repeatable experiments.
                    keep_indices = list(range(num_nps))

                # Reproducibility note: keep this step explicit for repeatable experiments.
                for idx in keep_indices:
                    all_np_attns.append(image_np_attns[idx])
                    all_np_feats.append(image_np_feats[idx])

                final_np_counts.append(len(keep_indices))

            # Reproducibility note: keep this step explicit for repeatable experiments.
            for i, n in enumerate(final_np_counts):
                if n == 1:
                    continue
                for j in range((n ** 2) - n):
                    all_visual_feats.append(fused_feat[i, :])
            # Reproducibility note: keep this step explicit for repeatable experiments.
            infonce_loss = model.compute_cir_loss(fused_feat, target_feat, ref_feat)
            if all_np_attns and all_visual_feats:
                all_np_attns = torch.stack(all_np_attns)  # (Total_Selected_NPs, H, W)
                all_np_feats = torch.stack(all_np_feats)  # (Total_Selected_NPs, D)
                all_visual_feats = torch.stack(all_visual_feats)
                sc_loss = model.compute_weighted_contrastive(
                    all_np_feats,
                    all_np_attns,
                    final_np_counts,
                    all_visual_feats,
                    debug
                )
            else:
                sc_loss = fused_feat.new_tensor(0.0)
            sc_multiplier = 1
            if current_stage != "warmup_head":
                sc_multiplier = 5
            total_loss = infonce_loss + args.sc_loss_lambda * sc_loss * sc_multiplier
            debug = False

            optimizer.zero_grad()
            total_loss.backward()
            if debug:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"Warning: {name} has no gradient!")
                    else:
                        print(f"OK: {name} gradient mean: {param.grad.abs().mean().item()}")
                    break  # Reproducibility note: keep this step explicit for repeatable experiments.
            optimizer.step()

            # Reproducibility note: keep this step explicit for repeatable experiments.
            current_step = step + 1
            global_step += 1
            next_stage = resolve_training_stage(global_step, args.joint_start_step)
            if next_stage != current_stage:
                logger.info(f"Switching to {next_stage}")
                current_stage = next_stage
                apply_training_stage(model, current_stage, freeze_alpha_in_joint=args.freeze_alpha_in_joint)

                # Reproducibility note: keep this step explicit for repeatable experiments.
                update_optimizer_stage(
                    optimizer,
                    model,
                    stage=current_stage,
                    lr_inter=1e-4,
                    lr_backbone=5e-7,  # Reproducibility note: keep this step explicit for repeatable experiments.
                    freeze_alpha_in_joint=args.freeze_alpha_in_joint,
                )

            if current_step % 50 == 0:
                logger.info(f"  Total Loss:   {total_loss.item():.4f}")
                logger.info(f"  InfoNCE Loss: {infonce_loss.item():.4f}")
                logger.info(f"  SC Loss:      {sc_loss.item():.8f}")
            if current_step % 300 == 0:
                checkpoint_data = {
                    'epoch': epoch,
                    'step': current_step,
                    'global_step': global_step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                save_checkpoint(checkpoint_data, is_best=False)
                logger.info("Recovery checkpoint saved")

                in_domain_metrics = evaluate_in_domain(model, in_domain_val_loader, device)
                if cirr_val_loader is not None:
                    cirr_metrics = evaluate_cirr(model, cirr_val_loader, device)
                else:
                    cirr_metrics = {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "R@50": 0.0, "mAP": 0.0}

                selection_score = (
                    args.selection_alpha * in_domain_metrics["mAP"]
                    + args.selection_beta * cirr_metrics["mAP"]
                )
                validation_log = {
                    "epoch": epoch,
                    "step": current_step,
                    "global_step": global_step,
                    "in_domain": in_domain_metrics,
                    "cirr": cirr_metrics,
                    "selection_score": float(selection_score),
                }
                training_log.append(validation_log)
                os.makedirs(os.path.dirname(args.training_log_path), exist_ok=True)
                save_json(args.training_log_path, training_log)

                topk_checkpoint_data = {
                    'epoch': epoch,
                    'step': current_step,
                    'global_step': global_step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics': validation_log,
                }
                topk_checkpoints = save_topk_checkpoint(
                    topk_checkpoints,
                    topk_checkpoint_data,
                    epoch,
                    selection_score,
                    {"in_domain": in_domain_metrics, "cirr": cirr_metrics},
                    args.topk_checkpoint_dir,
                    k=3,
                    step=current_step,
                )
                os.makedirs(os.path.dirname(args.topk_json_path), exist_ok=True)
                save_json(args.topk_json_path, topk_checkpoints)

                message = (
                    f"Epoch {epoch}, step {current_step}\n"
                    f"--------------------------------\n"
                    f"In-domain:\n  {format_metrics(in_domain_metrics)}\n"
                    f"CIRR:\n  {format_metrics(cirr_metrics)}\n"
                    f"Selection score: {selection_score:.6f}"
                )
                print(message)
                logger.info(message)
                apply_training_stage(model, current_stage, freeze_alpha_in_joint=args.freeze_alpha_in_joint)

        start_step_in_epoch = 0

    if topk_checkpoints:
        print("Top-3 checkpoints (by selection score):")
        logger.info("Top-3 checkpoints (by selection score):")
        for i, entry in enumerate(topk_checkpoints[:3], start=1):
            in_map = entry["metrics"]["in_domain"]["mAP"]
            cirr_map = entry["metrics"]["cirr"]["mAP"]
            line = (
                f"Checkpoint {i}: path={entry['path']}, epoch={entry['epoch']}, "
                f"in-domain mAP={in_map:.6f}, CIRR mAP={cirr_map:.6f}, score={entry['score']:.6f}"
            )
            print(line)
            logger.info(line)

        best_generalization = max(topk_checkpoints, key=lambda x: x["metrics"]["cirr"]["mAP"])
        best_in_domain = max(topk_checkpoints, key=lambda x: x["metrics"]["in_domain"]["mAP"])
        generalization_line = f"Best for generalization: {best_generalization['path']}"
        in_domain_line = f"Best for in-domain: {best_in_domain['path']}"
        print(generalization_line)
        print(in_domain_line)
        logger.info(generalization_line)
        logger.info(in_domain_line)
