import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPTokenizerFast, CLIPTextModel, CLIPModel
from .interaction import CrossAttentionBlock, SelfAttentionBlock, AlphaGenerator, TransformerAlphaGenerator
import torch.nn.functional as F


def masked_softmax(logits, mask, dim=-1):
    """
    logits: (Total_Pairs, 1)
    mask: Boolean matrix indicating which pairs are valid
    """
    # Reproducibility note: keep this step explicit for repeatable experiments.
    logits_masked = logits.masked_fill(~mask, float('-inf'))
    return torch.softmax(logits_masked, dim=dim)


class TextEncoder(nn.Module):
    def __init__(self, size="B"):
        super().__init__()
        if size == "B":
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        elif size == "L":
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
            self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
        elif size == "H":
            self.model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.tokenizer = CLIPTokenizerFast.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding="max_length", return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def get_token_embeddings(self, texts):
        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.model.device)

        # 2. Forward pass
        outputs = self.model(**inputs)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        token_embeddings = outputs.last_hidden_state

        # Reproducibility note: keep this step explicit for repeatable experiments.
        return token_embeddings


class AttentionPooler(nn.Module):
    def __init__(self, embed_dim, num_heads, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.attention = CrossAttentionBlock(embed_dim, embed_dim, embed_dim, num_heads=num_heads)

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        q = self.queries.expand(batch_size, -1, -1)
        pooled_output, _ = self.attention(q, encoder_outputs)
        return pooled_output


class VisionEncoder(nn.Module):
    def __init__(self, size="B"):
        super().__init__()
        # Reproducibility note: keep this step explicit for repeatable experiments.
        path_map = {
            "B": "openai/clip-vit-base-patch32",
            "L": "openai/clip-vit-large-patch14",
            "H": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        }
        model_path = path_map.get(size, path_map["B"])

        # Reproducibility note: keep this step explicit for repeatable experiments.
        from transformers import CLIPVisionModel
        self.model = CLIPVisionModel.from_pretrained(model_path)
        self.processor = CLIPImageProcessor.from_pretrained(model_path)

    def forward(self, images, get_embeddings=False):
        # Reproducibility note: keep this step explicit for repeatable experiments.
        if isinstance(images, torch.Tensor):
            # Reproducibility note: keep this step explicit for repeatable experiments.
            pixel_values = images
        else:
            # Reproducibility note: keep this step explicit for repeatable experiments.
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values']

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        pixel_values = pixel_values.to(self.model.device)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        outputs = self.model(pixel_values=pixel_values)

        if get_embeddings:
            # Reproducibility note: keep this step explicit for repeatable experiments.
            return outputs.pooler_output

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        return outputs.last_hidden_state


class ScheiCIR(nn.Module):
    def __init__(self, method='cross_attn_alpha', temperature=0.07, backbone_size='B', num_cross_attn_layers=4):
        super().__init__()
        self.size = backbone_size
        if self.size == "B":
            self.text_dim = 512
            self.vision_dim = 768
            self.patch_size = 7
        elif self.size == "L":
            self.text_dim = 768
            self.vision_dim = 1024
            self.patch_size = 14
        else:  # H
            self.text_dim = 1024
            self.vision_dim = 1280
            self.patch_size = 14
        self.method = method
        self.temperature = temperature
        self.sc_temperature = 0.07
        # Reproducibility note: keep this step explicit for repeatable experiments.
        self.visual_backbone = VisionEncoder(self.size)
        self.text_encoder = TextEncoder(self.size)
        self.interaction = nn.ModuleList([
            CrossAttentionBlock(
                dim_q=self.vision_dim,
                dim_kv=self.text_dim,
                out_dim=self.vision_dim
            )
            for _ in range(num_cross_attn_layers)
        ])
        self.attn_pooler = AttentionPooler(
            embed_dim=self.vision_dim,
            num_heads=8,
            num_queries=1
        )

        self.alpha_gen = None
        if 'alpha' in method:
            if 'pooling' in method:
                self.alpha_gen = TransformerAlphaGenerator(self.text_dim, self.vision_dim, self.patch_size)
            else:
                self.alpha_gen = AlphaGenerator(self.text_dim, self.vision_dim, self.patch_size)

    def forward(self, img, text, return_attention=False):
        # Reproducibility note: keep this step explicit for repeatable experiments.
        img_feat = self.visual_backbone(img)  # (B, N, D)
        txt_feat = self.text_encoder(text)  # (B, M, D)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        x = img_feat  # Reproducibility note: keep this step explicit for repeatable experiments.
        for layer in self.interaction:
            # Reproducibility note: keep this step explicit for repeatable experiments.
            delta, attn_map = layer(x, txt_feat)
            x = x + delta  # Reproducibility note: keep this step explicit for repeatable experiments.
        vision_features = x
        pooled_features = self.attn_pooler(vision_features)  # (B, 1, D)
        pooled_features = pooled_features.squeeze(1)  # (B, D)

        if return_attention:
            return pooled_features, attn_map
        return pooled_features

    def compute_cir_loss(self, query_features, target_features, ref_image_features):
        """
        Args:
            query_features: Query features after text fusion [B, D]
            target_features: Target image features [B, D]
            ref_image_features: Original reference image features [B, D]
        """
        # Reproducibility note: keep this step explicit for repeatable experiments.
        query_features = F.normalize(query_features, p=2, dim=-1)
        target_features = F.normalize(target_features, p=2, dim=-1)
        ref_image_features = F.normalize(ref_image_features, p=2, dim=-1)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        logits_target = torch.matmul(query_features, target_features.t()) / self.temperature

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        logits_ref = torch.matmul(query_features, ref_image_features.t()) / self.temperature

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        logits = torch.cat([logits_target, logits_ref], dim=1)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        batch_size = query_features.size(0)
        labels = torch.arange(batch_size, device=query_features.device)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_weighted_contrastive(self, all_np_feats, all_np_attns, np_counts, all_cls, debug):
        """
        all_np_feats: (Total_NPs, D)
        all_np_attns: (Total_NPs, H, W)
        np_counts: List[int]
        """
        total_nps = all_np_feats.size(0)
        device = all_np_feats.device

        # Reproducibility note: keep this step explicit for repeatable experiments.
        batch_idx = torch.cat([torch.full((c,), i, device=device) for i, c in enumerate(np_counts)])
        same_image_mask = (batch_idx.unsqueeze(1) == batch_idx.unsqueeze(0))
        diag_mask = torch.eye(total_nps, device=device).bool()
        valid_mask = same_image_mask & ~diag_mask  # Reproducibility note: keep this step explicit for repeatable experiments.

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        all_np_feats_norm = F.normalize(all_np_feats, p=2, dim=-1)
        S = torch.matmul(all_np_feats_norm, all_np_feats_norm.t()) / self.sc_temperature

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        indices = torch.nonzero(valid_mask, as_tuple=True)
        rows, cols = indices
        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        if not self.alpha_gen:
            alpha_logits = torch.ones((all_cls.shape[0])).to(device)
        else:
            alpha_logits = self.alpha_gen(
                feat_1=all_np_feats[rows],
                feat_2=all_np_feats[cols],
                attn_1=all_np_attns[rows],
                attn_2=all_np_attns[cols],
                cls=all_cls
            )

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        alpha_matrix = torch.full((total_nps, total_nps), -float('inf'), device=device)
        alpha_matrix[rows, cols] = alpha_logits.squeeze()

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        alpha_matrix = torch.softmax(alpha_matrix / self.sc_temperature, dim=1)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        alpha_matrix = torch.nan_to_num(alpha_matrix, nan=0.0)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        exp_S = torch.exp(S)

        # Reproducibility note: keep this step explicit for repeatable experiments.
        weighted_negatives = exp_S * alpha_matrix * valid_mask.float()

        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        # Reproducibility note: keep this step explicit for repeatable experiments.
        numerator = torch.diagonal(exp_S)
        denominator = weighted_negatives.sum(dim=1)  + numerator
        base = numerator / (denominator + 1e-8)

        loss = -torch.log(base).mean()

        return loss
