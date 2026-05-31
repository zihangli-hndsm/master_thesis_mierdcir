import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPTokenizerFast, CLIPTextModel, CLIPModel
from .interaction import CrossAttentionBlock, SelfAttentionBlock, AlphaGenerator, TransformerAlphaGenerator
import torch.nn.functional as F


def masked_softmax(logits, mask, dim=-1):
    """
    logits: (Total_Pairs, 1)
    mask: 布尔矩阵，标记哪些对是有效的
    """
    # 将无效位置填充为极小值
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
        # 1. Tokenization: 将文本转为数字 ID
        # padding 和 truncation 确保 batch 内长度一致且不超标（CLIP上限77）
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.model.device)

        # 2. Forward pass
        outputs = self.model(**inputs)

        # 3. 获取 Last Hidden State
        # 形状为 [Batch_Size, Sequence_Length, Hidden_Dim] (例如: 8, 77, 512)
        token_embeddings = outputs.last_hidden_state

        # 可选：如果你想去掉 Padding 部分的影响，可以把 inputs['attention_mask'] 也返回
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
        # 建议指定模型路径
        path_map = {
            "B": "openai/clip-vit-base-patch32",
            "L": "openai/clip-vit-large-patch14",
            "H": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        }
        model_path = path_map.get(size, path_map["B"])

        # 注意：如果你只想要视觉部分，可以使用 CLIPVisionModel 而不是 CLIPModel
        from transformers import CLIPVisionModel
        self.model = CLIPVisionModel.from_pretrained(model_path)
        self.processor = CLIPImageProcessor.from_pretrained(model_path)

    def forward(self, images, get_embeddings=False):
        # 1. 判断是否已经是处理好的 Tensor
        if isinstance(images, torch.Tensor):
            # 如果是 Tensor，直接作为 pixel_values
            pixel_values = images
        else:
            # 如果是 PIL List 或其他原始格式，调用 processor
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values']

        # 2. 确保移动到模型所在设备
        # 注意：即便已经是 Tensor，也需要确保 device 一致
        pixel_values = pixel_values.to(self.model.device)

        # 3. 调用视觉模型
        outputs = self.model(pixel_values=pixel_values)

        if get_embeddings:
            # pooler_output 是 [Batch, Hidden_Size]
            return outputs.pooler_output

        # last_hidden_state 是 [Batch, Sequence_Length, Hidden_Size]
        # 包含所有的 Patch Token，适合做后续的特征融合 (Fusion)
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
        # 1. 用 CLIP
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
        # 1. 提取特征
        img_feat = self.visual_backbone(img)  # (B, N, D)
        txt_feat = self.text_encoder(text)  # (B, M, D)

        # 2. 交互 (Attention)
        # 这里的 fused_feat 用于检索，attn_map 用于生成 alpha
        x = img_feat  # 初始输入通常是图像特征（作为 Query）
        for layer in self.interaction:
            # 每一层将上一层的输出作为输入进行迭代处理
            delta, attn_map = layer(x, txt_feat)
            x = x + delta # 确保这里有残差！
        vision_features = x
        pooled_features = self.attn_pooler(vision_features)  # (B, 1, D)
        pooled_features = pooled_features.squeeze(1)  # (B, D)

        if return_attention:
            return pooled_features, attn_map
        return pooled_features

    def compute_cir_loss(self, query_features, target_features, ref_image_features):
        """
        Args:
            query_features: 融合了文本后的 Query 特征 [B, D]
            target_features: 目标图片的图像特征 [B, D]
            ref_image_features: 原始参考图的图像特征 [B, D]
        """
        # 1. 归一化
        query_features = F.normalize(query_features, p=2, dim=-1)
        target_features = F.normalize(target_features, p=2, dim=-1)
        ref_image_features = F.normalize(ref_image_features, p=2, dim=-1)

        # 2. 计算与 Target 的相似度矩阵 [B, B]
        # 对角线上是正样本 (Query_i, Target_i)
        logits_target = torch.matmul(query_features, target_features.t()) / self.temperature

        # 3. 计算与 Reference 的相似度矩阵 [B, B]
        # 这里所有的组合 (Query_i, Ref_j) 都是负样本，包括对角线上的 (Query_i, Ref_i) cs\\\
        # 这一步是关键：它强制模型区分“修改后的图”和“原图”
        logits_ref = torch.matmul(query_features, ref_image_features.t()) / self.temperature

        # 4. 拼接 Logits [B, 2*B]
        # 每一行现在有 1 个正样本和 (2*B - 1) 个负样本
        logits = torch.cat([logits_target, logits_ref], dim=1)

        # 5. 生成 Label
        # 因为正样本都在 logits_target 的对角线上，即前 B 列的对角线
        # 所以 label 依然是 0 到 B-1
        batch_size = query_features.size(0)
        labels = torch.arange(batch_size, device=query_features.device)

        # 6. 计算 CrossEntropy
        # 模型会努力让 logits_target[i, i] 最大，同时压低 logits_target[i, j] 和 logits_ref[i, j]
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

        # 1. 基础 Mask 准备 (你已经写好的部分)
        batch_idx = torch.cat([torch.full((c,), i, device=device) for i, c in enumerate(np_counts)])
        same_image_mask = (batch_idx.unsqueeze(1) == batch_idx.unsqueeze(0))
        diag_mask = torch.eye(total_nps, device=device).bool()
        valid_mask = same_image_mask & ~diag_mask  # 仅包含同图内的负样本对

        # 2. 计算相似度矩阵 S (用于 InfoNCE 的指数项)
        # 假设 feats 已经过 L2 归一化
        all_np_feats_norm = F.normalize(all_np_feats, p=2, dim=-1)
        S = torch.matmul(all_np_feats_norm, all_np_feats_norm.t()) / self.sc_temperature

        # 3. 提取有效样本对喂入 MLP
        # 只取出 valid_mask 覆盖的同图 NP 对，避免无效计算
        indices = torch.nonzero(valid_mask, as_tuple=True)
        rows, cols = indices
        # 4. 生成 Alpha 权重
        # alpha_flat 的维度是 (Total_Valid_Pairs, 1)
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

        # 将 alpha 填回矩阵格式 [Total_NPs, Total_NPs]
        # 5. 构造并应用 Softmax 归一化
        alpha_matrix = torch.full((total_nps, total_nps), -float('inf'), device=device)
        alpha_matrix[rows, cols] = alpha_logits.squeeze()

        # 对每一行执行 Softmax，使得每张图内每个 NP 的负样本权重和为 1
        # 这样模型就不能通过让所有 alpha 趋向 0 来作弊了
        alpha_matrix = torch.softmax(alpha_matrix / self.sc_temperature, dim=1)

        # 处理那些没有负样本的行（防止 NaN）
        alpha_matrix = torch.nan_to_num(alpha_matrix, nan=0.0)

        # 6. 计算 Soft InfoNCE Loss
        # 这里的“正样本”在你的逻辑里是自己对齐自己（通常相似度最高）
        # 分母 = sum( exp(S_ij) * alpha_ij ) 其中 j 是同图内的其他 NP
        exp_S = torch.exp(S)

        # 关键：只给同图负样本加权
        weighted_negatives = exp_S * alpha_matrix * valid_mask.float()

        # 计算 Loss
        # 注意：这里我们假设正样本是矩阵对角线（虽然值很高，但我们主要惩罚同图内的重叠）
        # 如果你有外部的 Image-Text 对齐得分，请替换 numerator
        numerator = torch.diagonal(exp_S)
        denominator = weighted_negatives.sum(dim=1)  + numerator
        base = numerator / (denominator + 1e-8)

        loss = -torch.log(base).mean()

        return loss
