import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, out_dim, num_heads=8, dropout=0.1):
        super().__init__()
        # dim_q: 文本维度, dim_kv: 图像维度, out_dim: 内部计算和输出的统一维度
        self.num_heads = num_heads
        self.scale = (out_dim // num_heads) ** -0.5

        # 分别映射到统一的内部维度 out_dim
        self.to_q = nn.Linear(dim_q, out_dim)
        self.to_k = nn.Linear(dim_kv, out_dim)
        self.to_v = nn.Linear(dim_kv, out_dim)

        self.proj = nn.Linear(out_dim, out_dim)
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Linear(dim_q, out_dim) if dim_q != out_dim else nn.Identity()

    def forward(self, image_feats, text_feats):
        """
        text_feats: (B, M, D) - Query source
        image_feats: (B, N, D) - Key/Value source
        """
        B, M, D = image_feats.shape
        N = text_feats.shape[1]

        # 1. Layer Norm (Pre-Norm 结构)
        q_input = self.norm_q(image_feats)
        k_input = self.norm_kv(text_feats)
        v_input = self.norm_kv(text_feats)

        # 2. Linear Projections & Reshape for Multi-head
        # q: (B, Heads, M, D_head)
        q = self.to_q(q_input).reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.to_k(k_input).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.to_v(v_input).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # 3. Calculate Attention Scores
        # (B, Heads, M, D_head) @ (B, Heads, D_head, N) -> (B, Heads, M, N)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 4. Weighted Sum
        out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(B, M, D)
        out = self.proj(out)

        # 5. Residual Connection
        out = self.shortcut(image_feats) + self.dropout(out)

        return out, attn_weights


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Q, K, V 全部来自同一个输入
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, D) - 序列长度 L 可以是文本长度 M 或图像 Patch 数 N
        """
        B, L, D = x.shape

        # 1. Pre-Norm
        x_norm = self.norm(x)

        # 2. 一次性生成 Q, K, V 并拆分
        # qkv: (B, L, 3, Heads, D_head) -> 方便后面计算
        qkv = self.to_qkv(x_norm).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 都是 (B, Heads, L, D_head)

        # 3. Attention (标准公式)
        # $$ \text{Attn}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V $$
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(B, L, D)

        # 4. Residual
        return x + self.dropout(self.proj(out)), attn_weights


class SearchingMLP(nn.Module):
    def __init__(self, attn_dim=768, img_dim=768, dropout=0.1):
        super().__init__()

        self.query_image_embedding_size = img_dim
        self.modification_text_embedding_size = attn_dim
        self.input_dim = self.query_image_embedding_size + self.modification_text_embedding_size
        self.output_dim = img_dim

        self.hidden_dim_1 = 1536
        self.hidden_dim_2 = 1024

        self.nn = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim_1),
            nn.ReLU(),
            # nn.LayerNorm(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim_2, self.output_dim),
            # then maybe an activation function that fits the expected output
            # ReLU() for 0-1, tanh for -1-1, ...
        )

        # self.nn.to(dtype=torch.float32, device='cpu')

        # cat(query_image, modification_text) -> cat(target_image, target_description)
        # 1d array -> 1d array (fixed size)

    def forward(self, x):
        return self.nn(x)


class AlphaGenerator(nn.Module):
    def __init__(self, dim_feat, cls_dim, patch_size, hidden_dim=1024):
        super().__init__()
        # 输入维度: feat_i (D) + feat_j (D) + sem_dist (1) + attn_l2 (1)
        input_dim = dim_feat * 2 + cls_dim + patch_size ** 2 * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2), # 红队建议：使用 LeakyReLU 防止死亡神经元
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softmax()
        )

    def forward(self, feat_1, feat_2, attn_1, attn_2, cls):
        return self.mlp(torch.cat([feat_1, feat_2, attn_1, attn_2, cls], 1)).squeeze(-1)


class TransformerAlphaGenerator(nn.Module):
    def __init__(self, dim_feat, cls_dim, patch_size, num_heads=8):
        super().__init__()
        # 核心：交叉注意力层
        dim = dim_feat + patch_size ** 2
        self.cross_attn = CrossAttentionBlock(dim_q=dim, dim_kv=dim, out_dim=dim, num_heads=1)

        # 最终输出重要性权重的头
        self.weight_head = nn.Sequential(
            nn.Linear(dim + cls_dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Softmax()  # 权重归一
        )

    def forward(self, feat_1, feat_2, attn_1, attn_2, cls):

        q = torch.cat([feat_1, attn_1], 1).unsqueeze(1)
        kv = torch.cat([feat_2, attn_2], 1).unsqueeze(1)
        attn_output, _ = self.cross_attn(q, kv)

        # 3. 将聚合了视觉信息的 Token 特征映射为重要性分数
        importance_weights = self.weight_head(torch.cat([attn_output.squeeze(), cls], 1)).squeeze(-1)  # [B, N]

        return importance_weights
