import torchvision.transforms as T

# 这是 CLIP 模型官方使用的标准化参数 (Mean 和 Std)
# 如果你用的是普通的 ResNet，通常是 ImageNet 的参数 [0.485, 0.456, 0.406]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_train_transform(image_size=224):
    """
    训练集专用的 Transform：包含随机数据增强
    """
    return T.Compose([
        # 1. 随机裁剪并缩放 (CIR任务中常用，能让模型关注不同局部)
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),

        # 2. 随机水平翻转 (概率 0.5)
        # 注意：如果你的文本指令包含“左边/右边”，最好关掉这个！
        T.RandomHorizontalFlip(p=0.5),

        # 3. 颜色微调 (可选，防止对颜色过度敏感，但CIR里改颜色的指令多，建议轻量使用或不用)
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),

        # 4. 转换为 Tensor (H, W, C -> C, H, W，并除以 255)
        T.ToTensor(),

        # 5. 标准化 (减去均值，除以标准差)
        T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
    ])


def get_val_transform(image_size=224):
    """
    验证/测试集专用的 Transform：确定性操作，无随机性
    """
    return T.Compose([
        # 1. 等比例缩放，使得最短边达到设定尺寸
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),

        # 2. 从中心裁剪出严格的正方形
        T.CenterCrop(image_size),

        # 3. 转换为 Tensor
        T.ToTensor(),

        # 4. 标准化
        T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
    ])