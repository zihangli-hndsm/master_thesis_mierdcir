import torchvision.transforms as T

# Normalization constants used by OpenAI CLIP models.
# Use ImageNet normalization instead when reproducing ResNet baselines.
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_train_transform(image_size=224):
    """
    Training transform with stochastic augmentation.
    """
    return T.Compose([
        # 1. Randomly crop and resize to expose different local regions.
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),

        # 2. Apply horizontal flip with probability 0.5.
        # Disable this for experiments whose text depends on left/right orientation.
        T.RandomHorizontalFlip(p=0.5),

        # 3. Optional light color jitter; keep disabled for color-sensitive CIR experiments.
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),

        # 4. Convert PIL image to tensor in [C, H, W] scaled to [0, 1].
        T.ToTensor(),

        # 5. Normalize with the configured channel mean and standard deviation.
        T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
    ])


def get_val_transform(image_size=224):
    """
    Validation/test transform with deterministic preprocessing.
    """
    return T.Compose([
        # 1. Resize while preserving aspect ratio so the shorter side reaches the target size.
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),

        # 2. Center-crop a square image.
        T.CenterCrop(image_size),

        # 3. Convert PIL image to tensor.
        T.ToTensor(),

        # 4. Normalize tensor channels.
        T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
    ])