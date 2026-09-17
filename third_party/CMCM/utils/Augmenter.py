import torch
from torchvision import transforms


class VideoAugmenter:
    def __init__(self):
        # 弱增强：保留基础时空信息
        self.weak_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

        # 强增强：引入更多扰动（用于后门调整）
        self.strong_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
        ])

    def __call__(self, video_tensor):
        """
        Args:
            video_tensor: 输入视频张量 (B, C, T, H, W)，已位于 GPU
        Returns:
            增强后的视频张量 (B, C, T, H, W)，位于 GPU
        """
        # 注意：PyTorch 的 transforms 通常在 CPU 上执行，需先将视频移到 CPU 增强，再移回 GPU
        # （若需 GPU 加速增强，可使用自定义 CUDA 内核或第三方库如 kornia）
        device = video_tensor.device
        video_cpu = video_tensor.cpu()

        # 弱增强
        weak_aug = self.weak_transform(video_cpu.numpy().transpose(0, 2, 3, 4, 1))  # 转为 (B, T, H, W, C)
        weak_aug = torch.tensor(weak_aug).permute(0, 4, 1, 2, 3).to(device)  # 转回 (B, C, T, H, W)

        # 强增强
        strong_aug = self.strong_transform(video_cpu.numpy().transpose(0, 2, 3, 4, 1))
        strong_aug = torch.tensor(strong_aug).permute(0, 4, 1, 2, 3).to(device)

        return weak_aug, strong_aug
