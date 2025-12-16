import torch
from torchvision import transforms
import numpy as np


class CatBreedAugmentation:

    def __init__(self, mode='from_scratch', image_size=224):
        self.mode = mode
        self.image_size = image_size

        if mode == 'from_scratch':
            self.train_transform = self._get_heavy_augmentation()
        elif mode == 'transfer_learning':
            self.train_transform = self._get_moderate_augmentation()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'from_scratch' or 'transfer_learning'")

        self.val_transform = self._get_val_transform()

    def _get_heavy_augmentation(self):
        return transforms.Compose([
            transforms.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ])

    def _get_moderate_augmentation(self):
        return transforms.Compose([
            transforms.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _get_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_train_transform(self):
        return self.train_transform

    def get_val_transform(self):
        return self.val_transform

    def __repr__(self):
        return f"CatBreedAugmentation(mode='{self.mode}', image_size={self.image_size})"


def visualize_augmentations(image_path, num_samples=5, mode='from_scratch'):
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert('RGB')

    aug = CatBreedAugmentation(mode=mode)
    transform = aug.get_train_transform()

    fig, axes = plt.subplots(1, num_samples + 1, figsize=(3 * (num_samples + 1), 3))

    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i in range(num_samples):
        augmented_tensor = transform(img)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm_img = augmented_tensor * std + mean
        denorm_img = torch.clamp(denorm_img, 0, 1)

        img_np = denorm_img.numpy().transpose(1, 2, 0)

        axes[i + 1].imshow(img_np)
        axes[i + 1].set_title(f'Aug {i + 1}')
        axes[i + 1].axis('off')

    plt.suptitle(f'Augmentation Mode: {mode.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'augmentation_preview_{mode}.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: augmentation_preview_{mode}.png")
    plt.show()


if __name__ == "__main__":
    pass
