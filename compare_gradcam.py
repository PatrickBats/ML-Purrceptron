import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.models import CatCNN


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class):
        # Forward pass
        output = self.model(input_image)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        target = output[0, target_class]
        target.backward()

        # Get gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling
        weights = gradients.mean(dim=(1, 2))

        # Weighted combination
        device = input_image.device
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


def load_from_scratch_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path('from_scratch/experiments/from_scratch_5layer/checkpoints/best.pth')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CatCNN(num_classes=8, dropout_rate=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Target layer: last conv layer (conv5)
    target_layer = model.conv5[0]

    return model, target_layer, device


def load_transfer_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path('transfer_learning/experiments/resnet50_transfer/checkpoints/best.pth')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build ResNet50
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features

    # Match checkpoint structure
    try:
        dropout_rate = checkpoint['config'].get('dropout', 0.5)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 8)
        )
    except:
        model.fc = nn.Linear(num_features, 8)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Target layer: last conv layer
    target_layer = model.layer4[-1]

    return model, target_layer, device


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def main():
    # Load models
    fs_model, fs_layer, device = load_from_scratch_model()
    tl_model, tl_layer, _ = load_transfer_model()

    # Create GradCAM objects
    fs_gradcam = GradCAM(fs_model, fs_layer)
    tl_gradcam = GradCAM(tl_model, tl_layer)

    # Breed names
    breed_names = [
        'Bengal', 'Bombay', 'British Shorthair', 'Maine Coon',
        'Persian', 'Ragdoll', 'Russian Blue', 'Siamese'
    ]

    # Load test data
    base_dir = Path('data')
    test_csv = base_dir / 'processed_data/test.csv'
    df = pd.read_csv(test_csv)

    # Select 3 breeds for comparison
    selected_breeds = ['Bengal', 'Persian', 'Siamese']

    # Set random seeds for each breed to get different samples
    # Change the seed value for any breed to get a different image
    breed_seeds = {
        'Bengal': 42,      # Keep same
        'Persian': 100,    # CHANGE THIS to get different Persian image (try 100, 101, 102, etc.)
        'Siamese': 42      # Keep same
    }

    # Get one image per breed
    images_data = []
    for breed in selected_breeds:
        breed_images = df[df['breed'] == breed]
        if len(breed_images) == 0:
            continue

        # Use breed-specific seed for reproducibility
        sample = breed_images.sample(1, random_state=breed_seeds[breed]).iloc[0]

        # Map source to directory name
        source_map = {'kaggle': 'Kaggle', 'oxfordiit': 'OxfordIIT'}
        source_dir = source_map.get(sample['source'], sample['source'].capitalize())

        image_path = base_dir / 'Datacleaning' / source_dir / 'images' / sample['breed'] / sample['filename']

        images_data.append({
            'breed': breed,
            'path': str(image_path)
        })

    # Create 3×5 grid
    fig, axes = plt.subplots(len(images_data), 5, figsize=(25, 15))

    transform = get_transforms()

    for row_idx, img_data in enumerate(images_data):
        breed = img_data['breed']
        image_path = img_data['path']

        # Load image
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            fs_output = fs_model(input_tensor)
            fs_probs = F.softmax(fs_output, dim=1)[0]
            fs_conf, fs_pred = torch.max(fs_probs, 0)
            fs_breed = breed_names[fs_pred.item()]

            tl_output = tl_model(input_tensor)
            tl_probs = F.softmax(tl_output, dim=1)[0]
            tl_conf, tl_pred = torch.max(tl_probs, 0)
            tl_breed = breed_names[tl_pred.item()]

        # Generate GradCAMs
        fs_cam = fs_gradcam.generate_cam(input_tensor, fs_pred.item())
        tl_cam = tl_gradcam.generate_cam(input_tensor, tl_pred.item())

        # Resize CAMs to match image
        fs_cam_resized = np.array(Image.fromarray(fs_cam).resize(original_image.size, Image.BILINEAR))
        tl_cam_resized = np.array(Image.fromarray(tl_cam).resize(original_image.size, Image.BILINEAR))

        # Plot in row
        # Column 0: Original
        axes[row_idx, 0].imshow(original_image)
        axes[row_idx, 0].set_title(f'{breed}\n(Ground Truth)', fontsize=14, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Column 1: From-Scratch Heatmap
        axes[row_idx, 1].imshow(fs_cam_resized, cmap='jet')
        axes[row_idx, 1].set_title('From-Scratch\nGradCAM', fontsize=14, fontweight='bold')
        axes[row_idx, 1].axis('off')

        # Column 2: From-Scratch Overlay
        axes[row_idx, 2].imshow(original_image)
        axes[row_idx, 2].imshow(fs_cam_resized, cmap='jet', alpha=0.5)
        title_color = 'green' if fs_breed == breed else 'red'
        axes[row_idx, 2].set_title(f'From-Scratch\nPred: {fs_breed}\n{fs_conf:.1%}',
                                     fontsize=14, fontweight='bold', color=title_color)
        axes[row_idx, 2].axis('off')

        # Column 3: Transfer Learning Heatmap
        axes[row_idx, 3].imshow(tl_cam_resized, cmap='jet')
        axes[row_idx, 3].set_title('Transfer Learning\nGradCAM', fontsize=14, fontweight='bold')
        axes[row_idx, 3].axis('off')

        # Column 4: Transfer Learning Overlay
        axes[row_idx, 4].imshow(original_image)
        axes[row_idx, 4].imshow(tl_cam_resized, cmap='jet', alpha=0.5)
        title_color = 'green' if tl_breed == breed else 'red'
        axes[row_idx, 4].set_title(f'Transfer Learning\nPred: {tl_breed}\n{tl_conf:.1%}',
                                     fontsize=14, fontweight='bold', color=title_color)
        axes[row_idx, 4].axis('off')

    plt.suptitle('GradCAM Comparison: From-Scratch CNN vs Transfer Learning (ResNet50)',
                 fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    output_path = Path('gradcam_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
