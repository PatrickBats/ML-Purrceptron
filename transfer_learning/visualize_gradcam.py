import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from load_model import load_model


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class):
        output = self.model(input_image)

        self.model.zero_grad()

        target = output[0, target_class]
        target.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        device = input_image.device
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def visualize_gradcam(image_path, model, breed_names, gradcam, output_path):
    device = next(model.parameters()).device
    transform = get_transforms()

    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_breed = breed_names[predicted_idx.item()]
    predicted_confidence = confidence.item()

    cam = gradcam.generate_cam(input_tensor, predicted_idx.item())

    cam_resized = np.array(Image.fromarray(cam).resize(original_image.size, Image.BILINEAR))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(original_image)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Overlay\nPredicted: {predicted_breed}\nConfidence: {predicted_confidence:.2%}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    model, breed_names, config = load_model()

    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    base_dir = Path(__file__).parent.parent / 'data'
    test_csv = base_dir / 'processed_data/test.csv'
    df = pd.read_csv(test_csv)

    output_dir = Path('gradcam_visualizations')
    output_dir.mkdir(exist_ok=True)


    for breed in tqdm(breed_names):
        breed_images = df[df['breed'] == breed]
        if len(breed_images) == 0:
            continue

        sample = breed_images.sample(1).iloc[0]
        image_path = base_dir / 'Datacleaning' / sample['dataset'] / 'images' / sample['breed'] / sample['filename']

        output_path = output_dir / f'{breed.replace(" ", "_")}_gradcam.png'

        try:
            visualize_gradcam(str(image_path), model, breed_names, gradcam, output_path)
        except Exception as e:
            continue



if __name__ == '__main__':
    main()
