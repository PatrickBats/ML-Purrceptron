import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.models import CatCNN
from shared.data_augmentation import CatBreedAugmentation


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        # Get device from input
        device = input_image.device

        # Forward pass
        output = self.model(input_image)

        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Generate CAM
        gradients = self.gradients[0]  # [512, 7, 7]
        activations = self.activations[0]  # [512, 7, 7]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [512]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)  # [7, 7] on same device
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        return cam, target_class


def load_trained_model(checkpoint_path, num_classes=8):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    model = CatCNN(num_classes=num_classes)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def visualize_gradcam(image_path, model, breed_names, save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load and preprocess image
    aug = CatBreedAugmentation(mode='from_scratch')
    transform = aug.get_val_transform()  # No augmentation for visualization

    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # Create GradCAM
    gradcam = GradCAM(model, target_layer=model.conv5[0])  # Visualize last conv layer

    # Generate heatmap
    cam, predicted_class = gradcam.generate_cam(input_tensor)

    # Get prediction probabilities
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')

    # Overlay
    # Resize original image to match cam size
    original_resized = np.array(original_image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPredicted: {breed_names[predicted_class]}')
    axes[2].axis('off')

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(breed_names))
    probs_np = probs.cpu().numpy()

    colors = ['red' if i == predicted_class else 'skyblue' for i in range(len(breed_names))]
    plt.barh(y_pos, probs_np, color=colors)
    plt.yticks(y_pos, breed_names)
    plt.xlabel('Probability')
    plt.title(f'Class Probabilities\nPredicted: {breed_names[predicted_class]} ({probs_np[predicted_class]:.2%})')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def visualize_multiple_images(image_paths, model, breed_names, save_dir=None):
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        print(f"\nProcessing {i+1}/{len(image_paths)}: {Path(img_path).name}")

        save_path = None
        if save_dir:
            save_path = Path(save_dir) / f"gradcam_{i+1}.png"

        visualize_gradcam(img_path, model, breed_names, save_path)


def get_random_images_per_breed(csv_path='../data/processed_data/test.csv', num_per_breed=1):
    import pandas as pd

    # Load CSV
    df = pd.read_csv(csv_path)

    # Group by breed and sample
    selected_images = {}
    for breed in df['breed'].unique():
        breed_df = df[df['breed'] == breed]
        sample_size = min(num_per_breed, len(breed_df))
        samples = breed_df.sample(n=sample_size, random_state=None)  # Random each time
        selected_images[breed] = samples['full_path'].tolist()

    return selected_images


def visualize_all_breeds(checkpoint_path='experiments/from_scratch_5layer/checkpoints/best.pth',
                         csv_path='../data/processed_data/test.csv',
                         save_dir='gradcam_visualizations'):
    # Breed names (must match training order)
    breed_names = [
        'Bengal',
        'Bombay',
        'British Shorthair',
        'Maine Coon',
        'Persian',
        'Ragdoll',
        'Russian Blue',
        'Siamese'
    ]

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model, checkpoint = load_trained_model(checkpoint_path, num_classes=len(breed_names))

    # Get random images per breed
    print(f"\nSelecting random images from: {csv_path}")
    selected_images = get_random_images_per_breed(csv_path, num_per_breed=1)

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating GradCAM visualizations...")
    print(f"Saving to: {save_dir}/")

    # Visualize each breed
    for breed in breed_names:
        if breed in selected_images and selected_images[breed]:
            image_path = selected_images[breed][0]
            print(f"\n  Processing: {breed}")
            print(f"    Image: {Path(image_path).name}")

            # Generate safe filename
            safe_breed = breed.replace(' ', '_')
            save_file = save_path / f"{safe_breed}_gradcam.png"

            try:
                visualize_gradcam(image_path, model, breed_names, save_path=save_file)
                plt.close('all')  # Close plots to free memory
            except Exception as e:
                print(f"    ⚠️ Error: {e}")
        else:
            print(f"\n  ⚠️ Skipping {breed}: No images found")

def main():
    # Run automatic visualization for all breeds
    visualize_all_breeds(
        checkpoint_path='experiments/from_scratch_5layer/checkpoints/best.pth',
        csv_path='../data/processed_data/test.csv',  # Use test set
        save_dir='gradcam_visualizations'
    )


if __name__ == "__main__":
    main()
