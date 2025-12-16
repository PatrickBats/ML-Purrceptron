import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

from shared.models import CatCNN
from shared.dataset import CatBreedDataset
from shared.data_augmentation import CatBreedAugmentation


def load_model(checkpoint_path, num_classes=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build model architecture
    model = CatCNN(num_classes=num_classes, dropout_rate=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")

    # Try to get best validation accuracy (different checkpoint formats)
    if 'metrics' in checkpoint and 'best_val_acc' in checkpoint['metrics']:
        print(f"Best validation accuracy: {checkpoint['metrics']['best_val_acc']:.2f}%")
    elif 'best_val_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")

    return model, device, checkpoint


def compute_confusion_matrix(model, dataloader, device, num_classes=8):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Computing confusion matrix"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = outputs.max(1)

            # Update confusion matrix
            for true_label, pred_label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[true_label][pred_label] += 1

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, breed_names, output_path, normalize=False):
    if normalize:
        # Normalize by row (true label) to show percentages
        confusion_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_to_plot = confusion_matrix_norm
        fmt = '.2%'
        title = 'Normalized Confusion Matrix\nFrom-Scratch CNN'
    else:
        cm_to_plot = confusion_matrix
        fmt = 'd'
        title = 'Confusion Matrix\nFrom-Scratch CNN'

    # Create figure
    plt.figure(figsize=(14, 12))

    # Plot heatmap with larger fonts
    sns.heatmap(
        cm_to_plot,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=breed_names,
        yticklabels=breed_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        square=True,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'fontsize': 14}  # Larger annotation font
    )

    plt.title(title, fontsize=18, pad=20, fontweight='bold')
    plt.ylabel('True Breed', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Breed', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_confusions(confusion_matrix, breed_names):
    # Find most confused pairs (excluding diagonal)
    confusion_pairs = []
    for i in range(len(breed_names)):
        for j in range(len(breed_names)):
            if i != j and confusion_matrix[i][j] > 0:
                confusion_pairs.append((
                    breed_names[i],
                    breed_names[j],
                    confusion_matrix[i][j],
                    confusion_matrix[i][j] / confusion_matrix[i].sum() * 100  # percentage
                ))

    # Sort by count
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    for true_breed, pred_breed, count, percentage in confusion_pairs[:10]:
        print(f"{true_breed:<20} {pred_breed:<20} {count:<8} {percentage:>6.1f}%")

    accuracies = []
    for i, breed in enumerate(breed_names):
        correct = confusion_matrix[i][i]
        total = confusion_matrix[i].sum()
        accuracy = correct / total * 100 if total > 0 else 0
        accuracies.append(accuracy)
        print(f"{breed:<20} {correct:<10} {total:<10} {accuracy:>6.2f}%")

    print("-" * 60)
    print(f"{'Overall':<20} {confusion_matrix.diagonal().sum():<10} {confusion_matrix.sum():<10} {np.mean(accuracies):>6.2f}%")

    return confusion_pairs


def main():
    # Paths
    checkpoint_path = Path('experiments/from_scratch_5layer/checkpoints/best.pth')
    output_dir = Path('experiments/from_scratch_5layer')
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        print("Please make sure you have trained the from-scratch model first.")
        return

    # Load model
    model, device, checkpoint = load_model(checkpoint_path)

    # Breed names
    breed_names = [
        'Bengal', 'Bombay', 'British Shorthair', 'Maine Coon',
        'Persian', 'Ragdoll', 'Russian Blue', 'Siamese'
    ]

    # Setup test data
    print("\nLoading test dataset...")
    aug = CatBreedAugmentation(mode='from_scratch')
    base_dir = Path(__file__).parent.parent / 'data'

    test_dataset = CatBreedDataset(
        csv_file=str(base_dir / 'processed_data/test.csv'),
        transform=aug.get_val_transform()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    confusion_matrix = compute_confusion_matrix(model, test_loader, device, num_classes=8)

    # Save confusion matrix data
    cm_data = {
        'confusion_matrix': confusion_matrix.tolist(),
        'breed_names': breed_names,
        'total_samples': int(confusion_matrix.sum()),
        'overall_accuracy': float(confusion_matrix.diagonal().sum() / confusion_matrix.sum() * 100)
    }

    with open(output_dir / 'confusion_matrix.json', 'w') as f:
        json.dump(cm_data, f, indent=2)

    print(f"\nOverall Test Accuracy: {cm_data['overall_accuracy']:.2f}%")

    # Plot raw counts
    print("\nGenerating confusion matrix visualizations...")
    plot_confusion_matrix(
        confusion_matrix,
        breed_names,
        output_dir / 'confusion_matrix_raw.png',
        normalize=False
    )

    # Plot normalized (percentages)
    plot_confusion_matrix(
        confusion_matrix,
        breed_names,
        output_dir / 'confusion_matrix_normalized.png',
        normalize=True
    )

    # Analyze confusions
    confusion_pairs = analyze_confusions(confusion_matrix, breed_names)




if __name__ == "__main__":
    main()
