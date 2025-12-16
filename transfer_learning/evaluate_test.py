import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from shared.dataset import CatBreedDataset
from shared.data_augmentation import CatBreedAugmentation


class TransferLearningEvaluator:

    def __init__(self, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).parent / 'experiments/resnet50_transfer/checkpoints/best.pth'

        self.checkpoint_path = checkpoint_path
        self.breed_names = [
            'Bengal', 'Bombay', 'British Shorthair', 'Maine Coon',
            'Persian', 'Ragdoll', 'Russian Blue', 'Siamese'
        ]
        self.num_classes = len(self.breed_names)

    def load_model(self):
        print("\n" + "=" * 60)
        print("LOADING TRAINED MODEL")
        print("=" * 60)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Build model architecture
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Print checkpoint info
        print(f"Best epoch: {checkpoint['epoch'] + 1}")
        print(f"Best validation accuracy: {checkpoint['metrics']['best_val_acc']:.2f}%")

    def setup_data(self, batch_size=64, num_workers=4):
        print("\n" + "=" * 60)
        print("SETTING UP TEST DATA")
        print("=" * 60)

        # Use validation transforms (no augmentation)
        aug = CatBreedAugmentation(mode='transfer_learning')

        base_dir = Path(__file__).parent.parent / 'data'

        self.test_dataset = CatBreedDataset(
            csv_file=str(base_dir / 'processed_data/test.csv'),
            transform=aug.get_val_transform()
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"Breeds: {', '.join(self.breed_names)}")

    def evaluate(self):
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)
        print("This may take a few minutes...")

        self.model.eval()

        correct = 0
        total = 0

        # Per-class statistics
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        # Confusion matrix
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

        # Store predictions for analysis
        all_predictions = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted = probabilities.max(1)

                # Overall accuracy
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Store for analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

                # Per-class accuracy
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()

                    class_total[true_label] += 1
                    if pred_label == true_label:
                        class_correct[true_label] += 1

                    # Update confusion matrix
                    confusion_matrix[true_label][pred_label] += 1

        # Calculate overall accuracy
        overall_acc = 100. * correct / total

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Overall Test Accuracy: {overall_acc:.2f}%")
        print(f"Correct: {correct}/{total}")

        # Average confidence
        avg_confidence = np.mean(all_confidences) * 100
        print(f"Average Confidence: {avg_confidence:.2f}%")

        # Per-class accuracy
        print("\nPer-Breed Test Accuracy:")
        print("-" * 60)
        per_class_accs = {}
        for class_idx in range(self.num_classes):
            breed = self.breed_names[class_idx]
            if class_total[class_idx] > 0:
                acc = 100. * class_correct[class_idx] / class_total[class_idx]
                per_class_accs[breed] = acc
                print(f"  {breed:20s}: {acc:6.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")
            else:
                per_class_accs[breed] = 0.0
                print(f"  {breed:20s}: No samples")

        # Find best and worst performing breeds
        best_breed = max(per_class_accs, key=per_class_accs.get)
        worst_breed = min(per_class_accs, key=per_class_accs.get)

        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)
        print(f"Best performing breed:  {best_breed} ({per_class_accs[best_breed]:.2f}%)")
        print(f"Worst performing breed: {worst_breed} ({per_class_accs[worst_breed]:.2f}%)")

        # Show most confused pairs
        print("\nMost Confused Breed Pairs:")
        print("-" * 60)
        confusion_pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and confusion_matrix[i][j] > 0:
                    confusion_pairs.append((
                        self.breed_names[i],
                        self.breed_names[j],
                        confusion_matrix[i][j]
                    ))

        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        for true_breed, pred_breed, count in confusion_pairs[:5]:
            print(f"  {true_breed:20s} → {pred_breed:20s}: {count} times")

        # Save results
        results = {
            'overall_accuracy': overall_acc,
            'average_confidence': float(avg_confidence),
            'correct': int(correct),
            'total': int(total),
            'per_class_accuracy': per_class_accs,
            'confusion_matrix': confusion_matrix.tolist(),
            'breed_names': self.breed_names,
            'best_breed': best_breed,
            'worst_breed': worst_breed
        }

        output_dir = Path(__file__).parent / 'experiments/resnet50_transfer'
        output_file = output_dir / 'test_results.json'

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        return overall_acc, per_class_accs


def main():
    print("=" * 60)
    print("TRANSFER LEARNING MODEL - TEST SET EVALUATION")
    print("=" * 60)
    print("\nEvaluating fine-tuned ResNet50 on held-out test data.")

    # Create evaluator
    evaluator = TransferLearningEvaluator()

    # Load model and data
    evaluator.load_model()
    evaluator.setup_data(batch_size=64, num_workers=4)

    # Evaluate
    overall_acc, per_class_accs = evaluator.evaluate()

    print("\n" + "=" * 60)
    print("=" * 60)
    print(f"\nFinal Test Accuracy: {overall_acc:.2f}%")
    print("\nThis is the final performance on unseen test data.")


if __name__ == "__main__":
    main()
