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

        if checkpoint_path is None:
            checkpoint_path = Path(__file__).parent / 'experiments/resnet50_transfer/checkpoints/best.pth'

        self.checkpoint_path = checkpoint_path
        self.breed_names = [
            'Bengal', 'Bombay', 'British Shorthair', 'Maine Coon',
            'Persian', 'Ragdoll', 'Russian Blue', 'Siamese'
        ]
        self.num_classes = len(self.breed_names)

    def load_model(self):

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Best epoch: {checkpoint['epoch'] + 1}")
        print(f"Best validation accuracy: {checkpoint['metrics']['best_val_acc']:.2f}%")

    def setup_data(self, batch_size=64, num_workers=4):

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


    def evaluate(self):

        self.model.eval()

        correct = 0
        total = 0

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

        all_predictions = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted = probabilities.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()

                    class_total[true_label] += 1
                    if pred_label == true_label:
                        class_correct[true_label] += 1

                    confusion_matrix[true_label][pred_label] += 1

        overall_acc = 100. * correct / total

        avg_confidence = np.mean(all_confidences) * 100
        print(f"Average Confidence: {avg_confidence:.2f}%")

        print("\nPer-Breed Test Accuracy:")
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


if __name__ == "__main__":
    main()
