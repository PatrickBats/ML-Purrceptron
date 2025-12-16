import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import json
from tqdm import tqdm
import time

from shared.dataset import CatBreedDataset
from shared.data_augmentation import CatBreedAugmentation


class TransferLearningTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Setup directories
        self.experiment_dir = Path('experiments') / config['experiment_name']
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self._build_model()

        # Setup data
        self._setup_data()

        # Setup training components
        self._setup_training()

        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'best_val_acc': 0.0,
            'best_epoch': 0
        }

    def _build_model(self):
        print("\nLoading pretrained ResNet50...")
        self.model = models.resnet50(pretrained=True)

        # Freeze early layers (optional - can experiment with this)
        if self.config.get('freeze_backbone', False):
            print("Freezing backbone layers...")
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace final layer with dropout for regularization
        num_features = self.model.fc.in_features
        dropout_rate = self.config.get('dropout', 0.5)  # Default 0.5

        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, self.config['num_classes'])
        )

        self.model = self.model.to(self.device)
        print(f"Added dropout layer (p={dropout_rate}) before final layer")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def _setup_data(self):
        base_dir = Path(__file__).parent.parent / 'data'

        # Create datasets with transfer learning augmentation (moderate)
        train_aug = CatBreedAugmentation(mode='transfer_learning')
        val_aug = CatBreedAugmentation(mode='transfer_learning')  # Use val transforms

        train_dataset = CatBreedDataset(
            csv_file=str(base_dir / 'processed_data/train.csv'),
            transform=train_aug.get_train_transform()
        )

        val_dataset = CatBreedDataset(
            csv_file=str(base_dir / 'processed_data/val.csv'),
            transform=val_aug.get_val_transform()
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)} images")
        print(f"  Val: {len(val_dataset)} images")
        print(f"  Batches per epoch: {len(self.train_loader)}")

    def _setup_training(self):
        self.criterion = nn.CrossEntropyLoss()

        # Lower learning rate for fine-tuning
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / total
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            print(f"  ✓ Saved best model (val_acc: {self.metrics['best_val_acc']:.2f}%)")

    def save_metrics(self):
        metrics_file = self.experiment_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def train(self):
        print(f"\n{'='*60}")
        print(f"Starting Transfer Learning Training - {self.config['experiment_name']}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['learning_rates'].append(current_lr)

            # Check if best model
            is_best = val_acc > self.metrics['best_val_acc']
            if is_best:
                self.metrics['best_val_acc'] = val_acc
                self.metrics['best_epoch'] = epoch

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.2e}")

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            self.save_metrics()

            # Early stopping
            if current_lr < 1e-7:
                print("\nLearning rate too small. Stopping training.")
                break

        # Training complete
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {self.metrics['best_val_acc']:.2f}% (Epoch {self.metrics['best_epoch']+1})")
        print(f"Total Training Time: {elapsed/60:.1f} minutes")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


def main():
    # Training configuration
    config = {
        'experiment_name': 'resnet50_transfer',
        'num_classes': 8,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,  # Lower LR for fine-tuning
        'weight_decay': 1e-4,
        'num_workers': 4,
        'freeze_backbone': False,  # Set True to only train final layer
        'dropout': 0.5  # Dropout rate before final layer (0.5 = 50%)
    }

    print("\nTransfer Learning Configuration:")
    print(json.dumps(config, indent=2))

    # Create trainer and start training
    trainer = TransferLearningTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
