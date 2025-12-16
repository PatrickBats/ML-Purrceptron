import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from shared.models import CatCNN
from shared.dataset import CatBreedDataset
from shared.data_augmentation import CatBreedAugmentation


class FromScratchTrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': [],
            'learning_rates': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def setup_data(self):
        aug = CatBreedAugmentation(mode='from_scratch', image_size=self.config['image_size'])

        base_dir = Path(__file__).parent.parent / 'data'
        self.train_dataset = CatBreedDataset(
            csv_file=str(base_dir / 'processed_data/train.csv'),
            transform=aug.get_train_transform()
        )

        self.val_dataset = CatBreedDataset(
            csv_file=str(base_dir / 'processed_data/val.csv'),
            transform=aug.get_val_transform()
        )

        self.test_dataset = CatBreedDataset(
            csv_file=str(base_dir / 'processed_data/test.csv'),
            transform=aug.get_val_transform()
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples:   {len(self.val_dataset)}")
        print(f"Num classes:   {len(self.train_dataset.breeds)}")
        print(f"\nBreeds: {', '.join(self.train_dataset.breeds)}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

    def setup_model(self):
        self.model = CatCNN(
            num_classes=len(self.train_dataset.breeds),
            dropout_rate=self.config['dropout_rate']
        )
        self.model = self.model.to(self.device)

        print(f"Model: 5-Layer CNN (from scratch)")
        print(f"Total parameters: {self.model.get_num_params():,}")
        print(f"Trainable parameters: {self.model.get_trainable_params():,}")

        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )

        print(f"Optimizer: {self.config['optimizer'].upper()}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Weight decay: {self.config['weight_decay']}")

        if self.config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
            print("Scheduler: ReduceLROnPlateau")
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
            print("Scheduler: CosineAnnealingLR")
        else:
            self.scheduler = None
            print("Scheduler: None")

        self.scaler = GradScaler() if self.config['use_amp'] else None
        if self.config['use_amp']:
            print("Mixed precision: Enabled (FP16)")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.config['use_amp']:
                with autocast():
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

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch}/{self.config["epochs"]}] '
                      f'Batch [{batch_idx + 1}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100. * correct / total:.2f}%')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'metrics': self.metrics
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (val_acc: {self.best_val_acc:.2f}%)")

    def train(self):
        start_time = time.time()

        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            val_loss, val_acc = self.validate()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)
            self.metrics['epoch_times'].append(epoch_time)
            self.metrics['learning_rates'].append(current_lr)

            

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, is_best=is_best)

            if self.scheduler:
                if self.config['scheduler'] == 'plateau':
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            if self.config['early_stopping'] and self.epochs_without_improvement >= self.config['patience']:
                print(f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")
                break

        total_time = time.time() - start_time
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best val accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")

        # Save final metrics
        self.save_metrics()

    def save_metrics(self):
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n aved metrics to {metrics_path}")

    def test(self):
        # Load best model
        best_checkpoint = torch.load(self.checkpoint_dir / 'best.pth')
        self.model.load_state_dict(best_checkpoint['model_state_dict'])

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")

        return test_acc


def main():
    # Configuration
    config = {
        # Model
        'image_size': 224,
        'dropout_rate': 0.5,

        # Training
        'batch_size': 64,
        'epochs': 150,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',  # 'adam', 'adamw', or 'sgd'
        'scheduler': 'plateau',  # 'plateau', 'cosine', or None
        'use_amp': True,  # Mixed precision training

        # Regularization
        'early_stopping': True,
        'patience': 15,

        # Data
        'num_workers': 4,

        # Output
        'output_dir': 'experiments/from_scratch_5layer'
    }

    for key, value in config.items():
        print(f"  {key:20s}: {value}")

    # Create trainer
    trainer = FromScratchTrainer(config)

    # Setup
    trainer.setup_data()
    trainer.setup_model()

    # Train
    trainer.train()

    # Test
    test_acc = trainer.test()



if __name__ == "__main__":
    main()
