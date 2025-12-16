import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class CatBreedDataset(Dataset):

    def __init__(self, csv_file, transform=None, root_dir=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Auto-detect root_dir if not provided
        if root_dir is None:
            from pathlib import Path
            # CSV is in data/processed_data/, so root is 2 levels up
            csv_path = Path(csv_file).resolve()
            self.root_dir = csv_path.parent.parent.parent
        else:
            from pathlib import Path
            self.root_dir = Path(root_dir)

        # Create breed to index mapping
        self.breeds = sorted(self.data['breed'].unique())
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        self.idx_to_breed = {idx: breed for breed, idx in self.breed_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from pathlib import Path

        # Get image info
        row = self.data.iloc[idx]
        breed = row['breed']
        filename = row['filename']
        source = row['source']
        label = self.breed_to_idx[breed]

        # Construct path: root/data/Datacleaning/Source/images/breed/filename
        # Map source names to directory names
        source_map = {
            'kaggle': 'Kaggle',
            'oxfordiit': 'OxfordIIT'
        }
        source_dir = source_map.get(source, source)

        img_path = self.root_dir / 'data' / 'Datacleaning' / source_dir / 'images' / breed / filename

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_breed_name(self, idx):
        return self.idx_to_breed[idx]

    def get_breed_distribution(self):
        return self.data['breed'].value_counts().sort_index()


# Test dataset
if __name__ == "__main__":
    from shared.data_augmentation import CatBreedAugmentation
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("TESTING CAT BREED DATASET")
    print("=" * 60)

    # Check if processed data exists
    csv_path = '../data/processed_data/train.csv'
    if not os.path.exists(csv_path):
        print(f"\nERROR: {csv_path} not found!")
        print("Please run: python prepare_balanced_dataset.py")
        exit(1)

    # Create dataset
    print(f"\nLoading dataset from: {csv_path}")
    aug = CatBreedAugmentation(mode='from_scratch')
    dataset = CatBreedDataset(
        csv_file=csv_path,
        transform=aug.get_train_transform()
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of breeds: {len(dataset.breeds)}")
    print(f"\nBreeds: {dataset.breeds}")

    # Show breed distribution
    print("\nBreed distribution:")
    print(dataset.get_breed_distribution())

    # Test loading a sample
    print("\nTesting sample loading...")
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Label: {label} ({dataset.get_breed_name(label)})")
    print(f"Image min/max: {image.min():.3f} / {image.max():.3f}")

    # Test DataLoader
    print("\nTesting DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Set to 0 for testing, increase for training
        pin_memory=True
    )

    batch_images, batch_labels = next(iter(dataloader))
    print(f"Batch images shape: {batch_images.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch labels: {batch_labels}")

    print("\n" + "=" * 60)
    print("DATASET TEST COMPLETE!")
    print("=" * 60)
