import pandas as pd
import os
from pathlib import Path
import random
from collections import defaultdict

# Configuration
RANDOM_SEED = 42
SAMPLES_PER_BREED = 2500
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Overlapping breeds to keep
OVERLAPPING_BREEDS = [
    'Bengal',
    'Bombay',
    'British Shorthair',
    'Maine Coon',
    'Persian',
    'Ragdoll',
    'Russian Blue',
    'Siamese'
]

# Paths
BASE_DIR = Path(__file__).parent
OXFORD_CSV = BASE_DIR / 'Datacleaning/OxfordIIT/data/oxfordiit_cats.csv'
KAGGLE_CSV = BASE_DIR / 'Datacleaning/Kaggle/data/kaggle_cats.csv'
OXFORD_IMG_DIR = BASE_DIR / 'Datacleaning/OxfordIIT/images'
KAGGLE_IMG_DIR = BASE_DIR / 'Datacleaning/Kaggle/images'

OUTPUT_DIR = BASE_DIR / 'processed_data'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_filter_data():
    print("Loading datasets...")

    # Load OxfordIIT
    oxford_df = pd.read_csv(OXFORD_CSV)
    oxford_df['source'] = 'oxfordiit'
    oxford_df['full_path'] = oxford_df.apply(
        lambda row: str(OXFORD_IMG_DIR / row['breed'] / row['filename']),
        axis=1
    )

    kaggle_df = pd.read_csv(KAGGLE_CSV)
    kaggle_df['source'] = 'kaggle'
    kaggle_df['full_path'] = kaggle_df.apply(
        lambda row: str(KAGGLE_IMG_DIR / row['breed'] / row['filename']),
        axis=1
    )

    oxford_filtered = oxford_df[oxford_df['breed'].isin(OVERLAPPING_BREEDS)]
    kaggle_filtered = kaggle_df[kaggle_df['breed'].isin(OVERLAPPING_BREEDS)]


    combined_df = pd.concat([oxford_filtered, kaggle_filtered], ignore_index=True)

    print(f"\nTotal samples after filtering: {len(combined_df)}")
    print(f"Breeds: {sorted(combined_df['breed'].unique())}")
    print("\nSamples per breed BEFORE balancing:")
    print(combined_df['breed'].value_counts().sort_index())

    return combined_df


def balance_dataset(df):
    print(f"\nBalancing to {SAMPLES_PER_BREED} samples per breed...")

    random.seed(RANDOM_SEED)
    balanced_data = []

    for breed in OVERLAPPING_BREEDS:
        breed_samples = df[df['breed'] == breed].copy()
        current_count = len(breed_samples)

        if current_count >= SAMPLES_PER_BREED:
            
            sampled = breed_samples.sample(n=SAMPLES_PER_BREED, random_state=RANDOM_SEED)
            print(f"  {breed}: {current_count} -> {SAMPLES_PER_BREED} (undersampled)")
        else:
            sampled = breed_samples
            needed = SAMPLES_PER_BREED - current_count
            print(f"  {breed}: {current_count} -> {SAMPLES_PER_BREED} (need {needed} augmented)")

        balanced_data.append(sampled)

    balanced_df = pd.concat(balanced_data, ignore_index=True)

    print(f"\nTotal samples after balancing: {len(balanced_df)}")
    print("\nSamples per breed AFTER balancing:")
    print(balanced_df['breed'].value_counts().sort_index())

    return balanced_df


def split_dataset(df):
    print(f"\nSplitting dataset: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")

    random.seed(RANDOM_SEED)
    train_data = []
    val_data = []
    test_data = []

    for breed in OVERLAPPING_BREEDS:
        breed_samples = df[df['breed'] == breed].copy()
        breed_samples = breed_samples.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        n = len(breed_samples)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        train_data.append(breed_samples[:train_end])
        val_data.append(breed_samples[train_end:val_end])
        test_data.append(breed_samples[val_end:])

    train_df = pd.concat(train_data, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    val_df = pd.concat(val_data, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    test_df = pd.concat(test_data, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)

    print(f"\nTrain: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")

    print("\nTrain set distribution:")
    print(train_df['breed'].value_counts().sort_index())

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    print(f"\nSaving splits to {OUTPUT_DIR}...")

    train_df.to_csv(OUTPUT_DIR / 'train.csv', index=False)
    val_df.to_csv(OUTPUT_DIR / 'val.csv', index=False)
    test_df.to_csv(OUTPUT_DIR / 'test.csv', index=False)

    with open(OUTPUT_DIR / 'dataset_info.txt', 'w') as f:
        f.write("Balanced Cat Breed Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Breeds: {len(OVERLAPPING_BREEDS)}\n")
        f.write(f"Target samples per breed: {SAMPLES_PER_BREED}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")

        f.write("Split Ratios:\n")
        f.write(f"  Train: {TRAIN_RATIO:.0%}\n")
        f.write(f"  Val:   {VAL_RATIO:.0%}\n")
        f.write(f"  Test:  {TEST_RATIO:.0%}\n\n")

        f.write("Actual Counts:\n")
        f.write(f"  Train: {len(train_df)}\n")
        f.write(f"  Val:   {len(val_df)}\n")
        f.write(f"  Test:  {len(test_df)}\n")
        f.write(f"  Total: {len(train_df) + len(val_df) + len(test_df)}\n\n")

        f.write("Breeds:\n")
        for breed in sorted(OVERLAPPING_BREEDS):
            f.write(f"  - {breed}\n")

    print("\nFiles created:")
    print(f"  - train.csv ({len(train_df)} samples)")
    print(f"  - val.csv ({len(val_df)} samples)")
    print(f"  - test.csv ({len(test_df)} samples)")
    print(f"  - dataset_info.txt")


def main():


    df = load_and_filter_data()

    balanced_df = balance_dataset(df)

    train_df, val_df, test_df = split_dataset(balanced_df)

    # Save
    save_splits(train_df, val_df, test_df)



if __name__ == "__main__":
    main()
