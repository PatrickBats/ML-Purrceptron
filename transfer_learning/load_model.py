import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json


def load_model(checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent / 'experiments/resnet50_transfer/checkpoints/best.pth'

    print(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config['num_classes'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    base_dir = Path(__file__).parent.parent / 'data'
    import pandas as pd
    df = pd.read_csv(base_dir / 'processed_data/train.csv')
    breed_names = sorted(df['breed'].unique())


    return model, breed_names, config


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def predict_image(image_path, model, breed_names):
    device = next(model.parameters()).device
    transform = get_transforms()

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    confidence, predicted_idx = torch.max(probabilities, 0)
    predicted_breed = breed_names[predicted_idx.item()]

    all_probs = {breed: prob.item() for breed, prob in zip(breed_names, probabilities)}

    return predicted_breed, confidence.item(), all_probs


def predict_batch(image_paths, model, breed_names, batch_size=32):
    device = next(model.parameters()).device
    transform = get_transforms()
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        images = []
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            images.append(transform(image))

        batch_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        for j, path in enumerate(batch_paths):
            confidence, predicted_idx = torch.max(probabilities[j], 0)
            predicted_breed = breed_names[predicted_idx.item()]
            results.append((path, predicted_breed, confidence.item()))

    return results


def main():
    model, breed_names, config = load_model()

    print("\nAvailable breeds:")
    for i, breed in enumerate(breed_names):
        print(f"  {i}: {breed}")

    base_dir = Path(__file__).parent.parent / 'data'
    test_csv = base_dir / 'processed_data/test.csv'

    if test_csv.exists():
        import pandas as pd
        df = pd.read_csv(test_csv)

        sample = df.sample(1).iloc[0]
        image_path = base_dir / 'Datacleaning' / sample['dataset'] / 'images' / sample['breed'] / sample['filename']

        print(f"\nTesting on: {image_path}")
        print(f"True breed: {sample['breed']}")

        predicted_breed, confidence, all_probs = predict_image(str(image_path), model, breed_names)

        print(f"\nPrediction: {predicted_breed}")
        print(f"Confidence: {confidence:.2%}")

        print("\nTop 3 predictions:")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for breed, prob in sorted_probs:
            print(f"  {breed}: {prob:.2%}")


if __name__ == '__main__':
    main()
