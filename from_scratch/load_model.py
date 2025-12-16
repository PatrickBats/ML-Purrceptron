import torch
from PIL import Image
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from shared.models import CatCNN
from shared.data_augmentation import CatBreedAugmentation


def load_model(checkpoint_path='experiments/from_scratch_5layer/checkpoints/best.pth'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

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

    model = CatCNN(num_classes=len(breed_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")

    return model, breed_names, checkpoint['config']


def predict_image(image_path, model, breed_names, device='cpu'):
    model = model.to(device)

    aug = CatBreedAugmentation(mode='from_scratch')
    transform = aug.get_val_transform()

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    predicted_idx = probs.argmax().item()
    predicted_breed = breed_names[predicted_idx]
    confidence = probs[predicted_idx].item()

    return predicted_breed, confidence, probs.cpu().numpy()


def predict_batch(image_paths, model, breed_names, device='cpu'):
    results = []

    for img_path in image_paths:
        breed, conf, _ = predict_image(img_path, model, breed_names, device)
        results.append((img_path, breed, conf))

    return results
