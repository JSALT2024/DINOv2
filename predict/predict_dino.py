import cv2
from torchvision import transforms
import torch
from torch import nn
import numpy as np
from typing import List, Union
from PIL import Image

transform_dino = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_dino_model(checkpoint_path: str = ""):
    # Load the original DINOv2 model with the correct architecture and parameters.
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)  # Removed map_location

    # Change shape of pos_embed
    pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    model.pos_embed = pos_embed

    # Load finetuned weights
    if checkpoint_path:
        pretrained = torch.load(checkpoint_path, map_location="cpu")
        # Make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained['teacher'].items():
            if 'dino_head' in key:
                print(key, 'not used')
            else:
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value
        # Load state dict
        model.load_state_dict(new_state_dict, strict=True)

    return model #.eval()


def normalize_image(image: np.ndarray, transform: transforms.Compose):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)
    return image


def dino_predict(
        images: Union[List[np.array], np.array],
        model: nn.Module,
        transform: transforms.Compose,
        device: torch.device,
) -> np.array:
    """
    image: image in numpy array in bgr format or list of images in bgr format.
    """
    if not isinstance(images, list):
        images = [images]

    images = [normalize_image(image, transform) for image in images]
    images = torch.stack(images).to(device)

    prediction = model(images)
    prediction = prediction.detach().cpu().numpy()

    return prediction
