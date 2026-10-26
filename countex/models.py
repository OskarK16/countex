import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# Resnet stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP stats
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

MODEL_NAMES = ("clip_linear", "resnet18", "resnet50")


class ClipLinear(nn.Module):
    """
        CLIP ViT-B/32 image encoder followed by a linear layer. (Linear probe)
    """

    def __init__(self, num_classes, normalize):
        super().__init__()

        clip_model = open_clip.create_model("ViT-B-32", pretrained="openai")
        self.visual = clip_model.visual
        self.normalize = normalize
        self.fc = nn.Linear(self.visual.output_dim, num_classes)

    def forward(self, x):
        z = self.visual(x)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return self.fc(z)


def build_classifier(name, num_classes, clip_normalise):
    """
    Build a classifier with a linear layer and the image transforms it expects.

    :param name: one of MODEL_NAMES
    :param num_classes: number of output classes
    :param clip_normalise: L2-normalise CLIP image embeddings before the linear layer
    :return: (model, train_transform, eval_transform)
    """

    if name in ("resnet18", "resnet50"):
        if name == "resnet18":
            model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1) # z: 512
        else:
            model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1) # z: 2048
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalise,
        ])
        eval_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalise,
        ])
    elif name == "clip_linear":
        model = ClipLinear(num_classes, normalize = clip_normalise)

        train_transforms = eval_transforms = transforms.Compose([
            transforms.Resize(224, interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    else:
        raise ValueError(f"Unknown model: {name}")

    return model, train_transforms, eval_transforms


def load_classifier(checkpoint_path, map_location="cpu"):
    """
    Rebuild a classifier saved by train_classifier.py.

    :param checkpoint_path: path to classifier.pt
    :param map_location: device to load the weights on
    :return: (model in eval mode, eval_transform, checkpoint dict)
    """

    checkpoint = torch.load(checkpoint_path, map_location = map_location, weights_only = False)
    model, _, eval_tf = build_classifier(checkpoint["model_name"], len(checkpoint["class_names"]), clip_normalise = checkpoint["clip_normalize"])
    if checkpoint["model"] is not None:
        model.load_state_dict(checkpoint["model"])
    else:
        model.fc.load_state_dict(checkpoint["fc"])

    return model.eval(), eval_tf, checkpoint
