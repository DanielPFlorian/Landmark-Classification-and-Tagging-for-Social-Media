import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=50):

    # Get the requested architecture
    if hasattr(models, model_name):

        model_transfer = getattr(models, model_name)(pretrained=True)

    else:

        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Add the linear layer at the end with the appropriate number of classes

    # if efficientnet_b4 is chosen, first load unpretrained weights and then load pretrained weights via state dict
    # in order to get useable layer names to replace model classifier
    if model_name=="efficientnet_b4":
        model_transfer_not_pretrained = getattr(models, model_name)(pretrained=False)
        model_transfer_not_pretrained.load_state_dict(
            torch.load("checkpoints/efficientnet_b4_rwightman-7eb33cd5.pth")
        )

        model_transfer = model_transfer_not_pretrained

        for param in model_transfer.parameters():
            param.requires_grad = False

        num_ftrs = model_transfer.classifier[1].in_features
        model_transfer.classifier = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, num_ftrs * 2),
            nn.BatchNorm1d(num_ftrs * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_ftrs * 2, n_classes),
        )

    else:
        num_ftrs = model_transfer.fc.in_features
        model_transfer.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, num_ftrs * 2),
            nn.BatchNorm1d(num_ftrs * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_ftrs * 2, n_classes),
        )

    return model_transfer