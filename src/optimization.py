import torch
import torch.nn as nn
import torch.optim


def get_loss():
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True
    """

    # define the loss function
    loss = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss


def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
    parameters: list = [],
):
    """
    Returns an optimizer instance

    :param model: the model to optimize
    :param optimizer: one of 'SGD' or 'Adam'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    """
    parameters = model.parameters() if len(parameters) == 0 else parameters

    if optimizer.lower() == "sgd":

        opt = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

    elif optimizer.lower() == "adam":

        opt = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt