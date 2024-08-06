def lr_layer_multiplier(model, lr=0.001, lr_mult=0.9, verbose=False):
    """Consecutively multiplies the learning rate(lr) of previous named layers or named blocks
    of layers in model parameters.

    Arguments
    ---------
    model: PyTorch model, model layer's lr will be adjusted
    lr: float, lr of last layer of model
    lr_multi: float, multiply lr of deeper layers/block of layers by this number
    """

    # save layer names
    layer_names = []
    for idx, (name, _) in enumerate(model.named_parameters()):
        layer_names.append(name)

    # reverse layers
    layer_names.reverse()

    # placeholder
    parameters = []
    prev_group_name = layer_names[-1].split("features.")[1].split(".")[0]

    # store params & learning rates
    for idx, name in enumerate(layer_names):

        try:

            # parameter group name
            cur_group_name = name.split("features.")[1].split(".")[0]

            # update learning rate
            if cur_group_name != prev_group_name:
                lr *= lr_mult
            prev_group_name = cur_group_name

            # display info
            if verbose:
                print(f"{idx}: lr = {lr:.6f}, {name}")

            # append layer parameters
            parameters += [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if n == name and p.requires_grad
                    ],
                    "lr": lr,
                }
            ]

        # exception will taken into account layers without string 'features.' in their name
        except:
            parameters += [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if n == name and p.requires_grad
                    ],
                    "lr": lr,
                }
            ]

            # display info
            if verbose:
                print(f"{idx}: lr = {lr:.6f}, {name}")

    return parameters
