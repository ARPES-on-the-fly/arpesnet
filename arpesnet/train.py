import torch
from torch import nn
import torchmetrics as tm

from arpesnet import transform as tr


def calc_loss(
    x: torch.FloatTensor,
    recons_x: torch.FloatTensor,
    loss_criterion: str = "mse",
    loss_weight: float = 1.0,
    device: str = None,
) -> torch.FloatTensor:
    """Calculate loss between two images

    Args:
        x (torch.Tensor): the input image
        recons_x (torch.Tensor): the target image
        loss_criterion (str): the loss criteria to use
            Available: 'mse', 'mae', 'psnr', 'ssim', 'ms_ssim 'pcs', 'ergas'
            Example: ['mse', 'ssim']
        loss_weight (float): the loss weight between [0-1] for each loss_criterion to use, but the
            total weight for all criteria should be 1. Example: [0.4, 0.6]

    Returns:
        torch.Tensor: the loss value
    """
    if device is None:
        device = x.device
    if recons_x.device != x.device:
        raise ValueError(f"Input and target devices do not match: {x.device} != {recons_x.device}")
    if x.shape != recons_x.shape:
        raise ValueError(f"Input and target shapes do not match: {x.shape} != {recons_x.shape}")
    if recons_x.dtype != x.dtype:
        raise TypeError(f"Input and target dtypes do not match: {x.dtype} != {recons_x.dtype}")

    if x.dim() == 3 and recons_x.dim() == 3:
        x = x.unsqueeze(1)
        recons_x = recons_x.unsqueeze(1)

    # Preparing the library to calculate loss
    mse_loss = nn.MSELoss().to(device)
    mae_loss = nn.L1Loss().to(device)
    psnr_loss = tm.PeakSignalNoiseRatio().to(device)
    ssim_loss = tm.StructuralSimilarityIndexMeasure().to(device)
    msssim_loss = tm.MultiScaleStructuralSimilarityIndexMeasure().to(device)
    pcs_loss = tm.CosineSimilarity().to(device)
    ergas_loss = tm.ErrorRelativeGlobalDimensionlessSynthesis().to(device)

    loss_total = 0
    # for crix, criy in zip(*[iter(criterion)]*2):
    for crix, criy in zip(loss_criterion, loss_weight):
        if crix == "mse":
            loss_total = loss_total + float(criy) * mse_loss(x, recons_x)
        elif crix == "mae":
            loss_total = loss_total + float(criy) * mae_loss(x, recons_x)
        elif crix == "psnr":
            loss_total = loss_total + float(criy) * (-psnr_loss(x, recons_x))
        elif crix == "ssim":
            loss_total = loss_total + float(criy) * (1 - ssim_loss(x, recons_x))
        elif crix == "ms_ssim":
            loss_total = loss_total + float(criy) * (1 - msssim_loss(x, recons_x))
        elif crix == "pcs":
            loss_total = loss_total + float(criy) * pcs_loss(x, recons_x)
        elif crix == "ergas":
            loss_total = loss_total + float(criy) * ergas_loss(x, recons_x)
        else:
            raise ValueError(
                f"Unknown loss criterion: {loss_criterion}. Use one or more of "
                f"mse, mae, psnr, ssim, ms_ssim, pcs, ergas."
            )
    return loss_total


def train_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim,
    dataloader: torch.utils.data.DataLoader,
    config: dict,
    augmentations: dict,
) -> float:
    """Train the model for one epoch

    Args:
        encoder (nn.Module): the encoder network
        decoder (nn.Module): the decoder network
        device (torch.device): the device to use for training
        optimizer (torch.optim): the optimizer to use
        dataloader (torch.utils.data.DataLoader): the dataloader for the training set
        config (dict): the configuration dictionary
        augmentations (dict): the augmentations dictionary
        iteration (int): the current iteration number
        scheduler (torch.optim.lr_scheduler, optional): the learning rate scheduler.
            Defaults to None.

    Returns:
        float: the average loss value for the epoch
    """
    device = next(encoder.parameters()).device

    encoder.train()
    decoder.train()
    train_loss = []

    for image_batch in dataloader:
        image_batch = image_batch.to(device)

        training_augmentations = augmentations.get("training_augmentations", None)
        if training_augmentations is not None:
            image_batch = training_augmentations(image_batch).to(device)

        noise_augmentations = augmentations.get("noise_augmentations", None)
        if noise_augmentations is not None:
            image_batch_noisy = noise_augmentations(image_batch).to(device)
        else:
            image_batch_noisy = image_batch

        # Encode data
        encoded_data = encoder(image_batch_noisy.unsqueeze(1))
        decoded_data = decoder(encoded_data).squeeze()

        if config["train"].get("denoiser", False) and len(noise_augmentations) > 0:
            x = image_batch
        else:
            x = image_batch_noisy
        loss = calc_loss(
            x=x,
            recons_x=decoded_data,
            loss_criterion=config["loss"]["criteria"],
            loss_weight=config["loss"]["weights"],
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return train_loss


def test_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: dict,
    augmentations: dict,
) -> float:
    """Test the model for one epoch

    Args:
        encoder (nn.Module): the encoder network
        decoder (nn.Module): the decoder network
        device (torch.device): the device to use for training
        dataloader (torch.utils.data.DataLoader): the dataloader for the test set
        loss_function (nn.Module): the loss function to use
        contractive (bool, optional): whether to use a contractive loss function. Defaults to False.

    Returns:
        float: the average loss value for the epoch
    """
    # Set evaluation mode for encoder and decoder
    device = next(encoder.parameters()).device
    encoder.eval()
    decoder.eval()
    test_loss = []

    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        for image_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)

            validation_augmentations = augmentations.get("validation_augmentations", None)
            if validation_augmentations is not None:
                image_batch = validation_augmentations(image_batch).to(device)

            noise_augmentations = augmentations.get("noise_augmentations", None)
            if noise_augmentations is not None:
                image_batch_noisy = noise_augmentations(image_batch).to(device)
            else:
                image_batch_noisy = image_batch

            # Encode data
            encoded_data = encoder(image_batch_noisy.unsqueeze(1))
            decoded_data = decoder(encoded_data).squeeze()

            if config["train"].get("denoiser", False) and len(noise_augmentations) > 0:
                x = image_batch
            else:
                x = image_batch_noisy
            loss = calc_loss(
                x=x,
                recons_x=decoded_data,
                loss_criterion=config["loss"]["criteria"],
                loss_weight=config["loss"]["weights"],
            )
            test_loss.append(loss.detach().cpu().numpy())

    return test_loss
