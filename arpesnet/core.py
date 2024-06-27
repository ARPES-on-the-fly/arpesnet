from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchmetrics as tm
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from tqdm.auto import trange

import arpesnet as an
from arpesnet.utils import load_config, load_torch_datasets, sec2min


def load_trainer(filepath: str | Path) -> ModelTrainer:
    """
    Load a ModelTrainer object from a saved file.

    Args:
        filepath (str | Path): Path to the saved file.

    Returns:
        ModelTrainer: Loaded ModelTrainer object.
    """
    loaded = torch.load(filepath)
    trainer = ModelTrainer(loaded["config"])
    trainer._init_model()
    trainer._init_optimizer()
    trainer.encoder.load_state_dict(loaded.pop("encoder_state_dict"))
    trainer.decoder.load_state_dict(loaded.pop("decoder_state_dict"))
    trainer.optimizer.load_state_dict(loaded.pop("optimizer_state_dict"))
    for k, v in loaded.items():
        setattr(trainer, k, v)
    return trainer


class ModelTrainer:
    def __init__(
        self,
        config: dict | str | Path,
        verbose="compact",
        train_dataset: torch.tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize a ModelTrainer object.

        Args:
            config (dict | str | Path): Configuration for the ModelTrainer.
            verbose (str, optional): Verbosity level. Defaults to "compact".
            train_dataset (torch.tensor | None, optional): Training dataset. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.config = load_config(config)
        self.generator = torch.manual_seed(self.config["device"]["seed"])
        self.device = (
            torch.device("cuda")
            if (torch.cuda.is_available() and self.config["device"]["use_gpu"])
            else torch.device("cpu")
        )
        self.verbose = verbose

        self.model = None
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.scheduler = None

        self.train_losses = []
        self.train_losses_min = []
        self.train_losses_max = []
        self.train_losses_mean = []
        self.train_losses_std = []
        self.validation_losses = []
        self.validation_losses_min = []
        self.validation_losses_max = []
        self.validation_losses_mean = []
        self.validation_losses_std = []
        self.times = [time.time()]
        self.total_time = 0
        self.last_epoch = 0
        self.checkpoints = []

        self.n_epochs = config["train"]["n_epochs"]
        self.copy_to_cuda = config["device"].get("copy_to_cuda", False)
        self.transforms = {
            "preprocessing": [],
            "training_augmentations": [],
            "validation_augmentations": [],
            "noise_augmentations": [],
        }
        self.train_dataset = train_dataset
        self.train_data = None
        self.validation_data = None
        self.train_data_preprocessed = None
        self.validation_data_preprocessed = None
        self.train_loader = None
        self.validation_loader = None

        self._init_model()
        self._init_optimizer()
        self._init_transforms()
        # self._init_data()

    @property
    def results(self) -> dict:
        """
        Get the results of the ModelTrainer.

        Returns:
            dict: Results of the ModelTrainer.
        """
        return {
            "model": self.model,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "test_processing": self.transforms["validation_augmentations"],
            # "encoder_state_dict": self.encoder.state_dict(),
            # "decoder_state_dict": self.decoder.state_dict(),
            # "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "train_losses_mean": self.train_losses_mean,
            "train_losses_min": self.train_losses_min,
            "train_losses_max": self.train_losses_max,
            "validation_losses": self.validation_losses,
            "validation_losses_mean": self.validation_losses_mean,
            "validation_losses_min": self.validation_losses_min,
            "validation_losses_max": self.validation_losses_max,
            "config": self.config,
            "times": self.times,
            "total_time": self.total_time,
            "last_epoch": self.last_epoch,
        }

    def get_save_name(self, directory) -> str:
        """
        Get the save name for the ModelTrainer.

        Args:
            directory: Directory to save the ModelTrainer.

        Returns:
            str: Save name for the ModelTrainer.
        """
        if directory is None:
            directory = self.config["paths"]["results"]
        directory = Path(directory)
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"Created directory {directory}")
        save_name = f"{self.model}_e{self.last_epoch}"
        save_path = directory / save_name
        i = 1
        while save_path.with_suffix(".pth").exists():
            save_path = directory / f"{save_name}_{i:03d}"
            i += 1
            if i > 1000:  # just in case
                raise FileExistsError(f"File {save_path} exists")
        return save_path

    def save(
        self,
        directory: str | Path | None = None,
        compact_loss=True,
    ) -> None:
        """
        Save the ModelTrainer.

        Args:
            directory (str | Path | None, optional): Directory to save the ModelTrainer.
                Defaults to None.
            compact_loss (bool, optional): Whether to the full history of the loss, or only the
                mean. Defaults to True.
        """
        directory = Path(directory) if directory is not None else None

        save_path = self.get_save_name(directory)
        if save_path.with_suffix(".pth").exists():
            print(f"WARNING: File {save_path} exists")
        res = {
            "model": self.model,
            "test_processing": self.transforms["validation_augmentations"],
            "train_losses_mean": self.train_losses_mean,
            "train_losses_min": self.train_losses_min,
            "train_losses_max": self.train_losses_max,
            "validation_losses_mean": self.validation_losses_mean,
            "validation_losses_min": self.validation_losses_min,
            "validation_losses_max": self.validation_losses_max,
            "config": self.config,
            "times": self.times,
            "total_time": self.total_time,
            "last_epoch": self.last_epoch,
            "n_epochs": self.n_epochs,
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "checkpoints": self.checkpoints + [save_path.stem],
        }
        if not compact_loss:
            res["train_losses"] = self.train_losses
            res["validation_losses"] = self.validation_losses

        torch.save(res, save_path.with_suffix(".pth"))
        if self.verbose:
            print(f"Model saved as {save_path.with_suffix('.pth')}")

    @staticmethod
    def load(filepath: str | Path) -> ModelTrainer:
        """
        Load a ModelTrainer object from a saved file.

        Args:
            filepath (str | Path): Path to the saved file.

        Returns:
            ModelTrainer: Loaded ModelTrainer object.
        """
        loaded = torch.load(filepath)
        trainer = ModelTrainer(loaded["config"])
        trainer._init_model()
        trainer._init_optimizer()
        trainer.encoder.load_state_dict(loaded.pop("encoder_state_dict"))
        trainer.decoder.load_state_dict(loaded.pop("decoder_state_dict"))
        trainer.optimizer.load_state_dict(loaded.pop("optimizer_state_dict"))
        for k, v in loaded.items():
            setattr(trainer, k, v)
        return trainer

    def _init_transforms(self) -> None:
        """Parse the configuration file and set the parameters for the preprocessing and the
        augmentation
        """
        for target in self.transforms.keys():
            if target in self.config and self.config[target] is not None:
                for k, v in self.config[target].items():
                    args = []
                    kwargs = {}
                    if isinstance(v, dict):
                        kwargs = v
                    elif isinstance(v, list):
                        args = v
                    elif v is None:
                        pass
                    else:
                        args = [v]
                    try:
                        if hasattr(an.transform, k):
                            self.transforms[target].append(
                                getattr(an.transform, k)(*args, **kwargs)
                            )
                        else:
                            self.transforms[target].append(
                                getattr(v2, k)(*args, **kwargs)
                            )
                    except AttributeError:
                        if self.verbose:
                            print(f"Unknown {target}: {k}")
                self.transforms[target] = v2.Compose(self.transforms[target])

    def _init_model(self) -> None:
        """Load the model from the configuration file"""
        self.model = "ARPESNet"
        self.encoder = an.model.Encoder(**self.config["model"]["kwargs"]).to(
            self.device
        )
        self.decoder = an.model.Decoder(**self.config["model"]["kwargs"]).to(
            self.device
        )

    def describe(self) -> str:
        """
        Describe the ModelTrainer.
        """
        self.describe_data()
        self.describe_optimizer()
        self.describe_model()

    def describe_model(self) -> str:
        """
        Describe the model used by the ModelTrainer.
        """
        if self.train_data is None:
            self._init_data()
        print(f"MODEL: {self.model}")
        n_enc_params = sum(p.numel() for p in self.encoder.parameters())
        n_dec_params = sum(p.numel() for p in self.decoder.parameters())
        print(
            f"Parameters: {n_enc_params:,.0f} encoder | {n_dec_params:,.0f} decoder | total: {n_enc_params+n_dec_params:,.0f}"
        )
        print(self.encoder)
        print(self.decoder)
        orig_img = self.train_data[0]
        orig_size = orig_img.nelement() * 32
        print(
            f"input  : {orig_img.shape} {orig_img.dtype}\t"
            f"| size: {orig_img.nelement():,.0f}\t| {orig_size/1024:,.2f} KB"
        )
        t0 = time.time()
        encoded = self.encoder(orig_img.unsqueeze(0).to(self.device))
        enc_time = time.time() - t0
        enc_size = encoded.nelement() * 32
        print(
            f"encoded: {encoded.shape} {encoded.dtype}\t"
            f"| size: {encoded.nelement():,.0f}\t| {enc_size/1024:,.2f} KB"
        )
        t0 = time.time()
        decoded = self.decoder(encoded)
        dec_time = time.time() - t0
        dec_size = decoded.nelement() * 32

        print(
            f"decoded: {decoded.shape} {decoded.dtype}\t"
            f"| size: {decoded.nelement():,.0f}\t| {dec_size/1024:,.2f} KB"
        )
        print(
            f"compression ratio: {orig_size/enc_size:.2f} | bpp: {enc_size/orig_size:.4f}"
        )
        print(
            f"Encoding time: {enc_time:.3f} s | Decoding time: {dec_time:.3f} s |"
            f" Total time: {enc_time+dec_time:.3f} s"
        )
        print(
            f"original size: {orig_size/1024:.2f} KB | encoded size: {enc_size/1024:.2f} KB | "
            f"decoded size: {dec_size/1024:.2f} KB"
        )

    def _init_optimizer(self) -> None:
        """Initialize the optimizer"""
        self.optimizer = getattr(torch.optim, self.config["optimizer"]["name"])(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            **self.config["optimizer"]["kwargs"],
        )
        self.last_lr = self.optimizer.param_groups[0]["lr"]
        scheduler_dict = self.config.get("scheduler", None)
        if scheduler_dict is not None:
            self.scheduler = getattr(torch.optim.lr_scheduler, scheduler_dict["name"])(
                self.optimizer, **scheduler_dict["kwargs"]
            )

    def describe_optimizer(self) -> None:
        print(f"OPTIMIZER: {self.config['optimizer']['name']}")
        for k, v in self.config["optimizer"]["kwargs"].items():
            print(f"\t{k}: {v}")
        if self.scheduler is not None:
            print(f"SCHEDULER: {self.scheduler}")
            for k, v in self.config["scheduler"]["kwargs"].items():
                print(f"\t{k}: {v}")

    def _init_data(self) -> None:
        """Load the data and split the dataset"""

        if self.train_dataset is None:
            if self.verbose:
                print(f"loading Training dataset: {self.config['paths']['train_data']}")
            self.train_dataset = load_torch_datasets(
                self.config["paths"]["train_data"], "train", self.config
            )
            self.train_dataset = an.transform.preprocess(
                self.train_dataset, self.transforms["preprocessing"], pbar=self.verbose
            )
        if self.copy_to_cuda:
            self.train_dataset = self.train_dataset.to(self.device)

        n_samples = len(self.train_dataset)
        split_sizes = [
            int(n_samples * fr) for fr in self.config["train"]["split_ratio"]
        ]
        split_sizes[0] += n_samples - sum(split_sizes)

        self.train_data, self.validation_data = random_split(
            self.train_dataset, split_sizes, generator=self.generator
        )

        pin_memory = (str(self.device) == "cuda") and not self.copy_to_cuda
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.config["train"]["batch_size"],
            pin_memory=pin_memory,
            shuffle=self.config["train"].get("shuffle", False),
            drop_last=self.config["train"].get("drop_last", False),
        )
        self.validation_loader = DataLoader(
            self.validation_data,
            batch_size=self.config["train"]["batch_size"],
            pin_memory=pin_memory,
            shuffle=self.config["train"].get("shuffle", False),
            drop_last=self.config["train"].get("drop_last", False),
        )
        # if self.verbose:
        #     self.describe_data()

    def describe_data(self) -> None:
        if self.train_data is None:
            self._init_data()
        n_samples = len(self.train_dataset)
        split_sizes = [len(self.train_data), len(self.validation_data)]
        print(
            f"DATA: {n_samples} samples | shape {self.train_dataset.shape[1:]}",
            f" | dtype {self.train_dataset.dtype}",
        )
        print(
            f"\tTraining:   {split_sizes[0]:4.0f} ({split_sizes[0] / n_samples:.1%}) | ",
            end="",
        )
        print(f"Validation: {split_sizes[1]:4.0f} ({split_sizes[1] / n_samples:.1%})")

    def train(
        self,
        n_epochs: int | None = None,
        milestones: list[int] | None = None,
        milestone_every: int | None = None,
        save_dir: str | Path | None = None,
        plot: bool = True,
        save: bool = True,
        test_imgs: torch.tensor | None = None,
        pbar=True,
    ) -> None:
        """Train the model using the training and validation datasets

        Args:
            n_epochs (int): Number of epochs to train the model. If None, the default number of
                epochs will be used.
            milestones (list[int]): List of epochs at which to perform certain actions, such as
                saving the model or plotting the results. If None, no milestones will be used.
            milestone_every (int): Interval between milestones. If None, no interval will be used.
            save_dir (str | Path): Directory to save the model. If None, the model will not be
                saved.
            plot (bool): Whether to plot the results during training. Default is True.
            save (bool): Whether to save the model during training. Default is True.
            test_imgs (torch.tensor): Test images to use for plotting the results. Required if plot
                is True and milestones or milestone_every is provided.
            pbar (bool): Whether to display a progress bar during training. Default is True.
        """
        if milestone_every is not None or milestones is not None:
            if plot and test_imgs is None:
                raise ValueError("test_imgs must be provided to plot the results")
        else:
            milestones = []

        if self.train_data is None:
            self._init_data()

        self.training_start_datetime = time.strftime("%Y%m%d-%H%M%S")
        if n_epochs is not None:
            self.n_epochs = n_epochs
        max_epochs = self.n_epochs + self.last_epoch

        start_time = time.time()
        for epoch in trange(
            self.last_epoch,
            max_epochs,
            desc="Training",
            disable=(self.verbose is None or not pbar),
        ):
            self.last_epoch = epoch + 1

            try:
                t0 = time.time()
                if self.verbose == "full":
                    print(f"Epoch {epoch + 1}/{max_epochs}", end="")
                train_loss = an.train.train_epoch(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    optimizer=self.optimizer,
                    dataloader=self.train_loader,
                    config=self.config,
                    augmentations=self.transforms,
                )

                validation_loss = an.train.test_epoch(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    dataloader=self.validation_loader,
                    config=self.config,
                    augmentations=self.transforms,
                )
                tl = torch.mean(train_loss)
                tl_lower = torch.quantile(train_loss, 0.2)
                tl_upper = torch.quantile(train_loss, 0.8)
                vl = torch.mean(validation_loss)
                vl_lower = torch.quantile(validation_loss, 0.2)
                vl_upper = torch.quantile(validation_loss, 0.8)

                self.train_losses.append(train_loss)
                self.train_losses_min.append(tl_lower)
                self.train_losses_max.append(tl_upper)
                self.train_losses_mean.append(tl)
                self.validation_losses.append(validation_loss)
                self.validation_losses_min.append(vl_lower)
                self.validation_losses_max.append(vl_upper)
                self.validation_losses_mean.append(vl)

                self.times.append(time.time() - t0)
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / (epoch + 1)
                expected_time = avg_time * self.n_epochs
                remaining_time = expected_time - elapsed_time

                run_time = time.time() - t0
                prev_train_loss = (
                    self.train_losses_mean[-2]
                    if len(self.train_losses_mean) > 1
                    else tl
                )
                prev_validation_loss = (
                    self.validation_losses_mean[-2]
                    if len(self.validation_losses_mean) > 1
                    else vl
                )
                tlc = (tl - prev_train_loss) / prev_train_loss
                vlc = (vl - prev_validation_loss) / prev_validation_loss

                print_str = f" \t| tloss: {tl:.3e} ({100*tlc:+.2f}%)"
                print_str += f" \t| vloss: {vl:.3e} ({100*vlc:+.2f}%)"
                print_str += f" \t| {sec2min(elapsed_time)} < {sec2min(remaining_time)}"
                print_str += f", {run_time:.2f} s/epoch"
                if self.verbose == "compact":
                    print(f"Epoch {epoch + 1}/{max_epochs}{print_str}", end="\r")
                elif self.verbose == "full":
                    print(print_str)

                if self.scheduler is not None:
                    self.scheduler.step()
                    new_lr = self.scheduler.get_last_lr()
                    if self.verbose and new_lr != self.last_lr:
                        print(f"changed Learning rate: {self.last_lr} -> {new_lr}")
                    self.last_lr = new_lr

                if epoch + 1 in milestones or (
                    milestone_every is not None and (epoch + 1) % milestone_every == 0
                ):
                    if save:
                        self.save(save_dir)
                    if plot and test_imgs is not None:
                        savename = self.get_save_name(save_dir)
                        self.plot_loss_and_reconstruction(
                            test_imgs=test_imgs, savename=savename
                        )
                        plt.pause(0.05)
            except KeyboardInterrupt:
                if self.verbose:
                    print(
                        f"INTERRUPTED at epoch {epoch + 1}/{max_epochs} "
                        f"after {sec2min(elapsed_time)}"
                    )
                break
        if self.verbose == "compact":
            print(f"Epoch {epoch + 1}/{max_epochs}{print_str}")
        if self.verbose:
            print(f"Training finished after {sec2min(elapsed_time)}")

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input image using the encoder model.

        Args:
            img (torch.Tensor): The input image to be encoded.

        Returns:
            torch.Tensor: The encoded representation of the input image.
        """
        if img.shape[-2:] != torch.Size(self.encoder.input_shape):
            raise ValueError(
                f"Input shape for this encoder must be {self.encoder.input_shape}"
                f", got {img.shape[-2:]}"
            )
        if len(img.shape) == 3:
            img = img.unsqueeze(1)
        elif len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif len(img.shape) != 4:
            raise ValueError(
                f"Input shape for this encoder must be 3D or 4D, got {img.shape}"
            )
        return self.encoder(img.to(self.device))  # .cpu().detach().squeeze()

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Decodes the encoded tensor using the decoder model.

        Args:
            encoded (torch.Tensor): The encoded tensor to be decoded.

        Returns:
            torch.Tensor: The decoded tensor.

        """
        if len(encoded.shape) == 1:
            encoded = encoded.unsqueeze(0)
        elif len(encoded.shape) == 2:
            encoded = encoded.unsqueeze(1)
        return self.decoder(encoded.to(self.device))  # .cpu().detach().squeeze()

    def eval(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """Encode and decode an image using the trained model.

        Args:
            img (torch.FloatTensor): The image to evaluate.

        Returns:
            torch.FloatTensor: The reconstructed image.
        """
        if len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif len(img.shape) == 3:
            img = img.unsqueeze(0)
        enc = self.encoder(img.to(self.device))
        if len(enc.shape) == 1:
            enc = enc.unsqueeze(0)
        return self.decode(enc).squeeze().detach().cpu()

    def plot_losses(self, ax=None, logx=False, logy=False, **kwargs) -> None:
        """
        Plots the training and validation losses over epochs.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot the losses. If not provided, the current axes will be used.
            logx (bool, optional): Whether to use a logarithmic scale for the x-axis. Defaults to False.
            logy (bool, optional): Whether to use a logarithmic scale for the y-axis. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            None
        """
        if len(self.train_losses_mean) == 0:
            print("No training data available")
            return
        if ax is None:
            ax = plt.gca()

        x_train = (
            torch.arange(len(self.train_losses_mean)) - 0.5
        )  # training data is half a step behind
        x_val = torch.arange(len(self.train_losses_mean))
        # see https://twitter.com/aureliengeron/status/1110839223878184960 for explanation

        ax.plot(x_train, self.train_losses_mean, label="Training loss", **kwargs)
        ax.plot(x_val, self.validation_losses_mean, label="Validation loss", **kwargs)
        try:
            tupper = torch.Tensor(self.train_losses_max)
            tlower = torch.Tensor(self.train_losses_min)
            ax.fill_between(x_train, tlower, tupper, alpha=0.2)

            vupper = torch.Tensor(self.validation_losses_max)
            vlower = torch.Tensor(self.validation_losses_min)
            ax.fill_between(x_val, vlower, vupper, alpha=0.2)
        except ValueError:
            pass

        # remove outliers
        train_losses_mean = torch.tensor(self.train_losses_mean)
        valid_losses_mean = torch.tensor(self.validation_losses_mean)
        clean_train_loss = torch.clamp(
            train_losses_mean,
            0,
            torch.mean(train_losses_mean + 3 * torch.std(train_losses_mean)),
        )
        clean_valid_loss = torch.clamp(
            valid_losses_mean,
            0,
            torch.mean(valid_losses_mean + 3 * torch.std(valid_losses_mean)),
        )

        ax.set_ylim(0, torch.max(torch.cat([clean_train_loss, clean_valid_loss])) * 1.1)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Loss {self.config['loss']['criteria']}")
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.legend()
        # plt.pause(0.05)

        # plt.show()

    def plot_loss_and_reconstruction(
        self, test_imgs: torch.tensor, savename=None
    ) -> None:
        """
        Plots the loss and reconstruction images for the given test images.

        Args:
            test_imgs (torch.tensor): The test images to evaluate and plot.
            savename (str, optional): The name of the file to save the figure as. Defaults to None.

        Returns:
            None
        """
        n_imgs = len(test_imgs)
        plt.ion()
        grid = [
            ["loss", "loss", "loss"] + [f"original_{i}" for i in range(n_imgs)],
            ["loss", "loss", "loss"] + [f"rec_{i}" for i in range(n_imgs)],
            ["loss", "loss", "loss"] + [f"diff_{i}" for i in range(n_imgs)],
        ]
        fig, axes = plt.subplot_mosaic(
            grid, figsize=(6 + len(test_imgs), 4), constrained_layout=True
        )
        fig.suptitle(f"Model: {self.model} | {self.last_epoch} epochs")
        for name, ax in axes.items():
            if name != "loss":
                ax.axis("off")
        self.plot_losses(ax=axes["loss"])
        for i, img in enumerate(test_imgs):
            rec = self.eval(img)
            img = img.detach().squeeze().cpu()
            clim = img.min(), img.max()
            diff = img - rec
            vmax = clim[1]
            axes[f"original_{i}"].imshow(img.numpy(), cmap="viridis", clim=clim, origin="lower")
            axes[f"rec_{i}"].imshow(rec.numpy(), cmap="viridis", clim=clim, origin="lower")
            axes[f"diff_{i}"].imshow(
                diff.numpy(), cmap="bwr", clim=(-vmax, vmax), origin="lower"
            )
        if savename is not None:
            fig.savefig(Path(savename).with_suffix(".png"), dpi=300)
            print(f"saved figure as {savename}.png")

    def test_model(
        self,
        test_data: torch.Tensor,
        mean=False,
        pbar=False,
        metrics="all",
        ext_model=None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Test the model using the provided test data.

        Args:
            test_data (torch.Tensor): The test data to evaluate the model on.
            mean (bool, optional): Whether to calculate the mean of the evaluation metrics.
                Defaults to False.
            pbar (bool, optional): Whether to display a progress bar during preprocessing.
                Defaults to False.
            metrics (str or list, optional): The evaluation metrics to calculate.
                Must be one of "mse", "psnr", "ssim", "ms_ssim" or "all".
                Defaults to "all", which includes all available metrics.
            ext_model (object, optional): An external model to use for reconstruction.
                Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the evaluation metrics.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results for each image in the test
                data.

        Raises:
            None

        Examples:
            # Test the model using all available evaluation metrics
            results = model.test_model(test_data)

            # Test the model using specific evaluation metrics and an external model for
            # reconstruction
            results = model.test_model(test_data, metrics=["mse", "psnr"], ext_model=external_model)
            This assumes that the external model has a `forward` method that takes an input image
            and returns the reconstructed image.
        """

        if isinstance(metrics, str) and metrics == "all":
            metrics = ["mse", "psnr", "ssim", "ms_ssim"]
        elif isinstance(metrics, str):
            metrics = [metrics]
        mfunc = []
        for m in metrics:
            if m.upper() == "MSE":
                mfunc.append(nn.MSELoss())
            elif m.upper() == "PSNR":
                mfunc.append(tm.PeakSignalNoiseRatio())
            elif m.upper() == "SSIM":
                mfunc.append(tm.StructuralSimilarityIndexMeasure())
            elif m.upper() == "MS_SSIM":
                mfunc.append(tm.MultiScaleStructuralSimilarityIndexMeasure())
            else:
                raise ValueError(f"Unknown metric: {m}")

        processed_test_data = torch.stack(
            [self.transforms["preprocessing"](img) for img in test_data]
        ).cpu()

        if ext_model is not None:
            reconstructed = (
                torch.stack([ext_model(img) for img in processed_test_data])
                .detach()
                .cpu()
            )
        else:
            reconstructed = (
                torch.stack(
                    [self.decode(self.encode(img)) for img in processed_test_data]
                )
                .detach()
                .cpu()
            )
        df = []
        for img, rec in zip(processed_test_data, reconstructed):
            vals = []
            for metric in mfunc:
                res = (
                    metric(rec.squeeze(), img.squeeze(), **kwargs)
                    .to(torch.float32)
                    .numpy()
                )
                vals.append(res)
            df.append(pd.Series(vals, index=metrics, dtype=float))
        out = pd.DataFrame(df, columns=metrics)
        if mean:
            out = out.mean()
        return out
