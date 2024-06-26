import sys
import time
from pathlib import Path
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torchvision.transforms import v2

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import arpesnet as an

warnings.filterwarnings("ignore")

print("Python", sys.version)
print(f"Pytorch version: {torch.__version__} | CUDA enabled = {torch.cuda.is_available()}")

ROOT_DIR = Path(r"D:\data\ARPESdatabase\ARPESNet")
TRAINING_DIR = ROOT_DIR / "train_data"
TEST_DIR = ROOT_DIR / "test_data"
TEST_IMGS_FILE = ROOT_DIR / "test_imgs.pt"

INPUT_SHAPE = (256, 256)
NORM_RANGE = (0, 100)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
SPLIT_RATIO = (0.8, 0.2)
COPY_TO_CUDA = True

N_EPOCHS = 10
DENOISER = False
SAVE_EVERY = 100
SAVE_AT_EPOCH = [10, 50]


# set the seed for reproducibility

generator = torch.manual_seed(42)

# define the transformations
preprocess = an.transform.Compose(
    [an.transform.Resize(INPUT_SHAPE), an.transform.NormalizeMinMax(*NORM_RANGE)]
)


# load the training dataset
assert TRAINING_DIR.exists(), f"{TRAINING_DIR} does not exist"
train_dataset = torch.stack(
    [
        preprocess(torch.load(f))
        for f in tqdm(
            TRAINING_DIR.glob("*.pt"),
            desc="Loading training dataset",
            total=len(list(TRAINING_DIR.glob("*.pt"))),
        )
    ]
).view(-1, *INPUT_SHAPE)

# load the test dataset
# test_dataset = torch.stack([preprocess(torch.load(f)) for f in tqdm(TRAINING_DIR.glob("*.pt"))])

test_imgs = torch.load(TEST_IMGS_FILE)
test_imgs = torch.stack([preprocess(img) for img in tqdm(test_imgs)])

if COPY_TO_CUDA:
    train_dataset = train_dataset.to(DEVICE)
    # test_dataset = test_dataset.to(DEVICE)
    test_imgs = test_imgs.to(DEVICE)

print(f"Train dataset shape: {train_dataset.shape}")
# print(f"Test dataset shape: {test_dataset.shape}")
print(f"Test images shape: {test_imgs.shape}")

# split the training dataset
n_samples = len(train_dataset)
split_sizes = [int(n_samples * fr) for fr in SPLIT_RATIO]
split_sizes[0] += n_samples - sum(split_sizes)
train_split, val_split = random_split(train_dataset, split_sizes, generator=generator)
pin_memory = (DEVICE == "cuda") and not COPY_TO_CUDA

training_loader = DataLoader(
    train_split,
    batch_size=BATCH_SIZE,
    pin_memory=pin_memory,
    shuffle=True,
    drop_last=True,
)
validation_loader = DataLoader(
    val_split,
    batch_size=BATCH_SIZE,
    pin_memory=pin_memory,
    shuffle=True,
    drop_last=True,
)


# define the model

encoder = an.model.Encoder(
    kernel_size=11,
    kernel_decay=2,
    n_layers=1,
    start_channels=4,
    max_channels=32,
    n_blocks=6,
    input_shape=INPUT_SHAPE,
    relu=nn.PReLU,
    relu_kwargs=dict(num_parameters=1, init=0.25),
).to(DEVICE)
decoder = an.model.Decoder(
    kernel_size=11,
    kernel_decay=2,
    n_layers=1,
    start_channels=4,
    max_channels=32,
    input_shape=INPUT_SHAPE,
    n_blocks=6,
    relu=nn.PReLU,
    relu_kwargs=dict(num_parameters=1, init=0.25),
).to(DEVICE)

# define optimizer and loss function

for param in encoder.parameters():
    param.requires_grad = True
for param in decoder.parameters():
    param.requires_grad = True

optimizer = Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=0.001,
    weight_decay=0,
)

# augmentations
noise_augmentations = an.transform.Compose(
    [
        an.transform.SetRandomPoissonExposure(
            low=50_000,
            high=100_000_000,
            normalize=False,
        ),
        an.transform.NormalizeMinMax(*NORM_RANGE),
    ]
)
training_augmentations = an.transform.Compose(
    [
        v2.RandomResizedCrop(
            size=INPUT_SHAPE,
            scale=[0.8, 1.0],
            ratio=[0.8, 1.2],
        ),
        v2.RandomHorizontalFlip(),
    ]
)
testing_augmentations = an.transform.Compose(
    [
        an.transform.Resize(INPUT_SHAPE),
        an.transform.NormalizeMinMax(*NORM_RANGE),
    ]
)

# training loop
times = [time.time()]
train_losses = []
val_losses = []
print("Starting training...")
print(
    f"Device: {DEVICE}",
    f"Training for {N_EPOCHS} epochs",
    f"Batch size: {BATCH_SIZE}",
    f"Optimizer: {optimizer.__class__.__name__}",
    f"Learning rate: {optimizer.param_groups[0]['lr']}",
    f"Denosier: {DENOISER}",
    sep="\n",
)
for epoch in range(N_EPOCHS):
    encoder.train()
    decoder.train()
    train_loss = 0
    for batch in training_loader:
        optimizer.zero_grad()
        x = batch.to(DEVICE)
        x = training_augmentations(x)
        if DENOISER:
            x_noisy = noise_augmentations(x)
            enc = encoder(x_noisy.unsqueeze(1))
            dec = decoder(enc).squeeze()
            loss = nn.MSELoss()(x, dec)
        else:
            x = noise_augmentations(x)
            enc = encoder(x.unsqueeze(1))
            dec = decoder(enc).squeeze()
            loss = nn.MSELoss()(x, dec)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(training_loader)
    train_losses.append(train_loss)

    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        val_loss = 0
        for batch in validation_loader:
            x = batch.to(DEVICE)
            x = testing_augmentations(x)
            if DENOISER:
                x_noisy = noise_augmentations(x)
                enc = encoder(x_noisy.unsqueeze(1))
                dec = decoder(enc).squeeze()
                loss = nn.MSELoss()(x, dec)
            else:
                x = noise_augmentations(x)
                enc = encoder(x.unsqueeze(1))
                dec = decoder(enc).squeeze()
                loss = nn.MSELoss()(x, dec)
            val_loss += loss.item()
        val_loss /= len(validation_loader)
        val_losses.append(val_loss)
    times.append(time.time())
    epoch_time = times[-1] - times[-2]
    print(
        f"Epoch {epoch+1}/{N_EPOCHS} | Train loss: {train_loss:.3e} | Val loss: {val_loss:.3e} | Time: {epoch_time:.2f}s"
    )
i=0
while True:
    i += 1
    save_name = f"arpesnet_{N_EPOCHS}epochs_{i:03}.pt"
    if not Path(save_name).exists():
        break



torch.save(
    {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    save_name,
)

# test the model

encoder.eval()
decoder.eval()

with torch.no_grad():
    n_imgs = len(test_imgs)
    grid = [
        ["loss", "loss", "loss"] + [f"original_{i}" for i in range(n_imgs)],
        ["loss", "loss", "loss"] + [f"rec_{i}" for i in range(n_imgs)],
        ["loss", "loss", "loss"] + [f"diff_{i}" for i in range(n_imgs)],
    ]
    fig, axes = plt.subplot_mosaic(grid, figsize=(6 + len(test_imgs), 4), constrained_layout=True)
    for name, ax in axes.items():
        if name != "loss":
            ax.axis("off")
    # self.plot_losses(ax=axes["loss"])
    axes["loss"].plot(train_losses, label="Train loss")
    axes["loss"].plot(val_losses, label="Validation loss")

    axes["loss"].legend()
    test_loss = 0
    for i, img in enumerate(test_imgs):
        img = testing_augmentations(img)
        rec = decoder(encoder(img.unsqueeze(0)))
        loss = nn.MSELoss()(rec, img).detach().squeeze().cpu().numpy()
        test_loss += loss
        img = img.detach().squeeze().cpu().numpy()
        rec = rec.detach().squeeze().cpu().numpy()
        clim = img.min(), img.max()
        diff = img - rec
        vmax = clim[1]
        axes[f"original_{i}"].imshow(img, cmap="viridis", clim=clim, origin="lower")
        axes[f"rec_{i}"].imshow(rec, cmap="viridis", clim=clim, origin="lower")
        axes[f"diff_{i}"].imshow(diff, cmap="bwr", clim=(-vmax, vmax), origin="lower")
        axes[f"diff_{i}"].set_title(f"MSE: {loss:.3f}")
    test_loss /= n_imgs
    axes["loss"].scatter(len(train_losses)-1, test_loss, label="Test loss")
    axes["loss"].legend()
    fig.suptitle(
        f"{save_name} | MSE: {train_loss:.2f} | Val MSE {val_loss:.2f} | Test MSE: {test_loss:.3f}"
    )
fig.savefig(f"{save_name}.png")
plt.show()
