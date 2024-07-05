# neurarpes
Neural networks applied to ARPES data

# Installation

To install the package in an existing environment, clone the repository and run the following command in the root directory:

### Virtual Environment
In order to properly install this package, it is recommended to use a virtual environment. 
This can be done using conda by running the following commands:

```bash
conda create -n arpesnet python=3.11
```

This will create a new empty environment called `arpesnet`

Activate this environment

```bash
conda activate arpesnet
```

### Package Installation

To start, we install PyTorch, the main dependency of the package.
How to correctly install pytorch depends on the system and the desired settings.
We recomend to follow the instructions on the [pytorch website](https://pytorch.org/get-started/locally/),
however here are a few common case examples:

#### CPU only

To install pytorch with CPU only, run the following command:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### CUDA 12.1

To install pytorch with CUDA 11.1, run the following command:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### Apple MX processor

To install pytorch with the Apple MX processor, run the following command:

```bash
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

### Package Installation
After installing pytorch, we can install `arpesnet` as described above, running:

```bash
pip install -e .
```

#### Notebooks
A few additional packages are required to run the notebooks available in the `notebooks` folder.
To install these, install the package with the following command:

```bash
pip install -e .[notebooks]
```

# Citation and Acknowledgements

If you use this package in your work, please cite the following paper:

```bibtex
@article{arpesnet,
  title={Neural networks applied to ARPES data},
  author={Steinn  Ýmir Ágústsson, Mohammad Ahsanul Haque, Thi Tam Truong, Marco Bianchi, Nikita Klyuchnikov, Davide Mottin, Panagiotis Karras and Philip Hofmann},
  status={In Preparation}
}
```