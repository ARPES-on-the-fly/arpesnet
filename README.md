# neurarpes
Neural networks applied to ARPES data

## Installation

To install the package in an existing environment, clone the repository and run the following command in the root directory:

```bash
pip install -e .
```

### Proper installation

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

In order to install pytorch with the optimal settings, to have access, for example to GPU computing, we recomend folloiwing the instructions on the [pytorch website](https://pytorch.org/get-started/locally/)

Assuming we want to install pytorch with CUDA 12.1, we can run the following command:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

After installing pytorch, we can install `arpesnet` as described above, running:

```bash
pip install -e .
```

In order to run Jupyter notebooks, we need to install the `jupyter` package along with the `ipykernel` package. We packaged these together with `arpesnet`, so we can install them by running:

```bash
pip install -e .[notebook]
```

# Citation and Acknowledgements

If you use this package in your work, please cite the following paper:

```bibtex
@article{arpesnet,
  title={Neural networks applied to ARPES data},
  author={S.Y. Agustsson and P.Hofmann},
  journal={arXiv},
}
```