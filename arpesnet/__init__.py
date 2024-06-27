"""
ARPESNet
~~~~~~~~~~~~

An autoencoder for ARPES data.
"""
from . import model, train, transform
from .core import ModelTrainer, load_trainer
from .utils import load_config

__version__ = "1.0.0"
__author__ = "Steinn Ymir Agustsson"
__email__ = "steinny@gmail.com"
__all__ = ["ModelTrainer", "load_trainer", "load_config"]
