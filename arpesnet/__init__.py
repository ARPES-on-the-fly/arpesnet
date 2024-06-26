"""
ARPESNet
~~~~~~~~~~~~

An autoencoder for ARPES data.
"""

__version__ = "1.0.0"
__author__ = "Steinn Ymir Agustsson"
__email__ = "steinny@gmail.com"

from .core import ModelTrainer
from .utils import load_config
from . import transform, train, model