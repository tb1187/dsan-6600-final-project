import sys
import torch
import torch.nn as nn
from torchvision import transforms
from model import FoodRegressor
from data_loader import FoodDataset
from utils import get_transforms
from optimizer import Optimizer
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt