import sys
import torch
import torch.nn as nn
from torchvision import transforms
from model import FoodRegressor
from data_loader import FoodDataset
from utils import get_transforms
from optimizer import Optimizer
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtWidgets import QLineEdit, QComboBox
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt
import qdarktheme

# Load model function
def load_model(path = "../models/model_1.pt", device="cpu"):

    # Load in best model from training
    model = FoodRegressor().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# Initalize GUI Class
class AppGUI(QWidget):

    def __init__(self):

        super().__init__()

        # Set title, size, model, and image path
        self.setWindowTitle("Personalized Nutritional Advisor")
        self.resize(1200, 1500)
        self.device = "cuda" if torch.cuda_is_available() else "cpu"
        self.regressor = load_model(device=self.device)
        self.image_path = None

        # Enable Drag/Drop Images
        self.setAcceptDrops(True)

        # Image placement area
        self.image_label = QLabel("Drag & Drop Meal Image Here!")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(350)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 16px;
                font-size: 20px;
                padding: 20px;
            }
        """)

        # ---------------------------
        # Analyze Button
        # ---------------------------
        self.run_button = QPushButton("Analyze Meal")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_inference)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #3778C2;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #4A90E2;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)

        # ---------------------------
        # Input Notes
        # ---------------------------
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("Height")
        self.height_input.setStyleSheet("padding: 6px; border-radius: 6px; font-size: 15px;")

        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("Weight")
        self.weight_input.setStyleSheet("padding: 6px; border-radius: 6px; font-size: 15px;")

        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Age")
        self.age_input.setStyleSheet("padding: 6px; border-radius: 6px; font-size: 15px;")

        self.goal_dropdown = QComboBox()
        self.goal_dropdown.addItems(["General Health", "Cutting", "Bulking", "Maintenance"])
        self.goal_dropdown.setStyleSheet("padding: 6px; border-radius: 6px; font-size: 15px;")
        
        # Any additional information the user feels like the model should know
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText("Optional: Add additional information you feel is important here!")
        self.user_input.setStyleSheet("""
            QTextEdit {
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
            }
        """)

        # ---------------------------
        # Output Advice Box
        # ---------------------------
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setStyleSheet("""
            QTextEdit {
                border-radius: 8px;
                font-size: 16px;
                padding: 12px;
            }
        """)

        # ---------------------------
        # Layout
        # ---------------------------
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Drop Image + Analyze Button
        layout.addWidget(self.image_label)
        layout.addWidget(self.run_button)

        # Show physical stats
        layout.addWidget(QLabel("Physical Characteristics:"))
        layout.addWidget(self.height_input)
        layout.addWidget(self.weight_input)
        layout.addWidget(self.age_input)

        # Dietary Goal
        layout.addWidget(QLabel("Diet Goal:"))
        layout.addWidget(self.goal_dropdown)

        # Extra User Info
        layout.addWidget(QLabel("Additional Information:"))
        layout.addWidget(self.user_input)
        
        # LLM Output
        layout.addWidget(QLabel("Nutritional Recommendations:"))
        layout.addWidget(self.output_box)

        self.setLayout(layout)

    # ==================================================
    # Drag & Drop Events
    # ==================================================
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        url = event.mimeData().urls()[0]
        self.image_path = url.toLocalFile()

        pixmap = QPixmap(self.image_path).scaled(
            400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        self.run_button.setEnabled(True)


# Run GUI
if __name__ == "__main__":
    # Initialize App
    app = QApplication(sys.argv)
    
    # Set dark theme
    qdarktheme.setup_theme("dark")
    
    # Display GUI
    gui = AppGUI()
    gui.show()

    # Exit out of app when desired
    sys.exit(app.exec_())