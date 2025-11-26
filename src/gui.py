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
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


# Load model function
def load_model(path="./models/model_1.pt", device="cpu"):
    model = FoodRegressor().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# GUI Application
class AppGUI(QWidget):

    def __init__(self):
        super().__init__()

        # Window settings
        self.setWindowTitle("Personalized Nutritional Advisor")
        self.resize(1400, 600)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.regressor = load_model(device=self.device)
        self.image_path = None

        # IMAGE DISPLAY AREA
        self.image_label = QLabel("No image selected.\nClick 'Upload Image' to choose a meal photo.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(350)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 16px;
                font-size: 18px;
                padding: 20px;
            }
        """)

        # Buttons: Upload + Analyze
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: white;
                border-radius: 10px;
                padding: 8px 16px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #777;
            }
        """)

        self.run_button = QPushButton("Analyze Meal")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_inference)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #3778C2;
                color: white;
                border-radius: 10px;
                padding: 8px 16px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #4A90E2;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.upload_button)
        buttons_row.addWidget(self.run_button)

        # PHYSICAL STATS INPUTS
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("Height")
        self.height_input.setStyleSheet("padding: 6px; border-radius: 6px; font-size: 15px;")

        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("Weight (lbs)")
        self.weight_input.setStyleSheet("padding: 6px; border-radius: 6px; font-size: 15px;")

        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Age (years)")
        self.age_input.setStyleSheet("padding: 6px; border-radius: 6px; font-size: 15px;")

        self.goal_dropdown = QComboBox()
        self.goal_dropdown.addItems(["General Health", "Cutting", "Bulking", "Maintenance"])
        self.goal_dropdown.setStyleSheet("padding: 6px; border-radius: 6px; font-size: 15px;")

        # Extra user input
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText(
            "Optional: Add additional information you feel is important here!"
        )
        self.user_input.setStyleSheet("""
            QTextEdit {
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
            }
        """)

        # OUTPUT BOX
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setStyleSheet("""
            QTextEdit {
                border-radius: 8px;
                font-size: 16px;
                padding: 12px;
            }
        """)

        # LAYOUT
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        layout.addWidget(self.image_label)
        layout.addLayout(buttons_row)

        layout.addWidget(QLabel("Physical Characteristics:"))
        layout.addWidget(self.height_input)
        layout.addWidget(self.weight_input)
        layout.addWidget(self.age_input)

        layout.addWidget(QLabel("Diet Goal:"))
        layout.addWidget(self.goal_dropdown)

        layout.addWidget(QLabel("Additional Information:"))
        layout.addWidget(self.user_input)

        layout.addWidget(QLabel("Nutritional Recommendations:"))
        layout.addWidget(self.output_box)

        self.setLayout(layout)


    # File upload
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Meal Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if not file_path:
            return  # user cancelled

        self.image_path = file_path

        pixmap = QPixmap(self.image_path).scaled(
            400, 400,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        self.run_button.setEnabled(True)


    # MODEL INFERENCE
    def run_inference(self):

        if not self.image_path:
            self.output_box.setPlainText("No image selected.")
            return

        # Load image
        img = Image.open(self.image_path).convert("RGB")

        # Transform image
        transformations = get_transforms()
        img_tensor = transformations(img).unsqueeze(0).to(self.device)

        # Run model
        with torch.no_grad():
            macros = self.regressor(img_tensor).cpu().numpy()[0]

        # Format output (assuming: [portion, calories, protein, fat, carbs])
        macro_text = (
            f"Portion:  {macros[0]:.2f} g\n"
            f"Calories: {macros[1]:.2f}\n"
            f"Protein:  {macros[2]:.2f} g\n"
            f"Fat:      {macros[3]:.2f} g\n"
            f"Carbs:    {macros[4]:.2f} g\n"
        )

        self.output_box.setPlainText(macro_text)


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    gui = AppGUI()
    gui.show()

    sys.exit(app.exec_())
