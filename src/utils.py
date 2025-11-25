from torchvision import transforms

# returns standard image preprocessing
# ResNet is input size 224x224
def get_transforms(img_size = 224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Resize to ResNet input
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # Derived from ImageNet
    ])
    return transform

def load_model(path="best_regressor.pt", device="cpu"):
    model = FoodRegressorModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model