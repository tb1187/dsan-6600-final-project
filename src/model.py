import torch
import torch.nn as nn 
import torchvision.models as models

class FoodRegressor(nn.Module):
    def __init__(self, 
                 pretrained_model_name = "resnet18", # pretrained model, can tune later 
                 hidden_dim: int = 512, # size of regression head, can tune later 
                 dropout: float = 0.0, # dropout rate, can tune later
                 out_dim: int = 5, # portion + macros
                 freeze_pretrained: bool = False): # Parameter to freeze the pretrained model weights
        super().__init__()

        # I'm going to test on resnet18 first and see how that goes and then we can scale up if compute allows
        if pretrained_model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 # get pretrained weights
            pretrained_model = models.resnet18(weights = weights) # set pretrained weights
            feat_dim = pretrained_model.fc.in_features # Last layer output

        elif pretrained_model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 # get pretrained weights
            pretrained_model = models.resnet50(weights = weights) # load pretrained model with ImageNet weights
            feat_dim = pretrained_model.fc.in_features # get feature dimension of final layer
        
        else:
            raise ValueError(f"Unsupported pretrained_model_name: {pretrained_model_name}")

        # Freeze pretrained model if designated
        if freeze_pretrained:
            for param in pretrained_model.parameters():
                param.requires_grad = False
        
        pretrained_model.fc = nn.Identity() # Remove classification head

        self.pretrained_model = pretrained_model # set pretrained model

        # Create the regression head
        # This will either be a 512-512 connection or a 2048-512 connection depending on model
        layers = [nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace = True)]

        # Conditionally add dropout
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Output layer    
        layers.append(nn.Linear(hidden_dim, out_dim))

        # Initialize the regression head
        self.reg_head = nn.Sequential(*layers)

    def forward(self, x):
        features = self.pretrained_model(x) # feature generation using pretrained model
        output = self.reg_head(features) # regression output
        return output

