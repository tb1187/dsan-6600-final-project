import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy


class Optimizer:
    def __init__(
        self,
        model,
        train_data, # Training data from the loader class
        val_data, # Validation data from the loader class
        epochs=20, # Epochs for training, can be increased as needed
        lr=1e-4, # Initial learning rate
        device="cuda",
        use_scheduler=True, #dynamically adjusts learning rate
        patience=3 #for early stopping
    ):

        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(device)

        self.training_losses = [] #blank list for plotting later
        self.validation_losses = [] #blank list for plotting later

        self.use_scheduler = use_scheduler
        self.patience = patience

    # Training Loop With Early Stopping
    def train(self):

        model = self.model.to(self.device) # write model to device

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # LR Scheduler (This scheduler decreases learning rate when loss increases)
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=1, factor=0.1) 
            if self.use_scheduler
            else None
        )

        # Early stopping initializations
        best_val_loss = float("inf")
        best_weights = None
        overfit_counter = 0

        for epoch in range(self.epochs):

            # Training
            model.train()
            train_loss = 0.0 # initialize training loss

            for images, dish_ids, targets in tqdm(
                self.train_data,
                desc=f"[Training] Epoch {epoch+1}/{self.epochs}"
            ):
                images = images.to(self.device)
                dish_ids = dish_ids.to(self.device)
                targets = targets.to(self.device)

                # Compute loss; backprop to improve gradients
                optimizer.zero_grad()
                preds = model(images, dish_ids)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            # Append training loss of epoch to list for later plotting
            train_loss /= len(self.train_data.dataset)
            self.training_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0.0 # initialize validation loss

            # Compute validation loss
            with torch.no_grad():
                for images, dish_ids, targets in tqdm(
                    self.val_data,
                    desc=f"[Validation] Epoch {epoch+1}/{self.epochs}"
                ):
                    images = images.to(self.device)
                    dish_ids = dish_ids.to(self.device)
                    targets = targets.to(self.device)

                    preds = model(images, dish_ids)
                    loss = criterion(preds, targets)
                    val_loss += loss.item() * images.size(0)

            # Append validation loss of epoch to list for later plotting
            val_loss /= len(self.val_data.dataset)
            self.validation_losses.append(val_loss)

            # Scheduler update
            if scheduler:
                scheduler.step(val_loss)


            # Early Stopping Considerations (check to see if model is overfitting)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                overfit_counter = 0 # reset when not overfitting
            else:
                overfit_counter += 1  # add to counter when no improvement

            # Check stopping condition
            if overfit_counter >= self.patience:
                print("Early stopping activated due to overfitting.")
                print(f"Restoring best model (val_loss={best_val_loss:.4f})\n")
                model.load_state_dict(best_weights)
                break

        # Restore best weights if training ends naturally
        if best_weights is not None:
            model.load_state_dict(best_weights)

        return model


    # Plot Loss Curve
    def plot(self, save_path = None):
        plt.figure(figsize=(8, 5))
        plt.plot(self.training_losses, label="Training Loss", marker="o")
        plt.plot(self.validation_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.grid(True)
        plt.legend()

        if save_path is not None:
            plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
            
        plt.show()
