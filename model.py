import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import time


class ProfileDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = float(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


class ProfileModel:
    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.transform = None
        self.train_transform = None
        self.train_transform_no_aug = None
        self.init_model()

    def init_model(self):
        """Initialize ResNet18 model"""
        self.model = models.resnet18(pretrained=True)
        # Replace final layer for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model = self.model.to(self.device)

        # Setup transform for inference
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Setup transform for training (with augmentation)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Setup transform for training (without augmentation)
        self.train_transform_no_aug = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Try to load saved model
        self.load_model()

    def load_model(self, model_path="model_best.pth"):
        """Load saved model weights"""
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                print(f"Loaded model from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False

    def save_model(self, model_path="model_best.pth"):
        """Save model weights"""
        try:
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def predict(self, image, threshold=0.5):
        """Run inference on a PIL image"""
        if self.model is None:
            return None, 0.0

        # Ensure image is RGB (convert from RGBA if needed)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = torch.sigmoid(self.model(img_tensor))
            score = output.item()
            prediction = "YES" if score >= threshold else "NO"

        return prediction, score

    def set_eval_mode(self):
        """Set model to evaluation mode"""
        if self.model:
            self.model.eval()

    def set_train_mode(self):
        """Set model to training mode"""
        if self.model:
            self.model.train()

    def get_model_info(self):
        """Get basic model information"""
        if self.model is None:
            return "Model not initialized"

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "architecture": "ResNet18",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def load_data_from_csv(self, csv_path):
        """Load labeled data from CSV file"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Filter out missing files
        existing_files = []
        labels = []

        for _, row in df.iterrows():
            if os.path.exists(row["filename"]):
                existing_files.append(row["filename"])
                # Convert label to binary (YES=1, NO=0)
                label = 1.0 if row["label"].lower() == "yes" else 0.0
                labels.append(label)

        return existing_files, labels

    def create_data_loaders(
        self, csv_path, train_split=0.8, batch_size=32, use_augmentation=True
    ):
        """Create train and validation data loaders"""
        image_paths, labels = self.load_data_from_csv(csv_path)

        if len(image_paths) == 0:
            raise ValueError("No valid images found in CSV")

        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths,
            labels,
            train_size=train_split,
            random_state=42,
            stratify=labels,
        )

        # Choose training transform based on augmentation setting
        train_transform = (
            self.train_transform if use_augmentation else self.train_transform_no_aug
        )

        # Create datasets
        train_dataset = ProfileDataset(train_paths, train_labels, train_transform)
        val_dataset = ProfileDataset(val_paths, val_labels, self.transform)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, len(train_paths), len(val_paths)

    def _log_message(self, message, log_callback=None):
        """Helper function to send log messages"""
        if log_callback:
            log_callback(message)
        else:
            print(message)

    def train_model(
        self,
        csv_path,
        epochs=50,
        train_split=0.8,
        patience=5,
        min_delta=0.001,
        lr=0.001,
        batch_size=32,
        use_augmentation=True,
        log_level="basic",
        progress_callback=None,
        log_callback=None,
    ):
        """
        Train the model with progress callbacks for GUI updates

        Args:
            csv_path: Path to CSV file with labels
            epochs: Maximum number of epochs
            train_split: Train/validation split ratio
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            lr: Learning rate
            batch_size: Batch size
            use_augmentation: Whether to use data augmentation
            log_level: Level of detail in logging ("basic", "detailed", "debug")
            progress_callback: Function to call with progress updates
                             callback(epoch, total_epochs, train_loss, val_acc, best_val_acc, status)
            log_callback: Function to call with detailed logs
                        callback(log_message)

        Returns:
            Dict with training results
        """
        try:
            # Load data
            if progress_callback:
                progress_callback(0, epochs, 0.0, 0.0, 0.0, "Loading data...")

            self._log_message(f"Training configuration:", log_callback)
            self._log_message(f"  - Epochs: {epochs}", log_callback)
            self._log_message(f"  - Batch size: {batch_size}", log_callback)
            self._log_message(f"  - Learning rate: {lr}", log_callback)
            self._log_message(f"  - Use augmentation: {use_augmentation}", log_callback)
            self._log_message(f"  - Train/val split: {train_split}", log_callback)
            self._log_message(f"  - Log level: {log_level}", log_callback)

            train_loader, val_loader, train_size, val_size = self.create_data_loaders(
                csv_path, train_split, batch_size, use_augmentation
            )

            # Log dataset statistics
            image_paths, labels = self.load_data_from_csv(csv_path)
            yes_count = sum(1 for label in labels if label == 1.0)
            no_count = len(labels) - yes_count

            self._log_message(f"Dataset statistics:", log_callback)
            self._log_message(f"  - Total samples: {len(labels)}", log_callback)
            self._log_message(
                f"  - YES samples: {yes_count} ({yes_count / len(labels) * 100:.1f}%)",
                log_callback,
            )
            self._log_message(
                f"  - NO samples: {no_count} ({no_count / len(labels) * 100:.1f}%)",
                log_callback,
            )
            self._log_message(f"  - Train samples: {train_size}", log_callback)
            self._log_message(f"  - Val samples: {val_size}", log_callback)
            self._log_message(
                f"  - Batches per epoch: {len(train_loader)}", log_callback
            )

            if progress_callback:
                progress_callback(
                    0,
                    epochs,
                    0.0,
                    0.0,
                    0.0,
                    f"Data loaded: {train_size} train, {val_size} val samples",
                )

            # Setup training
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

            best_val_acc = 0.0
            patience_counter = 0
            start_time = time.time()

            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                batch_losses = []

                epoch_start_time = time.time()

                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    labels = labels.unsqueeze(1)  # Add dimension for BCEWithLogitsLoss

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    batch_loss = loss.item()
                    train_loss += batch_loss
                    batch_losses.append(batch_loss)

                    # Calculate accuracy
                    predicted = torch.sigmoid(outputs) > 0.5
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    # Detailed logging for batches
                    if log_level in ["detailed", "debug"]:
                        batch_acc = (predicted == labels).sum().item() / labels.size(0)
                        self._log_message(
                            f"  Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{len(train_loader)}: "
                            f"loss={batch_loss:.4f}, acc={batch_acc:.3f} ({(predicted == labels).sum().item()}/{labels.size(0)} correct)",
                            log_callback,
                        )

                        # Debug level: show sample predictions
                        if (
                            log_level == "debug" and batch_idx == 0
                        ):  # Only for first batch to avoid spam
                            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                            actual = labels.cpu().numpy().flatten()
                            self._log_message(
                                f"    Sample predictions: {probs[: min(4, len(probs))]}",
                                log_callback,
                            )
                            self._log_message(
                                f"    Actual labels: {actual[: min(4, len(actual))]}",
                                log_callback,
                            )

                # Validation phase
                self.model.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0.0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        labels = labels.unsqueeze(1)

                        outputs = self.model(images)
                        val_loss += criterion(outputs, labels).item()
                        predicted = torch.sigmoid(outputs) > 0.5
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                # Calculate metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                train_acc = 100.0 * train_correct / train_total
                val_acc = 100.0 * val_correct / val_total
                epoch_time = time.time() - epoch_start_time

                # Detailed epoch logging
                if log_level in ["detailed", "debug"]:
                    loss_std = np.std(batch_losses)
                    self._log_message(
                        f"Epoch {epoch + 1}/{epochs} Summary:", log_callback
                    )
                    self._log_message(
                        f"  - Train: loss={avg_train_loss:.4f}±{loss_std:.4f}, acc={train_acc:.2f}%",
                        log_callback,
                    )
                    self._log_message(
                        f"  - Val: loss={avg_val_loss:.4f}, acc={val_acc:.2f}%",
                        log_callback,
                    )
                    self._log_message(f"  - Time: {epoch_time:.1f}s", log_callback)

                # Basic logging
                if log_level == "basic":
                    self._log_message(
                        f"Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.4f}, "
                        f"train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%",
                        log_callback,
                    )

                # Update best model
                if val_acc > best_val_acc + min_delta:
                    best_val_acc = val_acc
                    self.save_model("model_best.pth")
                    patience_counter = 0
                    status = f"Epoch {epoch + 1}/{epochs} - New best model saved! (val_acc: {val_acc:.2f}%)"
                    self._log_message(f"*** {status} ***", log_callback)
                else:
                    patience_counter += 1
                    status = f"Epoch {epoch + 1}/{epochs} - No improvement for {patience_counter} epochs"

                # Progress callback
                if progress_callback:
                    elapsed_time = time.time() - start_time
                    progress_callback(
                        epoch + 1, epochs, avg_train_loss, val_acc, best_val_acc, status
                    )

                # Early stopping
                if patience_counter >= patience:
                    early_stop_msg = (
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    self._log_message(early_stop_msg, log_callback)
                    if progress_callback:
                        progress_callback(
                            epoch + 1,
                            epochs,
                            avg_train_loss,
                            val_acc,
                            best_val_acc,
                            early_stop_msg,
                        )
                    break

            # Load best model
            self.load_model("model_best.pth")

            total_time = time.time() - start_time
            final_status = f"Training completed! Best validation accuracy: {best_val_acc:.2f}% (Time: {total_time:.1f}s)"
            self._log_message(final_status, log_callback)

            if progress_callback:
                progress_callback(
                    epochs, epochs, avg_train_loss, val_acc, best_val_acc, final_status
                )

            return {
                "best_val_acc": best_val_acc,
                "final_train_loss": avg_train_loss,
                "final_val_acc": val_acc,
                "total_epochs": epoch + 1,
                "total_time": total_time,
                "train_samples": train_size,
                "val_samples": val_size,
            }

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self._log_message(error_msg, log_callback)
            if progress_callback:
                progress_callback(0, epochs, 0.0, 0.0, 0.0, error_msg)
            raise e
