import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os


class ProfileModel:
    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.transform = None
        self.init_model()

    def init_model(self):
        """Initialize ResNet18 model"""
        self.model = models.resnet18(pretrained=True)
        # Replace final layer for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model = self.model.to(self.device)

        # Setup transform for preprocessing
        self.transform = transforms.Compose(
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
