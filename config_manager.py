import json
import os


class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Default config if file doesn't exist
            default_config = {
                "capture_region": [0, 0, 800, 600],
                "x_button_coords": [0, 0],
                "save_dir": "./images",
                "label_csv": "./labels.csv",
                "model_threshold": 0.5,
            }
            self.save_config(default_config)
            return default_config

    def save_config(self, config=None):
        """Save configuration to JSON file"""
        if config is None:
            config = self.config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

    def update_config(self, updates):
        """Update config with new values"""
        self.config.update(updates)

    def get(self, key, default=None):
        """Get a config value"""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set a config value"""
        self.config[key] = value
