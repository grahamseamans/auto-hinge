import pyautogui
import os
from datetime import datetime
from PIL import Image
import pandas as pd
import time


class ScreenshotManager:
    def __init__(self, config_manager):
        self.config = config_manager

    def take_screenshot(self):
        """Take screenshot of specified region"""
        region = self.config.get("capture_region", [0, 0, 800, 600])
        x, y, w, h = region
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        return screenshot

    def save_screenshot(self, screenshot, label=None):
        """Save screenshot with timestamp filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"

        # Ensure save directory exists
        save_dir = self.config.get("save_dir", "./images")
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, filename)
        screenshot.save(filepath)

        # If label provided, add to CSV
        if label is not None:
            self.add_label_to_csv(filename, label)

        return filepath

    def add_label_to_csv(self, filename, label):
        """Add label entry to CSV file"""
        csv_path = self.config.get("label_csv", "./labels.csv")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create or append to CSV
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=["filename", "label", "timestamp"])

        save_dir = self.config.get("save_dir", "./images")
        new_row = pd.DataFrame(
            {
                "filename": [os.path.join(save_dir, filename)],
                "label": [label],
                "timestamp": [timestamp],
            }
        )
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)

    def click_x_button(self):
        """Simulate click at x_button_coords"""
        coords = self.config.get("x_button_coords", [0, 0])
        x, y = coords
        pyautogui.click(x, y)

    def get_labeled_images(self):
        """Get all labeled images from CSV"""
        csv_path = self.config.get("label_csv", "./labels.csv")

        if not os.path.exists(csv_path):
            return pd.DataFrame(columns=["filename", "label", "timestamp"])

        return pd.read_csv(csv_path)

    def count_labels(self):
        """Count the number of YES and NO labels"""
        df = self.get_labeled_images()
        if df.empty:
            return {"yes": 0, "no": 0, "total": 0}

        yes_count = len(df[df["label"].str.lower() == "yes"])
        no_count = len(df[df["label"].str.lower() == "no"])

        return {"yes": yes_count, "no": no_count, "total": len(df)}

    def validate_image_files(self):
        """Validate that all images in CSV actually exist"""
        df = self.get_labeled_images()
        missing_files = []

        for _, row in df.iterrows():
            if not os.path.exists(row["filename"]):
                missing_files.append(row["filename"])

        return missing_files

    def automation_loop(
        self, model, update_gui_callback=None, should_stop_callback=None
    ):
        """
        Main automation loop for continuous operation

        Args:
            model: ProfileModel instance for predictions
            update_gui_callback: Function to call with (prediction, score, screenshot)
            should_stop_callback: Function that returns True when loop should stop
        """
        while True:
            # Check if we should stop
            if should_stop_callback and should_stop_callback():
                break

            # Click X button
            self.click_x_button()
            time.sleep(2)  # Wait 2 seconds

            # Take screenshot
            screenshot = self.take_screenshot()

            # Get prediction
            threshold = self.config.get("model_threshold", 0.5)
            prediction, score = model.predict(screenshot, threshold)

            # Update GUI if callback provided
            if update_gui_callback:
                update_gui_callback(prediction, score, screenshot)

            # If YES, stop loop
            if prediction == "YES":
                break

    def resize_for_display(self, screenshot, size=(400, 300)):
        """Resize screenshot for GUI display"""
        if screenshot is None:
            return None
        return screenshot.resize(size)
