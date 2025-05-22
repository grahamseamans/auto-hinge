import pyautogui
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
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

    def take_preview_screenshot(self):
        """Take a larger screenshot that encompasses both click coords and capture region"""
        # Get coordinates
        capture_region = self.config.get("capture_region", [0, 0, 800, 600])
        click_coords = self.config.get("x_button_coords", [0, 0])

        cap_x, cap_y, cap_w, cap_h = capture_region
        click_x, click_y = click_coords

        # Calculate bounding box that includes both areas with padding
        padding = 50
        min_x = min(cap_x, click_x) - padding
        min_y = min(cap_y, click_y) - padding
        max_x = max(cap_x + cap_w, click_x) + padding
        max_y = max(cap_y + cap_h, click_y) + padding

        # Ensure bounds are within screen
        min_x = max(0, min_x)
        min_y = max(0, min_y)

        preview_w = max_x - min_x
        preview_h = max_y - min_y

        # Take the larger screenshot
        screenshot = pyautogui.screenshot(region=(min_x, min_y, preview_w, preview_h))

        # Create annotated version
        annotated = self._annotate_preview(
            screenshot, min_x, min_y, capture_region, click_coords
        )

        return annotated

    def _annotate_preview(
        self, screenshot, offset_x, offset_y, capture_region, click_coords
    ):
        """Add visual annotations to preview screenshot"""
        # Make a copy to draw on
        img = screenshot.copy()
        draw = ImageDraw.Draw(img)

        cap_x, cap_y, cap_w, cap_h = capture_region
        click_x, click_y = click_coords

        # Calculate relative positions
        rel_cap_x = cap_x - offset_x
        rel_cap_y = cap_y - offset_y
        rel_click_x = click_x - offset_x
        rel_click_y = click_y - offset_y

        # Draw capture region rectangle (red)
        draw.rectangle(
            [rel_cap_x, rel_cap_y, rel_cap_x + cap_w, rel_cap_y + cap_h],
            outline="red",
            width=3,
        )

        # Draw click coordinates X (red)
        x_size = 15
        draw.line(
            [
                rel_click_x - x_size,
                rel_click_y - x_size,
                rel_click_x + x_size,
                rel_click_y + x_size,
            ],
            fill="red",
            width=3,
        )
        draw.line(
            [
                rel_click_x - x_size,
                rel_click_y + x_size,
                rel_click_x + x_size,
                rel_click_y - x_size,
            ],
            fill="red",
            width=3,
        )

        # Add labels
        try:
            # Try to use a default font, fall back to built-in if not available
            try:
                font = ImageFont.truetype("Arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            # Label the capture region
            draw.text(
                (rel_cap_x, rel_cap_y - 25), "Capture Region", fill="red", font=font
            )

            # Label the click point
            draw.text(
                (rel_click_x + 20, rel_click_y - 10),
                "Click Point",
                fill="red",
                font=font,
            )

        except Exception:
            # If font loading fails, just skip labels
            pass

        return img

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

    def resize_for_display(self, screenshot, max_size=(400, 300)):
        """Resize screenshot for GUI display while maintaining aspect ratio"""
        if screenshot is None:
            return None

        # Get original dimensions
        original_width, original_height = screenshot.size
        max_width, max_height = max_size

        # Calculate scaling factor to fit within max_size while maintaining aspect ratio
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_factor = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Resize with proper aspect ratio
        return screenshot.resize((new_width, new_height))
