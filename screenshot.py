import pyautogui
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import time
import cv2
import numpy as np


class ScreenshotManager:
    def __init__(self, config_manager):
        self.config = config_manager

    def take_screenshot(self):
        """Take screenshot of specified region"""
        # Try template matching first
        phone_region = self.config.get("phone_screen_region", [0, 0, 400, 800])
        x, y, w, h = phone_region
        phone_screenshot = pyautogui.screenshot(region=(x, y, w, h))

        # Try to find photo using template matching
        photo_bounds, _ = self.find_active_profile_elements(phone_screenshot)

        if photo_bounds:
            # Crop to just the photo
            photo = phone_screenshot.crop(
                (
                    photo_bounds["x"],
                    photo_bounds["y"],
                    photo_bounds["x"] + photo_bounds["width"],
                    photo_bounds["y"] + photo_bounds["height"],
                )
            )
            return photo
        else:
            print("Template matching failed - no photo found!")
            return None

    def take_preview_screenshot(self):
        """Take preview screenshot showing template matching results"""
        # Take phone screen screenshot
        phone_region = self.config.get("phone_screen_region", [0, 0, 400, 800])
        x, y, w, h = phone_region
        phone_screenshot = pyautogui.screenshot(region=(x, y, w, h))

        # Save phone screenshot for debugging
        debug_path = "./debug_phone_screenshot.png"
        phone_screenshot.save(debug_path)

        # Find all template matches
        all_heart_matches, all_x_matches, photo_bounds, selected_x_button = (
            self.find_all_template_matches(phone_screenshot)
        )

        # Create annotated version showing all matches
        annotated = self._annotate_preview(
            phone_screenshot,
            all_heart_matches,
            all_x_matches,
            photo_bounds,
            selected_x_button,
        )
        return annotated

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

    def click_x_button(self, focus_window=False, restore_cursor=False):
        """
        Simulate click at x_button_coords with enhanced options

        Args:
            focus_window: If True, click on phone window first to focus it
            restore_cursor: If True, restore cursor to original position after clicking
        """
        # Save original cursor position if we need to restore it
        original_position = None
        if restore_cursor:
            original_position = pyautogui.position()

        # Try template matching first
        phone_region = self.config.get("phone_screen_region", [0, 0, 400, 800])
        phone_x, phone_y, w, h = phone_region
        phone_screenshot = pyautogui.screenshot(region=(phone_x, phone_y, w, h))

        # Find X button
        _, x_button = self.find_active_profile_elements(phone_screenshot)

        if x_button:
            # Convert relative coordinates to absolute screen coordinates
            abs_x = phone_x + x_button["x"] + x_button["width"] // 2
            abs_y = phone_y + x_button["y"] + x_button["height"] // 2

            if focus_window:
                # First click somewhere in the phone region to focus the window
                center_x = phone_x + w // 2
                center_y = phone_y + h // 2
                pyautogui.click(center_x, center_y)
                time.sleep(0.1)  # Brief pause to ensure window is focused

            # Click the X button
            pyautogui.click(abs_x, abs_y)

            # Restore cursor position if requested
            if restore_cursor and original_position:
                pyautogui.moveTo(original_position.x, original_position.y)

        else:
            print("Template matching failed for X button - no X button found!")

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

            # Click X button (no focusing needed in automation mode)
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

    def load_template(self, template_path):
        """Load template image for matching"""
        try:
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if template is None:
                raise Exception(f"Could not load template: {template_path}")
            return template
        except Exception as e:
            print(f"Error loading template {template_path}: {e}")
            return None

    def find_template_matches(self, screenshot, template, threshold=0.8):
        """Find all template matches above threshold"""
        # Convert PIL to OpenCV format
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        # Perform template matching
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)

        # Find all matches above threshold
        locations = np.where(result >= threshold)
        matches = []

        for pt in zip(*locations[::-1]):  # Switch x and y
            confidence = result[pt[1], pt[0]]
            matches.append(
                {
                    "x": pt[0],
                    "y": pt[1],
                    "confidence": confidence,
                    "width": template.shape[1],
                    "height": template.shape[0],
                }
            )

        return matches

    def find_all_template_matches(self, phone_screenshot):
        """Find all template matches for preview display"""
        try:
            # Load templates
            heart_template = self.load_template(
                self.config.get("heart_template", "./hinge-heart.png")
            )
            x_template = self.load_template(
                self.config.get("x_template", "./hinge-x.png")
            )

            all_heart_matches = []
            all_x_matches = []
            photo_bounds = None
            selected_x_button = None

            threshold = self.config.get("template_threshold", 0.8)

            if heart_template is not None:
                # Find ALL heart matches (even low confidence ones for debugging)
                all_heart_matches = self.find_template_matches(
                    phone_screenshot,
                    heart_template,
                    0.3,  # Lower threshold to see more
                )

                # Find the best heart match above threshold for photo calculation
                good_hearts = [
                    h for h in all_heart_matches if h["confidence"] >= threshold
                ]

                if good_hearts:
                    # Get topmost heart (smallest y value)
                    active_heart = min(good_hearts, key=lambda h: h["y"])

                    # Calculate photo boundaries from heart position
                    photo_width = self.config.get("photo_width", 280)
                    photo_height = self.config.get("photo_height", 280)
                    heart_offset_x = self.config.get("heart_offset_x", 10)
                    heart_offset_y = self.config.get("heart_offset_y", 10)

                    # Heart is at bottom-right of photo
                    photo_right = active_heart["x"] + heart_offset_x
                    photo_bottom = active_heart["y"] + heart_offset_y
                    photo_left = photo_right - photo_width
                    photo_top = photo_bottom - photo_height

                    photo_bounds = {
                        "x": photo_left,
                        "y": photo_top,
                        "width": photo_width,
                        "height": photo_height,
                        "selected_heart": active_heart,
                    }
            if x_template is not None:
                # Find ALL X button matches
                all_x_matches = self.find_template_matches(
                    phone_screenshot,
                    x_template,
                    0.3,  # Lower threshold to see more
                )

                # Select best X button above threshold
                good_x_buttons = [
                    x for x in all_x_matches if x["confidence"] >= threshold
                ]

                if good_x_buttons:
                    selected_x_button = good_x_buttons[0]  # Take first good match
            else:
                print("Failed to load X template")

            return all_heart_matches, all_x_matches, photo_bounds, selected_x_button

        except Exception as e:
            print(f"Error in template matching: {e}")
            return [], [], None, None

    def find_active_profile_elements(self, phone_screenshot):
        """Find photo and X button using template matching (simplified version)"""
        _, _, photo_bounds, selected_x_button = self.find_all_template_matches(
            phone_screenshot
        )
        return photo_bounds, selected_x_button

    def _annotate_preview(
        self,
        screenshot,
        all_heart_matches,
        all_x_matches,
        photo_bounds,
        selected_x_button,
    ):
        """Add visual annotations showing all template matching results"""
        img = screenshot.copy()
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Draw ALL heart matches with pink X marks and confidence scores
        for heart in all_heart_matches:
            center_x = heart["x"] + heart["width"] // 2
            center_y = heart["y"] + heart["height"] // 2
            confidence = heart["confidence"]

            # Pink X mark
            x_size = 8
            color = "hotpink"
            draw.line(
                [
                    center_x - x_size,
                    center_y - x_size,
                    center_x + x_size,
                    center_y + x_size,
                ],
                fill=color,
                width=2,
            )
            draw.line(
                [
                    center_x - x_size,
                    center_y + x_size,
                    center_x + x_size,
                    center_y - x_size,
                ],
                fill=color,
                width=2,
            )

            # Confidence label
            if font:
                draw.text(
                    (center_x + 10, center_y - 5),
                    f"♥{confidence:.2f}",
                    fill=color,
                    font=font,
                )

        # Draw ALL X button matches with blue X marks and confidence scores
        for x_match in all_x_matches:
            center_x = x_match["x"] + x_match["width"] // 2
            center_y = x_match["y"] + x_match["height"] // 2
            confidence = x_match["confidence"]

            # Blue X mark
            x_size = 8
            color = "cyan"
            draw.line(
                [
                    center_x - x_size,
                    center_y - x_size,
                    center_x + x_size,
                    center_y + x_size,
                ],
                fill=color,
                width=2,
            )
            draw.line(
                [
                    center_x - x_size,
                    center_y + x_size,
                    center_x + x_size,
                    center_y - x_size,
                ],
                fill=color,
                width=2,
            )

            # Confidence label
            if font:
                draw.text(
                    (center_x + 10, center_y - 5),
                    f"X{confidence:.2f}",
                    fill=color,
                    font=font,
                )

        # Draw green rectangle around calculated photo area
        if photo_bounds:
            color = "lime"
            draw.rectangle(
                [
                    photo_bounds["x"],
                    photo_bounds["y"],
                    photo_bounds["x"] + photo_bounds["width"],
                    photo_bounds["y"] + photo_bounds["height"],
                ],
                outline=color,
                width=3,
            )

            # Label photo area
            if font:
                draw.text(
                    (photo_bounds["x"], photo_bounds["y"] - 20),
                    "Photo Area",
                    fill=color,
                    font=font,
                )

            # Highlight the selected heart that was used for photo calculation
            if "selected_heart" in photo_bounds:
                selected_heart = photo_bounds["selected_heart"]
                center_x = selected_heart["x"] + selected_heart["width"] // 2
                center_y = selected_heart["y"] + selected_heart["height"] // 2

                # Draw green circle around selected heart
                circle_size = 15
                draw.ellipse(
                    [
                        center_x - circle_size,
                        center_y - circle_size,
                        center_x + circle_size,
                        center_y + circle_size,
                    ],
                    outline="lime",
                    width=3,
                )

        # Draw green rectangle around selected X button
        if selected_x_button:
            color = "lime"
            draw.rectangle(
                [
                    selected_x_button["x"],
                    selected_x_button["y"],
                    selected_x_button["x"] + selected_x_button["width"],
                    selected_x_button["y"] + selected_x_button["height"],
                ],
                outline=color,
                width=3,
            )

            # Label selected X
            if font:
                draw.text(
                    (selected_x_button["x"], selected_x_button["y"] - 20),
                    "Selected X",
                    fill=color,
                    font=font,
                )

        return img

    def get_paginated_images(self, page=0, images_per_page=6):
        """Get a page of labeled images for data exploration"""
        df = self.get_labeled_images()
        if df.empty:
            return [], 0, 0

        # Calculate pagination
        total_images = len(df)
        total_pages = (total_images + images_per_page - 1) // images_per_page
        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, total_images)

        # Get page data
        page_data = df.iloc[start_idx:end_idx]

        # Prepare image data with validation
        images = []
        for _, row in page_data.iterrows():
            filepath = row["filename"]
            label = row["label"]
            timestamp = row["timestamp"]

            # Check if file exists
            exists = os.path.exists(filepath)

            images.append(
                {
                    "filename": filepath,
                    "label": label,
                    "timestamp": timestamp,
                    "exists": exists,
                    "index": start_idx
                    + len(images),  # Global index for unique identification
                }
            )

        return images, page, total_pages

    def update_label(self, filename, new_label):
        """Update label for a specific image file"""
        csv_path = self.config.get("label_csv", "./labels.csv")

        if not os.path.exists(csv_path):
            return False

        try:
            df = pd.read_csv(csv_path)
            # Update the label
            mask = df["filename"] == filename
            if mask.any():
                df.loc[mask, "label"] = new_label
                df.loc[mask, "timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.to_csv(csv_path, index=False)
                return True
            return False
        except Exception as e:
            print(f"Error updating label: {e}")
            return False

    def delete_image_entry(self, filename, delete_file=False):
        """Delete an image entry from CSV and optionally delete the file"""
        csv_path = self.config.get("label_csv", "./labels.csv")

        if not os.path.exists(csv_path):
            return False

        try:
            df = pd.read_csv(csv_path)
            # Remove the entry
            df = df[df["filename"] != filename]
            df.to_csv(csv_path, index=False)

            # Optionally delete the actual file
            if delete_file and os.path.exists(filename):
                os.remove(filename)

            return True
        except Exception as e:
            print(f"Error deleting entry: {e}")
            return False

    def load_image_for_display(self, filepath):
        """Load an image file for GUI display"""
        try:
            if os.path.exists(filepath):
                image = Image.open(filepath)
                # Convert to RGB if needed (handles RGBA, grayscale, etc)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return image
            return None
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            return None
