import dearpygui.dearpygui as dpg
import threading
import numpy as np
from config_manager import ConfigManager
from model import ProfileModel
from screenshot import ScreenshotManager


class GUI:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.model = ProfileModel()
        self.screenshot_manager = ScreenshotManager(self.config_manager)
        self.running_automation = False
        self.running_training = False
        self.texture_counter = 0
        self.current_texture_tag = None
        self.training_logs = []

        # Data explorer state
        self.explorer_current_page = 0
        self.explorer_images_per_page = 6
        self.explorer_image_textures = {}  # Store texture tags for cleanup

    def guess_callback(self, sender, app_data):
        """Handle GUESS button click"""
        screenshot = self.screenshot_manager.take_screenshot()

        threshold = self.config_manager.get("model_threshold", 0.5)
        prediction, score = self.model.predict(screenshot, threshold)

        dpg.set_value(
            "prediction_text", f"Model Prediction: {prediction} ({score:.3f})"
        )
        self.update_screenshot_display(screenshot)

    def preview_callback(self, sender, app_data):
        """Handle PREVIEW button click"""
        # Update config first to ensure we have latest values
        self.update_config_callback(sender, app_data)

        # Take preview screenshot with template matching
        preview_screenshot = self.screenshot_manager.take_preview_screenshot()
        self.update_screenshot_display(preview_screenshot)
        dpg.set_value("prediction_text", "Preview: Showing detected photo and X button")

    def yes_callback(self, sender, app_data):
        """Handle YES button click"""
        screenshot = self.screenshot_manager.take_screenshot()
        self.screenshot_manager.save_screenshot(screenshot, "yes")
        self.update_screenshot_display(screenshot)
        dpg.set_value("prediction_text", "Labeled: YES")

    def no_callback(self, sender, app_data):
        """Handle NO button click"""
        screenshot = self.screenshot_manager.take_screenshot()
        self.screenshot_manager.save_screenshot(screenshot, "no")
        self.update_screenshot_display(screenshot)
        dpg.set_value("prediction_text", "Labeled: NO")
        # Automatically click X button with enhanced behavior for manual use
        self.screenshot_manager.click_x_button(focus_window=True, restore_cursor=True)

    def run_callback(self, sender, app_data):
        """Handle RUN button click"""
        if not self.running_automation:
            # Start automation in separate thread
            self.running_automation = True
            thread = threading.Thread(target=self._automation_worker)
            thread.daemon = True
            thread.start()
            dpg.set_item_label("run_button", "STOP")
        else:
            # Stop automation
            self.running_automation = False
            dpg.set_item_label("run_button", "RUN")

    def _automation_worker(self):
        """Worker function for automation loop"""

        def update_gui(prediction, score, screenshot):
            dpg.set_value(
                "prediction_text", f"Model Prediction: {prediction} ({score:.3f})"
            )
            self.update_screenshot_display(screenshot)

        def should_stop():
            return not self.running_automation

        self.screenshot_manager.automation_loop(
            model=self.model,
            update_gui_callback=update_gui,
            should_stop_callback=should_stop,
        )

        # Reset button when done
        self.running_automation = False
        dpg.set_item_label("run_button", "RUN")

    def train_callback(self, sender, app_data):
        """Handle TRAIN button click"""
        if not self.running_training:
            # Start training in separate thread
            self.running_training = True
            dpg.set_item_label("train_button", "STOP TRAINING")
            # Clear previous logs
            self.training_logs = []
            dpg.set_value("training_logs", "")
            thread = threading.Thread(target=self._training_worker)
            thread.daemon = True
            thread.start()
        else:
            # Stop training (this will be handled by the training thread)
            self.running_training = False
            dpg.set_item_label("train_button", "TRAIN")

    def _log_callback(self, message):
        """Handle log messages from training"""
        self.training_logs.append(message)
        # Keep only last 100 lines to prevent memory issues
        if len(self.training_logs) > 100:
            self.training_logs = self.training_logs[-100:]

        # Update log display
        log_text = "\n".join(self.training_logs)
        dpg.set_value("training_logs", log_text)

    def _training_worker(self):
        """Worker function for training"""
        try:
            # Get training parameters from GUI
            train_split = dpg.get_value("train_split")
            patience = dpg.get_value("patience")
            min_delta = dpg.get_value("min_delta")
            batch_size = dpg.get_value("batch_size")
            learning_rate = dpg.get_value("learning_rate")
            max_epochs = dpg.get_value("max_epochs")
            use_augmentation = dpg.get_value("use_augmentation")
            log_level = dpg.get_value("log_level")
            csv_path = self.config_manager.get("label_csv", "./labels.csv")

            # Training progress callback
            def progress_callback(
                epoch, total_epochs, train_loss, val_acc, best_val_acc, status
            ):
                if not self.running_training:
                    return  # Stop if user cancelled

                # Update progress bar
                progress = epoch / total_epochs
                dpg.set_value("training_progress", progress)

                # Update metrics
                dpg.set_value("epoch_text", f"Epoch: {epoch}/{total_epochs}")
                dpg.set_value("train_loss_text", f"Train Loss: {train_loss:.4f}")
                dpg.set_value("val_acc_text", f"Val Accuracy: {val_acc:.2f}%")
                dpg.set_value("best_acc_text", f"Best Accuracy: {best_val_acc:.2f}%")
                dpg.set_value("training_status", status)

                # Update main prediction text too
                dpg.set_value("prediction_text", status)

            # Start training with new parameters
            results = self.model.train_model(
                csv_path=csv_path,
                epochs=max_epochs,
                train_split=train_split,
                patience=patience,
                min_delta=min_delta,
                lr=learning_rate,
                batch_size=batch_size,
                use_augmentation=use_augmentation,
                log_level=log_level,
                progress_callback=progress_callback,
                log_callback=self._log_callback,
            )

            # Training completed successfully
            final_message = (
                f"Training completed! Best accuracy: {results['best_val_acc']:.2f}%"
            )
            dpg.set_value("training_status", final_message)
            dpg.set_value("prediction_text", final_message)
            self._log_callback(final_message)

        except Exception as e:
            error_message = f"Training failed: {str(e)}"
            dpg.set_value("training_status", error_message)
            dpg.set_value("prediction_text", error_message)
            self._log_callback(error_message)

        finally:
            # Reset training state
            self.running_training = False
            dpg.set_item_label("train_button", "TRAIN")

    def clear_logs_callback(self, sender, app_data):
        """Clear training logs"""
        self.training_logs = []
        dpg.set_value("training_logs", "")

    def save_logs_callback(self, sender, app_data):
        """Save training logs to file"""
        if self.training_logs:
            try:
                with open("training_logs.txt", "w") as f:
                    f.write("\n".join(self.training_logs))
                dpg.set_value("prediction_text", "Logs saved to training_logs.txt")
            except Exception as e:
                dpg.set_value("prediction_text", f"Failed to save logs: {e}")
        else:
            dpg.set_value("prediction_text", "No logs to save")

    # Data Explorer Callbacks
    def explorer_prev_page_callback(self, sender, app_data):
        """Handle previous page button"""
        if self.explorer_current_page > 0:
            self.explorer_current_page -= 1
            self.refresh_data_explorer()

    def explorer_next_page_callback(self, sender, app_data):
        """Handle next page button"""
        images, current_page, total_pages = (
            self.screenshot_manager.get_paginated_images(
                self.explorer_current_page, self.explorer_images_per_page
            )
        )
        if self.explorer_current_page < total_pages - 1:
            self.explorer_current_page += 1
            self.refresh_data_explorer()

    def explorer_refresh_callback(self, sender, app_data):
        """Handle refresh button"""
        self.refresh_data_explorer()

    def explorer_toggle_callback(self, sender, app_data):
        """Handle show/hide data explorer checkbox"""
        if app_data:
            dpg.show_item("data_explorer_section")
        else:
            dpg.hide_item("data_explorer_section")

    def explorer_label_checkbox_callback(self, sender, app_data, user_data):
        """Handle label checkbox toggle"""
        filename = user_data["filename"]
        new_label = "yes" if app_data else "no"

        success = self.screenshot_manager.update_label(filename, new_label)
        if success:
            dpg.set_value(
                "prediction_text", f"Updated {filename} to {new_label.upper()}"
            )
            # Update the checkbox label to reflect the new state
            new_checkbox_label = f"Label: {new_label.upper()}"
            dpg.configure_item(sender, label=new_checkbox_label)

            # Update statistics
            self.update_label_statistics()
        else:
            dpg.set_value("prediction_text", f"Failed to update label for {filename}")
            # Revert checkbox state
            dpg.set_value(sender, not app_data)

    def explorer_delete_callback(self, sender, app_data, user_data):
        """Handle delete button"""
        filename = user_data["filename"]

        success = self.screenshot_manager.delete_image_entry(filename, delete_file=True)
        if success:
            dpg.set_value("prediction_text", f"Deleted {filename}")
            # Refresh the explorer to remove the deleted item
            self.refresh_data_explorer()
            # Update statistics
            self.update_label_statistics()
        else:
            dpg.set_value("prediction_text", f"Failed to delete {filename}")

    def refresh_data_explorer(self):
        """Refresh the data explorer display"""
        # Clear existing textures for cleanup
        for texture_tag in self.explorer_image_textures.values():
            if dpg.does_item_exist(texture_tag):
                dpg.delete_item(texture_tag)
        self.explorer_image_textures.clear()

        # Get current page data
        images, current_page, total_pages = (
            self.screenshot_manager.get_paginated_images(
                self.explorer_current_page, self.explorer_images_per_page
            )
        )

        # Update page info
        if dpg.does_item_exist("explorer_page_info"):
            if total_pages > 0:
                dpg.set_value(
                    "explorer_page_info",
                    f"Page {current_page + 1} of {total_pages} | Total: {len(self.screenshot_manager.get_labeled_images())} images",
                )
            else:
                dpg.set_value("explorer_page_info", "No images found")

        # Update navigation buttons
        if dpg.does_item_exist("explorer_prev_btn"):
            dpg.configure_item("explorer_prev_btn", enabled=(current_page > 0))
        if dpg.does_item_exist("explorer_next_btn"):
            dpg.configure_item(
                "explorer_next_btn", enabled=(current_page < total_pages - 1)
            )

        # Clear existing image grid
        if dpg.does_item_exist("explorer_image_grid"):
            dpg.delete_item("explorer_image_grid")

        # Create new image grid
        if dpg.does_item_exist("explorer_content"):
            with dpg.group(tag="explorer_image_grid", parent="explorer_content"):
                # Create 2x3 grid
                for row in range(2):
                    with dpg.group(horizontal=True):
                        for col in range(3):
                            img_idx = row * 3 + col
                            if img_idx < len(images):
                                self.create_image_card(images[img_idx], img_idx)
                            else:
                                # Empty slot
                                with dpg.group():
                                    dpg.add_text("")

    def create_image_card(self, image_data, img_idx):
        """Create an image card with controls"""
        filename = image_data["filename"]
        label = image_data["label"]
        exists = image_data["exists"]

        with dpg.group():
            if exists:
                # Load and display image
                pil_image = self.screenshot_manager.load_image_for_display(filename)
                if pil_image:
                    texture_tag = self.create_image_texture(
                        pil_image, f"explorer_img_{img_idx}"
                    )
                    if texture_tag:
                        dpg.add_image(texture_tag, width=275, height=275)
                        self.explorer_image_textures[img_idx] = texture_tag
                    else:
                        dpg.add_text("Failed to load image", color=[255, 0, 0])
                else:
                    dpg.add_text("Image file corrupted", color=[255, 0, 0])
            else:
                # Missing file placeholder
                dpg.add_text("Image file missing", color=[255, 0, 0])
                dpg.add_text(f"Size: 275x275 (missing)")

            # Controls row
            with dpg.group(horizontal=True):
                # Label checkbox with dynamic label
                is_yes = label.lower() == "yes"
                checkbox_label = f"Label: {label.upper()}"
                dpg.add_checkbox(
                    label=checkbox_label,
                    default_value=is_yes,
                    tag=f"explorer_checkbox_{img_idx}",
                    callback=self.explorer_label_checkbox_callback,
                    user_data={"filename": filename, "img_idx": img_idx},
                )

                # Delete button
                dpg.add_button(
                    label="Del",
                    callback=self.explorer_delete_callback,
                    user_data={"filename": filename},
                    width=50,
                )

    def create_image_texture(self, pil_image, tag_prefix):
        """Create DearPyGui texture from PIL image"""
        try:
            # Convert to numpy array and normalize
            img_array = np.array(pil_image, dtype=np.float32) / 255.0

            # Ensure RGBA format
            if img_array.shape[2] == 3:  # RGB
                alpha = np.ones(
                    (img_array.shape[0], img_array.shape[1], 1), dtype=np.float32
                )
                img_array = np.concatenate([img_array, alpha], axis=2)

            # Flatten for DearPyGui
            img_data = img_array.flatten()

            # Create unique texture tag
            self.texture_counter += 1
            texture_tag = f"{tag_prefix}_{self.texture_counter}"

            # Create texture
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=pil_image.width,
                    height=pil_image.height,
                    default_value=img_data,
                    tag=texture_tag,
                    format=dpg.mvFormat_Float_rgba,
                )

            return texture_tag

        except Exception as e:
            print(f"Error creating texture: {e}")
            return None

    def update_label_statistics(self):
        """Update the label statistics display"""
        if dpg.does_item_exist("label_stats_text"):
            label_stats = self.screenshot_manager.count_labels()
            dpg.set_value(
                "label_stats_text",
                f"YES: {label_stats['yes']} | NO: {label_stats['no']} | Total: {label_stats['total']}",
            )

    def update_config_callback(self, sender, app_data):
        """Handle config field updates"""
        # Update config from GUI fields
        updates = {
            "phone_screen_region": [
                dpg.get_value("phone_x"),
                dpg.get_value("phone_y"),
                dpg.get_value("phone_w"),
                dpg.get_value("phone_h"),
            ],
            "photo_width": dpg.get_value("photo_width"),
            "photo_height": dpg.get_value("photo_height"),
            "heart_offset_x": dpg.get_value("heart_offset_x"),
            "heart_offset_y": dpg.get_value("heart_offset_y"),
            "template_threshold": dpg.get_value("template_threshold"),
            "model_threshold": dpg.get_value("threshold"),
            "save_dir": dpg.get_value("save_dir"),
            "label_csv": dpg.get_value("label_csv"),
        }
        self.config_manager.update_config(updates)

    def save_config_callback(self, sender, app_data):
        """Handle Save Config button"""
        self.update_config_callback(sender, app_data)
        self.config_manager.save_config()
        dpg.set_value("prediction_text", "Config saved!")

    def update_screenshot_display(self, screenshot):
        """Update the screenshot preview"""
        if screenshot is None:
            return

        # Show at actual size for better debugging
        actual_width, actual_height = screenshot.size
        screenshot_resized = screenshot

        # Convert to numpy array and normalize
        img_array = np.array(screenshot_resized, dtype=np.float32) / 255.0

        # DearPyGui expects RGBA format
        if img_array.shape[2] == 3:  # RGB
            alpha = np.ones(
                (img_array.shape[0], img_array.shape[1], 1), dtype=np.float32
            )
            img_array = np.concatenate([img_array, alpha], axis=2)

        # Flatten array for DearPyGui
        img_data = img_array.flatten()

        # Generate unique texture tag to avoid DearPyGui aliasing issues
        self.texture_counter += 1
        new_texture_tag = f"screenshot_texture_{self.texture_counter}"

        # Create new texture with actual dimensions
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=actual_width,
                height=actual_height,
                default_value=img_data,
                tag=new_texture_tag,
                format=dpg.mvFormat_Float_rgba,
            )

        # Clean up old texture after creating new one
        if self.current_texture_tag and dpg.does_item_exist(self.current_texture_tag):
            dpg.delete_item(self.current_texture_tag)

        # Update current texture reference
        self.current_texture_tag = new_texture_tag

        # Update image display with actual dimensions
        if dpg.does_item_exist("screenshot_image"):
            dpg.delete_item("screenshot_image")

        # Add new image with correct dimensions
        dpg.add_image(
            new_texture_tag,
            width=actual_width,
            height=actual_height,
            tag="screenshot_image",
            parent=dpg.get_item_parent("no_screenshot_text"),
        )

        # Hide the "no screenshot" text and show the image
        if dpg.does_item_exist("no_screenshot_text"):
            dpg.hide_item("no_screenshot_text")
        if dpg.does_item_exist("screenshot_image"):
            dpg.show_item("screenshot_image")

    def create_gui(self):
        """Create the main GUI"""
        dpg.create_context()

        # Create empty texture for screenshot display first
        with dpg.texture_registry():
            # Create a placeholder texture
            placeholder_data = [0.2, 0.2, 0.2, 1.0] * (400 * 300)  # Gray placeholder
            dpg.add_raw_texture(
                width=400,
                height=300,
                default_value=placeholder_data,
                tag="screenshot_texture",
                format=dpg.mvFormat_Float_rgba,
            )

        with dpg.window(
            label="Profile Classifier Tool", width=1200, height=1000, tag="main_window"
        ):
            # Top button row
            with dpg.group(horizontal=True):
                dpg.add_button(label="GUESS", callback=self.guess_callback, width=80)
                dpg.add_button(label="YES", callback=self.yes_callback, width=80)
                dpg.add_button(label="NO", callback=self.no_callback, width=80)
                dpg.add_button(
                    label="RUN", callback=self.run_callback, width=80, tag="run_button"
                )
                dpg.add_button(
                    label="TRAIN",
                    callback=self.train_callback,
                    width=100,
                    tag="train_button",
                )
                dpg.add_button(
                    label="PREVIEW", callback=self.preview_callback, width=80
                )

            dpg.add_separator()

            # Model prediction display
            dpg.add_text("Model Prediction: -", tag="prediction_text")

            dpg.add_separator()

            # Collapsible Data Explorer Section
            dpg.add_checkbox(
                label="Show Data Explorer",
                default_value=False,
                callback=self.explorer_toggle_callback,
            )

            with dpg.group(tag="data_explorer_section", show=False):
                dpg.add_text("Data Explorer:")

                # Controls row
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="← Prev",
                        callback=self.explorer_prev_page_callback,
                        width=80,
                        tag="explorer_prev_btn",
                        enabled=False,
                    )
                    dpg.add_button(
                        label="Next →",
                        callback=self.explorer_next_page_callback,
                        width=80,
                        tag="explorer_next_btn",
                        enabled=False,
                    )
                    dpg.add_button(
                        label="Refresh",
                        callback=self.explorer_refresh_callback,
                        width=80,
                    )

                # Page info
                dpg.add_text("Click Refresh to load images", tag="explorer_page_info")

                # Content area for images
                with dpg.group(tag="explorer_content"):
                    dpg.add_text("Click Refresh to load images")

            dpg.add_separator()

            # Training Progress Section
            dpg.add_text("Training Progress:")
            with dpg.group():
                dpg.add_progress_bar(
                    tag="training_progress", default_value=0.0, width=400
                )

                with dpg.group(horizontal=True):
                    dpg.add_text("Ready to train", tag="epoch_text")
                    dpg.add_text("", tag="train_loss_text")

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="val_acc_text")
                    dpg.add_text("", tag="best_acc_text")

                dpg.add_text("Click TRAIN to start training", tag="training_status")

            dpg.add_separator()

            # Training Logs Section
            dpg.add_text("Training Logs:")
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Clear Logs", callback=self.clear_logs_callback, width=100
                    )
                    dpg.add_button(
                        label="Save Logs", callback=self.save_logs_callback, width=100
                    )

                dpg.add_input_text(
                    tag="training_logs",
                    default_value="",
                    multiline=True,
                    readonly=True,
                    width=600,
                    height=150,
                )

            dpg.add_separator()

            # Screenshot preview area
            dpg.add_text("Screenshot Preview:")
            with dpg.group():
                dpg.add_image(
                    "screenshot_texture",
                    width=400,
                    height=300,
                    tag="screenshot_image",
                    show=False,
                )
                dpg.add_text("No screenshot taken yet", tag="no_screenshot_text")

            dpg.add_separator()

            # Config panel
            dpg.add_text("Configuration:")
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_text("Phone Screen Region:")
                    dpg.add_input_int(
                        label="X",
                        default_value=self.config_manager.get("phone_screen_region")[0],
                        width=80,
                        tag="phone_x",
                        callback=self.update_config_callback,
                    )
                    dpg.add_input_int(
                        label="Y",
                        default_value=self.config_manager.get("phone_screen_region")[1],
                        width=80,
                        tag="phone_y",
                        callback=self.update_config_callback,
                    )
                    dpg.add_input_int(
                        label="W",
                        default_value=self.config_manager.get("phone_screen_region")[2],
                        width=80,
                        tag="phone_w",
                        callback=self.update_config_callback,
                    )
                    dpg.add_input_int(
                        label="H",
                        default_value=self.config_manager.get("phone_screen_region")[3],
                        width=80,
                        tag="phone_h",
                        callback=self.update_config_callback,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Photo Size:")
                    dpg.add_input_int(
                        label="Width",
                        default_value=self.config_manager.get("photo_width"),
                        width=80,
                        tag="photo_width",
                        callback=self.update_config_callback,
                    )
                    dpg.add_input_int(
                        label="Height",
                        default_value=self.config_manager.get("photo_height"),
                        width=80,
                        tag="photo_height",
                        callback=self.update_config_callback,
                    )

                with dpg.group(horizontal=True):
                    dpg.add_text("Heart Offset:")
                    dpg.add_input_int(
                        label="X",
                        default_value=self.config_manager.get("heart_offset_x"),
                        width=80,
                        tag="heart_offset_x",
                        callback=self.update_config_callback,
                    )
                    dpg.add_input_int(
                        label="Y",
                        default_value=self.config_manager.get("heart_offset_y"),
                        width=80,
                        tag="heart_offset_y",
                        callback=self.update_config_callback,
                    )

                dpg.add_input_float(
                    label="Template Match Threshold",
                    default_value=self.config_manager.get("template_threshold"),
                    tag="template_threshold",
                    callback=self.update_config_callback,
                )

                dpg.add_input_float(
                    label="Model Threshold",
                    default_value=self.config_manager.get("model_threshold"),
                    tag="threshold",
                    callback=self.update_config_callback,
                )

                dpg.add_input_text(
                    label="Save Directory",
                    default_value=self.config_manager.get("save_dir"),
                    tag="save_dir",
                    callback=self.update_config_callback,
                )

                dpg.add_input_text(
                    label="Label CSV Path",
                    default_value=self.config_manager.get("label_csv"),
                    tag="label_csv",
                    callback=self.update_config_callback,
                )

                dpg.add_button(label="Save Config", callback=self.save_config_callback)

            dpg.add_separator()

            # Enhanced Training settings
            dpg.add_text("Training Settings:")
            with dpg.group():
                # First row: Basic settings
                with dpg.group(horizontal=True):
                    dpg.add_input_float(
                        label="Train/Val Split",
                        default_value=0.8,
                        tag="train_split",
                        width=120,
                    )
                    dpg.add_input_int(
                        label="Max Epochs",
                        default_value=50,
                        tag="max_epochs",
                        width=120,
                    )
                    dpg.add_input_int(
                        label="Batch Size",
                        default_value=16,
                        tag="batch_size",
                        width=120,
                    )

                # Second row: Advanced settings
                with dpg.group(horizontal=True):
                    dpg.add_input_float(
                        label="Learning Rate",
                        default_value=0.001,
                        tag="learning_rate",
                        width=120,
                        format="%.4f",
                        step=0.0001,
                    )
                    dpg.add_input_int(
                        label="Early Stop Patience",
                        default_value=5,
                        tag="patience",
                        width=120,
                    )
                    dpg.add_input_float(
                        label="Min Delta",
                        default_value=0.001,
                        tag="min_delta",
                        width=120,
                        format="%.4f",
                        step=0.0001,
                    )

                # Third row: Checkboxes and dropdown
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        label="Use Data Augmentation",
                        default_value=True,
                        tag="use_augmentation",
                    )

                dpg.add_combo(
                    label="Log Level",
                    items=["basic", "detailed", "debug"],
                    default_value="detailed",
                    tag="log_level",
                    width=120,
                )

            dpg.add_separator()

            # Statistics display
            dpg.add_text("Label Statistics:")
            label_stats = self.screenshot_manager.count_labels()
            dpg.add_text(
                f"YES: {label_stats['yes']} | NO: {label_stats['no']} | Total: {label_stats['total']}",
                tag="label_stats_text",
            )

        # Setup viewport with larger size to accommodate new elements
        dpg.create_viewport(title="Profile Classifier Tool", width=1400, height=1200)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)

        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
