Here's the updated GUI Spec Doc reflecting all the new behavior, config integration, and the preview functionality. This version is complete and implementation-ready.

⸻

📄 GUI SPEC: Profile Classifier Tool (v3)

⸻

🧭 Purpose

A desktop GUI app to:
	•	Manually label screenshots (YES / NO)
	•	Get predictions from a model
	•	Automate profile rejection
	•	Edit runtime config visually
	•	Preview capture regions and click coordinates visually
	•	Always show screenshot for current action

⸻

🖼 Layout (Main Window)

+------------------------------------------------------------------+
|[PREVIEW] [GUESS]   [YES]   [NO]   [RUN]   [TRAIN]   [SHOW OVERLAY]|
+------------------------------------------------------------------+
| Model Prediction: [ YES / NO / - ]                               |
+------------------------------------------------------------------+
|                                                                  |
|                [  Screenshot Preview  ]                          |
|                                                                  |
+------------------------------------------------------------------+
| Config Panel:                                                    |
|   Capture Region: X [____] Y [____] W [____] H [____]            |
|   X Button Coords: X [____] Y [____]                             |
|   Model Threshold: [_____]                                       |
|   Save Directory: [__________] [Browse]                          |
|   Label CSV Path: [__________] [Browse]                          |
|   Save Config [ ]                                                |
+------------------------------------------------------------------+
| Training Settings:                                               |
|   Train/Test Split: [_____]   (e.g., 0.8)                        |
|   Early Stop Patience: [_____]  Min Delta: [_____]               |
|   [Train Model]                                                  |
+------------------------------------------------------------------+


⸻

🎛 Controls

PREVIEW (Button)
	•	Takes a larger screenshot encompassing both capture region and click coordinates
	•	Draws red rectangle showing capture region boundaries
	•	Draws red X marking click coordinates
	•	Adds text labels "Capture Region" and "Click Point"
	•	Perfect for testing your setup before running automation
	•	Updates config from GUI fields automatically

⸻

GUESS (Button)
	•	Takes screenshot
	•	Runs model
	•	Updates prediction label (YES / NO)
	•	Displays screenshot
	•	Does not save

⸻

YES / NO (Buttons)
	•	Take screenshot
	•	Save image
	•	Append to labels.csv:
	•	[filename, yes/no, timestamp]
	•	Display screenshot
	•	No model inference

⸻

RUN (Button)
	•	Loops:
	1.	Simulate click at x_button_coords
	2.	Wait 2s
	3.	Take screenshot
	4.	Run model
	5.	If model says NO → repeat
	6.	If model says YES → stop loop
	•	Show screenshot and prediction

⸻

TRAIN (Button)
	•	Loads all labeled images
	•	Applies train/val split
	•	Trains model (ResNet18)
	•	Applies early stopping using val loss
	•	Saves model_best.pth and reloads for inference
	•	Training output shown in text area or terminal

⸻

SHOW OVERLAY (Toggle)
	•	Legacy feature - replaced by PREVIEW button
	•	Originally designed for on-screen overlay
	•	PREVIEW provides better visual feedback without overlay complexity

⸻

🧩 Config File: config.json

{
  "capture_region": [x, y, width, height],
  "x_button_coords": [x, y],
  "save_dir": "./images",
  "label_csv": "./labels.csv",
  "model_threshold": 0.5
}

	•	Values are live-editable in GUI
	•	Can be updated via inputs or [Browse]
	•	Saved on [Save Config]

⸻

🧠 Model Behavior
	•	Backbone: torchvision.models.resnet18(pretrained=True)
	•	Final layer: 1 neuron, sigmoid activation
	•	Output ∈ [0,1]
	•	If score ≥ threshold → YES
	•	Else → NO

⸻

🗃 Saved Output

Screenshot Files
	•	./images/YYYYMMDD_HHMMSS.png

Label Log: labels.csv

filename,label,timestamp
images/20250522_140301.png,yes,2025-05-22 14:03:01
images/20250522_140407.png,no,2025-05-22 14:04:07


⸻

🔧 Implementation Notes

✅ **Preview Functionality**
	•	Smart bounding box calculation includes both regions with padding
	•	Visual annotations using PIL ImageDraw with red overlays
	•	Automatic font fallback (Arial → default) for cross-platform compatibility
	•	Real-time config updates before taking preview

✅ **Modular Architecture**
	•	`screenshot.py` - Preview generation and annotation logic
	•	`gui.py` - PREVIEW button integration and callback
	•	`config_manager.py` - Live config updates
	•	Clean separation of concerns

✅ **User Experience**
	•	PREVIEW shows exactly what will be captured and where clicks occur
	•	No complex overlay system - simple, effective visual feedback
	•	Works with any screen resolution and coordinate ranges

⸻

🔧 Optional Additions (later)
	•	Disable buttons during [RUN]
	•	[Cancel RUN] button or Esc to stop
	•	[Manual Override] button if [RUN] hits a YES but you disagree
	•	Export preview images for documentation
