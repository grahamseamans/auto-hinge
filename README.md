Here’s the updated GUI Spec Doc reflecting all the new behavior, config integration, and the passive overlay system. This version is complete and implementation-ready.

⸻

📄 GUI SPEC: Profile Classifier Tool (v2)

⸻

🧭 Purpose

A desktop GUI app to:
	•	Manually label screenshots (YES / NO)
	•	Get predictions from a model
	•	Automate profile rejection
	•	Edit runtime config visually
	•	Display passive overlay for live screenshot region + X click location
	•	Always show screenshot for current action

⸻

🖼 Layout (Main Window)

+----------------------------------------------------------+
| [GUESS]   [YES]   [NO]   [RUN]   [TRAIN]   [SHOW OVERLAY]|
+----------------------------------------------------------+
| Model Prediction: [ YES / NO / - ]                       |
+----------------------------------------------------------+
|                                                          |
|                [  Screenshot Preview  ]                  |
|                                                          |
+----------------------------------------------------------+
| Config Panel:                                            |
|   Capture Region: X [____] Y [____] W [____] H [____]    |
|   X Button Coords: X [____] Y [____]                     |
|   Model Threshold: [_____]                               |
|   Save Directory: [__________] [Browse]                  |
|   Label CSV Path: [__________] [Browse]                  |
|   Save Config [ ]                                        |
+----------------------------------------------------------+
| Training Settings:                                       |
|   Train/Test Split: [_____]   (e.g., 0.8)                |
|   Early Stop Patience: [_____]  Min Delta: [_____]       |
|   [Train Model]                                          |
+----------------------------------------------------------+


⸻

🎛 Controls

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
	•	Draws on-screen passive overlay:
	•	Red box → capture_region
	•	Red X → x_button_coords
	•	Updates dynamically as config fields change

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

🔧 Optional Additions (later)
	•	Disable buttons during [RUN]
	•	[Cancel RUN] button or Esc to stop
	•	[Manual Override] button if [RUN] hits a YES but you disagree

⸻

✅ If this doc looks good, I’ll now implement main.py with:
	•	GUI layout per above
	•	Live-updating overlay window
	•	Model stub + CSV I/O
	•	Config loader/writer

Let me know if you want:
	•	Pre-filled dummy model
	•	Test images
	•	File save naming strategy changed (e.g. UUID instead of timestamp)