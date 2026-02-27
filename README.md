# HandTrack-enhanced-
FINALLY,  FULL ASL RECOGNITION 
# PHASE 1 — ASL Data Collector
─────────────────────────────
Records your hand landmarks for each ASL letter and saves them to asl_dataset.csv.

HOW TO USE:
  1. Run this script:       python 1_collect_data.py
  2. Press a letter key (A-Z) to select which letter you want to record.
  3. Hold your ASL gesture in front of the camera.
  4. Press SPACE to capture a sample (aim for 30-50 samples per letter).
  5. Repeat for all letters you want to recognize.
  6. Press Q to quit and save.

TIPS:
  - Vary your hand position slightly between captures (slight rotation, distance).
  - Good lighting helps MediaPipe track landmarks accurately.
  - You can re-run this script to ADD more samples — it appends to the CSV.

Requirements:
  pip install opencv-python mediapipe
