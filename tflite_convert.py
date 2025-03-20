from ultralytics import YOLO

# List of models to convert
model_files = ["best1.pt", "dog.pt"]

# Convert each model to TFLite
for model_file in model_files:
    model = YOLO(model_file)  # Load each model
    model.export(format="tflite")  # Convert to TFLite
    print(f"✅ Converted {model_file} to TFLite")