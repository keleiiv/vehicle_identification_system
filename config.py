"""
Configuration for Gokul MetaTech Vehicle Recognition System
"""

# Camera Settings
CAMERA_URL = "http://192.168.29.145:8080/video"
CAMERA_NAME = "Entrance Camera"

# Model Settings
YOLO_MODEL = "yolov8n.pt"
MODEL_CONFIDENCE = 0.25
MODEL_DEVICE = "cpu"

# OCR Settings
OCR_LANGUAGES = ["en"]
OCR_CONFIDENCE = 0.4
OCR_GPU = False

# Processing Settings
FRAME_INTERVAL = 0.5
SAVE_DETECTIONS = True
DETECTION_FOLDER = "detections"

# Debug Settings
DEBUG_MODE = True
SHOW_ALL_DETECTIONS = True
SAVE_DEBUG_IMAGES = True
LOG_OCR_ATTEMPTS = True

# Material Detection Colors (RGB)
MATERIAL_COLORS = {
    "Red Material (Iron Ore)": [120, 50, 50],
    "Yellow Material (Sand)": [120, 120, 50],
    "Brown Material (Coal)": [80, 60, 40],
    "Gray Material (Cement)": [100, 100, 100],
    "Green Material (Fertilizer)": [50, 120, 50],
    "Blue Material (Chemicals)": [50, 50, 120],
    "White Material (Limestone)": [200, 200, 200],
    "Black Material (Coal)": [40, 40, 40]
}

# Database Settings
DATABASE_FILE = "vehicles.db"
ENABLE_DATABASE = True

# Notification Settings
ENABLE_NOTIFICATIONS = True
NOTIFICATION_DURATION = 5

# API Settings
WEBHOOK_URL = None
ENABLE_API = False

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "system.log"

