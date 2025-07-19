"""
Enhanced Configuration for Gokul MetaTech Vehicle Recognition System
"""

# Camera Settings
CAMERA_URL = "http://192.168.29.145:8080/video"  # Replace with your camera IP
CAMERA_NAME = "Entrance Camera"

# Model Settings - Optimized for better detection
YOLO_MODEL = "yolov8n.pt"
MODEL_CONFIDENCE = 0.25    # Lowered for better detection
MODEL_DEVICE = "cpu"

# OCR Settings - Very permissive for better text recognition
OCR_LANGUAGES = ["en"]
OCR_CONFIDENCE = 0.15      # Very low threshold
OCR_GPU = False

# Processing Settings
FRAME_INTERVAL = 0.5       # Process every 0.5 seconds
SAVE_DETECTIONS = True
DETECTION_FOLDER = "detections"

# Debug Settings - Enable comprehensive debugging
DEBUG_MODE = True
SHOW_ALL_DETECTIONS = True
SAVE_DEBUG_IMAGES = True
LOG_OCR_ATTEMPTS = True
SHOW_PROCESSING_STEPS = True

# Enhanced Material Detection - Color Thresholds (RGB)
MATERIAL_COLORS = {
    "Red Material (Iron Ore)": [120, 50, 50],
    "Yellow Material (Sand)": [120, 120, 50],
    "Brown Material (Coal)": [80, 60, 40],
    "Gray Material (Cement)": [100, 100, 100],
    "Green Material (Fertilizer)": [50, 120, 50],
    "Blue Material (Chemicals)": [50, 50, 120],
    "White Material (Limestone)": [200, 200, 200],
    "Black Material (Coal)": [40, 40, 40],
    "Orange Material (Bauxite)": [180, 100, 40],
    "Mixed Materials": [100, 100, 100]
}

# Database Settings
DATABASE_FILE = "vehicles.db"
ENABLE_DATABASE = True
BACKUP_DATABASE = True

# Notification Settings
ENABLE_NOTIFICATIONS = True
NOTIFICATION_DURATION = 5

# API Settings (Optional - for weight machine integration)
WEBHOOK_URL = None  # Set to your weight machine API URL
ENABLE_API = False
API_TIMEOUT = 5
API_RETRY_ATTEMPTS = 3

# Indian License Plate Validation
ENABLE_PLATE_VALIDATION = True
MIN_PLATE_LENGTH = 3
MAX_PLATE_LENGTH = 15

# OCR Enhancement Settings
USE_MULTIPLE_OCR_ENGINES = True
OCR_PREPROCESSING_METHODS = 4
OCR_RESULT_VALIDATION = True

# Image Processing Settings
MIN_PLATE_WIDTH = 100
MIN_PLATE_HEIGHT = 30
MAX_PLATE_WIDTH = 500
MAX_PLATE_HEIGHT = 150

# Logging
LOG_LEVEL = "DEBUG"
LOG_FILE = "system.log"
MAX_LOG_SIZE = 10485760  # 10MB
LOG_BACKUP_COUNT = 3

# Performance Settings
DUPLICATE_DETECTION_TIMEOUT = 10  # seconds
MAX_CONCURRENT_PROCESSING = 2
MEMORY_CLEANUP_INTERVAL = 100  # frames

# System Information
SYSTEM_VERSION = "2.0"
COMPANY_NAME = "Gokul MetaTech"
SYSTEM_PURPOSE = "Vehicle Number Plate and Material Recognition"
