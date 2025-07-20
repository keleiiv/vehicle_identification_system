"""
Complete Configuration for Gokul MetaTech Vehicle Recognition System
"""

# Company Information
COMPANY_NAME = "Gokul MetaTech"
SYSTEM_VERSION = "2.0"
SYSTEM_PURPOSE = "Vehicle Number Plate and Material Recognition"

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
MIN_PLATE_LENGTH = 3
MAX_PLATE_LENGTH = 15
DUPLICATE_DETECTION_TIMEOUT = 10

# Debug Settings
DEBUG_MODE = True
SHOW_ALL_DETECTIONS = True
SAVE_DEBUG_IMAGES = True
LOG_OCR_ATTEMPTS = True
SHOW_PROCESSING_STEPS = True

# Enhanced Features
USE_MULTIPLE_OCR_ENGINES = True
ENABLE_PLATE_VALIDATION = True
OCR_PREPROCESSING_METHODS = 4
OCR_RESULT_VALIDATION = True

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
BACKUP_DATABASE = True

# Notification Settings
ENABLE_NOTIFICATIONS = True
NOTIFICATION_DURATION = 5

# API Settings
WEBHOOK_URL = None
ENABLE_API = False
API_TIMEOUT = 5
API_RETRY_ATTEMPTS = 3

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "system.log"
MAX_LOG_SIZE = 10485760  # 10MB
LOG_BACKUP_COUNT = 3

# Performance Settings
MAX_CONCURRENT_PROCESSING = 2
MEMORY_CLEANUP_INTERVAL = 100
