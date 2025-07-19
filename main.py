"""
Gokul MetaTech Vehicle Number Plate Recognition System
Enhanced Main Application with Advanced OCR and Debugging
Version 2.0
"""

import cv2
import numpy as np
import time
import logging
import sqlite3
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import threading
import queue

# Third-party imports
try:
    from ultralytics import YOLO
    import easyocr
    import pyperclip
    from plyer import notification
    import requests
    # Optional: Tesseract OCR as backup
    try:
        import pytesseract
        TESSERACT_AVAILABLE = True
    except ImportError:
        TESSERACT_AVAILABLE = False
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Please install: pip install -r requirements.txt")
    exit(1)

# Import configuration
import config

class VehicleRecognitionSystem:
    """Enhanced main system class for vehicle recognition"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        self.setup_database()
        self.load_models()
        self.running = False
        self.last_detection_time = {}
        self.detection_count = 0
        self.frame_count = 0
        self.processing_stats = {
            'total_frames': 0,
            'plates_detected': 0,
            'plates_recognized': 0,
            'ocr_attempts': 0,
            'ocr_successes': 0
        }
        
        logging.info(f"{config.COMPANY_NAME} {config.SYSTEM_PURPOSE} v{config.SYSTEM_VERSION} initialized")
    
    def setup_logging(self):
        """Setup enhanced logging configuration"""
        from logging.handlers import RotatingFileHandler
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Setup rotating file handler
        file_handler = RotatingFileHandler(
            f"logs/{config.LOG_FILE}",
            maxBytes=config.MAX_LOG_SIZE,
            backupCount=config.LOG_BACKUP_COUNT
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.LOG_LEVEL))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            config.DETECTION_FOLDER,
            "logs",
            "debug_images",
            "backup"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        logging.info("Directory structure created")
    
    def setup_database(self):
        """Setup enhanced SQLite database"""
        if not config.ENABLE_DATABASE:
            return
            
        try:
            conn = sqlite3.connect(config.DATABASE_FILE)
            cursor = conn.cursor()
            
            # Main detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    plate_number TEXT NOT NULL,
                    material_type TEXT,
                    confidence REAL,
                    ocr_confidence REAL,
                    image_path TEXT,
                    camera_id TEXT DEFAULT 'main',
                    processing_time REAL,
                    validation_status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # OCR attempts table for analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ocr_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER,
                    method TEXT,
                    raw_text TEXT,
                    cleaned_text TEXT,
                    confidence REAL,
                    success BOOLEAN,
                    timestamp TEXT,
                    FOREIGN KEY (detection_id) REFERENCES detections (id)
                )
            ''')
            
            # System statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    total_frames INTEGER,
                    detections INTEGER,
                    success_rate REAL,
                    avg_processing_time REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_detections_plate ON detections(plate_number)",
                "CREATE INDEX IF NOT EXISTS idx_ocr_detection_id ON ocr_attempts(detection_id)",
            ]
            
            for index in indexes:
                cursor.execute(index)
            
            conn.commit()
            conn.close()
            logging.info("Enhanced database setup complete")
            
        except Exception as e:
            logging.error(f"Database setup error: {e}")
    
    def load_models(self):
        """Load YOLO and OCR models with error handling"""
        try:
            # Load YOLO model
            print("🤖 Loading YOLO model... (This may take a few minutes on first run)")
            logging.info("Loading YOLO model...")
            self.yolo_model = YOLO(config.YOLO_MODEL)
            print("✅ YOLO model loaded successfully")
            logging.info("YOLO model loaded successfully")
            
            # Load EasyOCR reader
            print("📖 Loading OCR model... (This may take a few minutes)")
            logging.info("Loading EasyOCR model...")
            self.ocr_reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=config.OCR_GPU)
            print("✅ EasyOCR model loaded successfully")
            logging.info("EasyOCR model loaded successfully")
            
            # Test Tesseract availability
            if TESSERACT_AVAILABLE and config.USE_MULTIPLE_OCR_ENGINES:
                try:
                    import pytesseract
                    # Test Tesseract
                    test_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
                    cv2.putText(test_img, "TEST", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    pytesseract.image_to_string(test_img)
                    print("✅ Tesseract OCR available as backup")
                    logging.info("Tesseract OCR initialized successfully")
                    self.tesseract_available = True
                except:
                    print("⚠️ Tesseract OCR not available")
                    logging.warning("Tesseract OCR not available")
                    self.tesseract_available = False
            else:
                self.tesseract_available = False
            
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            logging.error(f"Model loading error: {e}")
            raise
    
    def validate_indian_plate(self, text: str) -> bool:
        """Validate if text matches Indian license plate patterns"""
        if not config.ENABLE_PLATE_VALIDATION or not text:
            return True
        
        # Remove spaces and special characters for pattern matching
        clean_text = text.replace(' ', '').replace('-', '').upper()
        
        # Indian license plate patterns
        patterns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',      # Standard: MH12AB1234
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$',  # Older format
            r'^[A-Z]{3}[0-9]{4}$',                         # Armed forces: MIL1234
            r'^[A-Z]{2}[0-9]{2}[A-Z]{3}[0-9]{4}$',        # New format: MH12ABC1234
            r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$',              # BH series: 22BH1234AB
        ]
        
        for pattern in patterns:
            if re.match(pattern, clean_text):
                logging.debug(f"Plate validation passed: '{text}' matches pattern {pattern}")
                return True
        
        # If no pattern matches but text looks reasonable, still accept it
        if len(clean_text) >= config.MIN_PLATE_LENGTH and len(clean_text) <= config.MAX_PLATE_LENGTH:
            if any(c.isdigit() for c in clean_text) and any(c.isalpha() for c in clean_text):
                logging.debug(f"Plate validation passed: '{text}' has mixed alphanumeric content")
                return True
        
        logging.debug(f"Plate validation failed: '{text}' doesn't match known patterns")
        return False
    
    def advanced_text_cleaning(self, text: str) -> str:
        """Advanced text cleaning for license plates"""
        if not text:
            return ""
        
        # Initial cleaning
        cleaned = text.upper().strip()
        
        # Remove unwanted characters but keep spaces, hyphens
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c in ' -')
        
        # Handle common OCR mistakes
        replacements = {
            # Number/Letter confusions
            '0': ['O', 'D', 'Q'],
            'O': ['0'],
            '1': ['I', 'L', '|'],
            'I': ['1', 'L', '|'],
            'L': ['1', 'I'],
            '5': ['S'],
            'S': ['5'],
            '8': ['B'],
            'B': ['8'],
            '6': ['G'],
            'G': ['6'],
            '2': ['Z'],
            'Z': ['2'],
        }
        
        # Create multiple variants and choose the best one
        variants = [cleaned]
        
        # Generate variants with substitutions
        for original, alternatives in replacements.items():
            if original in cleaned:
                for alt in alternatives:
                    variant = cleaned.replace(original, alt)
                    if variant != cleaned:
                        variants.append(variant)
        
        # Score variants based on Indian plate patterns
        best_variant = cleaned
        best_score = 0
        
        for variant in variants:
            score = 0
            # Add points for validation
            if self.validate_indian_plate(variant):
                score += 10
            # Add points for reasonable length
            clean_variant = variant.replace(' ', '').replace('-', '')
            if config.MIN_PLATE_LENGTH <= len(clean_variant) <= config.MAX_PLATE_LENGTH:
                score += 5
            # Add points for mixed content
            if any(c.isdigit() for c in clean_variant) and any(c.isalpha() for c in clean_variant):
                score += 3
            
            if score > best_score:
                best_score = score
                best_variant = variant
        
        # Normalize spaces
        final_text = ' '.join(best_variant.split())
        
        if config.DEBUG_MODE:
            logging.debug(f"Text cleaning: '{text}' -> '{final_text}' (score: {best_score})")
        
        return final_text
    
    def preprocess_plate_image(self, plate_img) -> List[Tuple[str, np.ndarray]]:
        """Enhanced preprocessing with multiple approaches"""
        try:
            if plate_img is None or plate_img.size == 0:
                return []
            
            # Convert to grayscale
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img
            
            # Validate dimensions
            height, width = gray.shape
            if height < 10 or width < 30:
                return []
            
            # Resize to optimal size for OCR
            if height < config.MIN_PLATE_HEIGHT or width < config.MIN_PLATE_WIDTH:
                scale_h = max(1.0, config.MIN_PLATE_HEIGHT / height)
                scale_w = max(1.0, config.MIN_PLATE_WIDTH / width)
                scale_factor = max(scale_h, scale_w)
                
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                if config.SHOW_PROCESSING_STEPS:
                    print(f"   📐 Resized to {new_width}x{new_height} (scale: {scale_factor:.2f})")
            
            processed_images = []
            
            # Method 1: Basic OTSU threshold
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("OTSU Threshold", thresh1))
            
            # Method 2: Inverted OTSU threshold (for dark text on light background)
            _, thresh1_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            processed_images.append(("OTSU Inverted", thresh1_inv))
            
            # Method 3: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            processed_images.append(("Adaptive Gaussian", adaptive))
            
            # Method 4: Enhanced contrast + threshold
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("CLAHE + OTSU", thresh2))
            
            # Method 5: Denoised + threshold
            denoised = cv2.fastNlMeansDenoising(gray)
            _, thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("Denoised + OTSU", thresh3))
            
            # Method 6: Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            processed_images.append(("Morphological", morph))
            
            # Method 7: Bilateral filter + threshold
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            _, thresh4 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(("Bilateral + OTSU", thresh4))
            
            # Save debug images
            if config.SAVE_DEBUG_IMAGES:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"debug_images/original_plate_{timestamp}.jpg", plate_img)
                for i, (name, img) in enumerate(processed_images):
                    safe_name = name.replace(" ", "_").replace("+", "plus")
                    cv2.imwrite(f"debug_images/processed_{i}_{safe_name}_{timestamp}.jpg", img)
            
            return processed_images[:config.OCR_PREPROCESSING_METHODS]
            
        except Exception as e:
            logging.error(f"Preprocessing error: {e}")
            return [("Original", plate_img)]
    
    def ocr_with_tesseract(self, image) -> List[Dict]:
        """Backup OCR using Tesseract"""
        if not self.tesseract_available:
            return []
        
        try:
            import pytesseract
            
            # Different Tesseract configurations for license plates
            configs = [
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            ]
            
            results = []
            for i, custom_config in enumerate(configs):
                try:
                    text = pytesseract.image_to_string(image, config=custom_config)
                    if text and text.strip():
                        results.append({
                            'text': text.strip(),
                            'confidence': 0.5,  # Default confidence for Tesseract
                            'method': f'Tesseract_PSM_{8-i}'
                        })
                except:
                    continue
            
            return results
            
        except Exception as e:
            if config.DEBUG_MODE:
                logging.debug(f"Tesseract OCR failed: {e}")
            return []
    
    def detect_and_read_plate(self, frame):
        """Enhanced license plate detection and OCR"""
        start_time = time.time()
        
        try:
            if config.DEBUG_MODE:
                logging.debug("Starting enhanced plate detection...")
            
            self.processing_stats['total_frames'] += 1
            
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False, conf=config.MODEL_CONFIDENCE)
            
            if not results:
                if config.DEBUG_MODE:
                    logging.debug("No YOLO results returned")
                return None, None, 0.0
            
            result = results[0]
            if not hasattr(result, 'boxes') or result.boxes is None:
                if config.DEBUG_MODE:
                    logging.debug("No boxes in YOLO results")
                return None, None, 0.0
            
            total_detections = len(result.boxes.data.tolist()) if result.boxes else 0
            
            if config.SHOW_ALL_DETECTIONS:
                print(f"🔍 Frame {self.frame_count}: Found {total_detections} objects")
            
            if total_detections == 0:
                return None, None, 0.0
            
            self.processing_stats['plates_detected'] += total_detections
            
            best_text = None
            best_coords = None
            best_conf = 0.0
            best_ocr_conf = 0.0
            all_ocr_attempts = []
            
            # Process each detection
            for i, box in enumerate(result.boxes.data.tolist()):
                if len(box) < 6:
                    continue
                
                x1, y1, x2, y2, conf, class_id = box
                
                if config.DEBUG_MODE:
                    print(f"   Object {i+1}: conf={conf:.3f}, coords=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
                
                if conf < config.MODEL_CONFIDENCE:
                    continue
                
                # Extract and validate plate region
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                if (x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0] or 
                    x2 <= x1 or y2 <= y1):
                    continue
                
                plate_img = frame[y1:y2, x1:x2]
                
                if plate_img.size == 0:
                    continue
                
                if config.SHOW_PROCESSING_STEPS:
                    print(f"   📸 Processing plate region: {plate_img.shape}")
                
                # Preprocess plate image with multiple methods
                processed_versions = self.preprocess_plate_image(plate_img)
                
                if not processed_versions:
                    continue
                
                # Run OCR on all processed versions
                all_ocr_results = []
                
                # EasyOCR with different configurations
                for approach_name, processed_img in processed_versions:
                    ocr_configs = [
                        {'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'},
                        {'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '},
                        {'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'},
                        {'paragraph': True},
                    ]
                    
                    for config_idx, ocr_config in enumerate(ocr_configs):
                        try:
                            self.processing_stats['ocr_attempts'] += 1
                            ocr_results = self.ocr_reader.readtext(processed_img, **ocr_config)
                            
                            for result in ocr_results:
                                if len(result) >= 3:
                                    text, bbox, confidence = result[1], result[0], result[2]
                                    method_name = f"EasyOCR_{approach_name}_Config{config_idx+1}"
                                    
                                    all_ocr_results.append({
                                        'text': text,
                                        'confidence': confidence,
                                        'method': method_name,
                                        'bbox': bbox
                                    })
                                    
                                    all_ocr_attempts.append({
                                        'method': method_name,
                                        'raw_text': text,
                                        'confidence': confidence,
                                        'success': len(text.strip()) >= 3
                                    })
                        
                        except Exception as e:
                            if config.DEBUG_MODE:
                                logging.debug(f"EasyOCR failed for {approach_name} config {config_idx+1}: {e}")
                
                # Tesseract OCR as backup
                if config.USE_MULTIPLE_OCR_ENGINES and processed_versions:
                    for approach_name, processed_img in processed_versions[:2]:  # Only try on best 2 methods
                        tesseract_results = self.ocr_with_tesseract(processed_img)
                        for result in tesseract_results:
                            method_name = f"Tesseract_{approach_name}_{result['method']}"
                            all_ocr_results.append({
                                'text': result['text'],
                                'confidence': result['confidence'],
                                'method': method_name,
                                'bbox': None
                            })
                            
                            all_ocr_attempts.append({
                                'method': method_name,
                                'raw_text': result['text'],
                                'confidence': result['confidence'],
                                'success': len(result['text'].strip()) >= 3
                            })
                
                # Process and rank OCR results
                if all_ocr_results:
                    if config.LOG_OCR_ATTEMPTS:
                        print(f"   📝 Found {len(all_ocr_results)} OCR results:")
                        for idx, result in enumerate(all_ocr_results[:5]):  # Show top 5
                            print(f"      {idx+1}. '{result['text']}' (conf: {result['confidence']:.3f}) [{result['method']}]")
                    
                    # Clean and validate results
                    valid_results = []
                    for result in all_ocr_results:
                        cleaned_text = self.advanced_text_cleaning(result['text'])
                        if len(cleaned_text) >= config.MIN_PLATE_LENGTH:
                            # Calculate composite score
                            score = result['confidence']
                            if self.validate_indian_plate(cleaned_text):
                                score += 0.3  # Bonus for valid pattern
                            if len(cleaned_text) >= 6:  # Typical plate length
                                score += 0.1
                            
                            valid_results.append({
                                'text': cleaned_text,
                                'confidence': result['confidence'],
                                'score': score,
                                'method': result['method']
                            })
                    
                    # Sort by composite score
                    valid_results.sort(key=lambda x: x['score'], reverse=True)
                    
                    if valid_results and valid_results[0]['confidence'] > config.OCR_CONFIDENCE:
                        candidate_text = valid_results[0]['text']
                        candidate_conf = valid_results[0]['confidence']
                        
                        if config.DEBUG_MODE:
                            print(f"   ✅ Best OCR result: '{candidate_text}' (conf: {candidate_conf:.3f}) [{valid_results[0]['method']}]")
                        
                        if conf > best_conf:
                            best_text = candidate_text
                            best_coords = (x1, y1, x2, y2)
                            best_conf = conf
                            best_ocr_conf = candidate_conf
                            self.processing_stats['ocr_successes'] += 1
            
            processing_time = time.time() - start_time
            
            if best_text:
                print(f"🎉 PLATE DETECTED: '{best_text}' (YOLO: {best_conf:.3f}, OCR: {best_ocr_conf:.3f}) in {processing_time:.2f}s")
                logging.info(f"Plate detected: '{best_text}' YOLO_conf={best_conf:.3f} OCR_conf={best_ocr_conf:.3f}")
                self.processing_stats['plates_recognized'] += 1
            
            return best_text, best_coords, best_conf
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Plate detection error: {e}")
            logging.error(f"Plate detection error: {e}")
            return None, None, 0.0
    
    def detect_material_color(self, frame, roi=None):
        """Enhanced material detection with region of interest"""
        try:
            # Define analysis region
            if roi and all(coord is not None for coord in roi):
                x1, y1, x2, y2 = roi
                # Expand ROI for better material context
                h, w = frame.shape[:2]
                expand_x, expand_y = 30, 20
                x1 = max(0, x1 - expand_x)
                y1 = max(0, y1 - expand_y)
                x2 = min(w, x2 + expand_x)
                y2 = min(h, y2 + expand_y)
                analysis_region = frame[y1:y2, x1:x2]
            else:
                # Use vehicle area (lower half of frame)
                h, w = frame.shape[:2]
                analysis_region = frame[h//3:, w//6:5*w//6]
            
            if analysis_region.size == 0:
                return "Unknown Material"
            
            # Color analysis
            avg_color = np.mean(analysis_region, axis=(0, 1))
            b, g, r = avg_color
            
            # Additional color statistics
            std_color = np.std(analysis_region.reshape(-1, 3), axis=0)
            dominant_colors = self.get_dominant_colors(analysis_region, k=3)
            
            if config.DEBUG_MODE:
                logging.debug(f"Color analysis - Avg: ({r:.1f}, {g:.1f}, {b:.1f}), Std: ({std_color[2]:.1f}, {std_color[1]:.1f}, {std_color[0]:.1f})")
            
            best_material = "Unknown Material"
            best_score = 0
            material_scores = {}
            
            # Score each material type
            for material, (r_thresh, g_thresh, b_thresh) in config.MATERIAL_COLORS.items():
                score = self.calculate_material_score(r, g, b, r_thresh, g_thresh, b_thresh, std_color, dominant_colors)
                material_scores[material] = score
                
                if score > best_score:
                    best_score = score
                    best_material = material
            
            if config.DEBUG_MODE:
                sorted_materials = sorted(material_scores.items(), key=lambda x: x[1], reverse=True)
                logging.debug(f"Material scores: {sorted_materials[:3]}")
            
            # Require minimum confidence
            if best_score < 0.3:
                best_material = "Unknown Material"
            
            return best_material
            
        except Exception as e:
            logging.error(f"Material detection error: {e}")
            return "Detection Error"
    
    def get_dominant_colors(self, image, k=3):
        """Get dominant colors using k-means clustering"""
        try:
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            return centers.astype(np.uint8)
        except:
            return []
    
    def calculate_material_score(self, r, g, b, r_thresh, g_thresh, b_thresh, std_color, dominant_colors):
        """Calculate material matching score with multiple factors"""
        # Basic color distance
        distance = np.sqrt((r - r_thresh)**2 + (g - g_thresh)**2 + (b - b_thresh)**2)
        base_score = max(0, 1 - distance / 441.67)
        
        # Texture factor (higher std dev suggests more texture)
        texture_factor = 1.0
        avg_std = np.mean(std_color)
        if avg_std > 30:  # High texture materials like coal, gravel
            if "coal" in r_thresh or "gravel" in str(r_thresh):
                texture_factor = 1.2
        elif avg_std < 15:  # Smooth materials like liquids
            if "liquid" in str(r_thresh) or "smooth" in str(r_thresh):
                texture_factor = 1.1
        
        # Dominant color matching
        dominant_factor = 1.0
        if len(dominant_colors) > 0:
            for color in dominant_colors:
                color_dist = np.sqrt((color[2] - r_thresh)**2 + (color[1] - g_thresh)**2 + (color[0] - b_thresh)**2)
                if color_dist < 50:  # Close match
                    dominant_factor = 1.3
                    break
        
        return base_score * texture_factor * dominant_factor
    
    def save_detection_to_database(self, plate_text, material, yolo_conf, ocr_conf, processing_time, image_path=None, validation_status="valid"):
        """Save detection to enhanced database"""
        if not config.ENABLE_DATABASE:
            return None
            
        try:
            conn = sqlite3.connect(config.DATABASE_FILE)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, plate_number, material_type, confidence, ocr_confidence, 
                 image_path, camera_id, processing_time, validation_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                plate_text,
                material,
                yolo_conf,
                ocr_conf,
                image_path,
                'main_camera',
                processing_time,
                validation_status
            ))
            
            detection_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return detection_id
            
        except Exception as e:
            logging.error(f"Database save error: {e}")
            return None
    
    def send_to_clipboard(self, text):
        """Enhanced clipboard functionality"""
        try:
            pyperclip.copy(text)
            if config.DEBUG_MODE:
                print(f"📋 Copied to clipboard: {text}")
            logging.info("Detection copied to clipboard")
            return True
        except Exception as e:
            logging.error(f"Clipboard error: {e}")
            return False
    
    def show_notification(self, title, message):
        """Enhanced notification system"""
        if not config.ENABLE_NOTIFICATIONS:
            return
            
        try:
            notification.notify(
                title=title,
                message=message,
                timeout=config.NOTIFICATION_DURATION,
                app_name=config.COMPANY_NAME
            )
        except Exception as e:
            logging.error(f"Notification error: {e}")
    
    def send_webhook(self, data):
        """Send detection data to webhook with retry logic"""
        if not config.ENABLE_API or not config.WEBHOOK_URL:
            return False
            
        for attempt in range(config.API_RETRY_ATTEMPTS):
            try:
                response = requests.post(
                    config.WEBHOOK_URL,
                    json=data,
                    timeout=config.API_TIMEOUT
                )
                response.raise_for_status()
                logging.info(f"Webhook sent successfully on attempt {attempt + 1}")
                return True
                
            except Exception as e:
                logging.error(f"Webhook attempt {attempt + 1} failed: {e}")
                if attempt < config.API_RETRY_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def is_duplicate(self, plate_text):
        """Enhanced duplicate detection"""
        current_time = time.time()
        if plate_text in self.last_detection_time:
            if current_time - self.last_detection_time[plate_text] < config.DUPLICATE_DETECTION_TIMEOUT:
                return True
        
        self.last_detection_time[plate_text] = current_time
        
        # Clean old entries
        cutoff_time = current_time - config.DUPLICATE_DETECTION_TIMEOUT * 2
        self.last_detection_time = {k: v for k, v in self.last_detection_time.items() if v > cutoff_time}
        
        return False
    
    def get_system_stats(self):
        """Get current system statistics"""
        total_frames = self.processing_stats['total_frames']
        detection_rate = (self.processing_stats['plates_detected'] / max(total_frames, 1)) * 100
        recognition_rate = (self.processing_stats['plates_recognized'] / max(self.processing_stats['plates_detected'], 1)) * 100
        ocr_success_rate = (self.processing_stats['ocr_successes'] / max(self.processing_stats['ocr_attempts'], 1)) * 100
        
        return {
            'total_frames': total_frames,
            'detections': self.detection_count,
            'detection_rate': detection_rate,
            'recognition_rate': recognition_rate,
            'ocr_success_rate': ocr_success_rate
        }
    
    def process_frame(self, frame):
        """Enhanced frame processing"""
        try:
            self.frame_count += 1
            process_start_time = time.time()
            
            # Detect license plate
            plate_text, plate_coords, confidence = self.detect_and_read_plate(frame)
            
            if not plate_text:
                # Periodic status update
                if config.DEBUG_MODE and self.frame_count % 30 == 0:
                    stats = self.get_system_stats()
                    print(f"📊 Status: {stats['total_frames']} frames, {stats['detections']} detections, {stats['recognition_rate']:.1f}% recognition rate")
                return frame
            
            # Check for duplicates
            if self.is_duplicate(plate_text):
                if config.DEBUG_MODE:
                    print(f"🔄 Duplicate detection ignored: {plate_text}")
                return frame
            
            self.detection_count += 1
            
            # Detect material
            material = self.detect_material_color(frame, plate_coords)
            
            processing_time = time.time() - process_start_time
            
            # Validate plate format
            validation_status = "valid" if self.validate_indian_plate(plate_text) else "questionable"
            
            # Create result text
            result_text = (f"Plate: {plate_text} | Material: {material} | "
                         f"Confidence: {confidence:.2f} | Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Enhanced console output
            print("="*60)
            print(f"🚗 VEHICLE DETECTED #{self.detection_count}")
            print(f"📱 Plate Number: {plate_text}")
            print(f"🏗️ Material: {material}")
            print(f"🎯 YOLO Confidence: {confidence:.3f}")
            print(f"⏱️ Processing Time: {processing_time:.2f}s")
            print(f"✅ Validation: {validation_status}")
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            
            logging.info(f"Vehicle detected: {result_text}")
            
            # Save detection image
            image_path = None
            if config.SAVE_DETECTIONS:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_plate = plate_text.replace(' ', '_').replace('/', '_')
                filename = f"detection_{self.detection_count:04d}_{timestamp}_{safe_plate}.jpg"
                image_path = Path(config.DETECTION_FOLDER) / filename
                
                # Draw detection info on saved image
                save_frame = frame.copy()
                if plate_coords:
                    x1, y1, x2, y2 = plate_coords
                    cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(save_frame, plate_text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                cv2.imwrite(str(image_path), save_frame)
                print(f"💾 Image saved: {filename}")
            
            # Save to database
            detection_id = self.save_detection_to_database(
                plate_text, material, confidence, 0.0, processing_time, 
                str(image_path), validation_status
            )
            
            # Copy to clipboard
            self.send_to_clipboard(result_text)
            
            # Show notification
            self.show_notification(
                f"{config.COMPANY_NAME} - Vehicle Detected",
                f"Plate: {plate_text}\nMaterial: {material}"
            )
            
            # Send webhook
            if config.ENABLE_API:
                webhook_data = {
                    'detection_id': detection_id,
                    'plate_number': plate_text,
                    'material': material,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'validation_status': validation_status,
                    'timestamp': datetime.now().isoformat(),
                    'system': f"{config.COMPANY_NAME} ANPR v{config.SYSTEM_VERSION}",
                    'location': 'Weighbridge'
                }
                self.send_webhook(webhook_data)
            
            # Draw detection on frame
            if plate_coords:
                x1, y1, x2, y2 = plate_coords
                # Detection rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Plate text
                text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.rectangle(frame, (x1, y1-35), (x1 + text_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, plate_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                
                # Confidence
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y2+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Material info
            cv2.putText(frame, f"Material: {material}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Detection count
            cv2.putText(frame, f"Detections: {self.detection_count}", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return frame
    
    def draw_interface(self, frame):
        """Enhanced system interface"""
        # Title bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(frame, f"{config.COMPANY_NAME} - Vehicle Recognition System v{config.SYSTEM_VERSION}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Camera info
        cv2.putText(frame, f"Camera: {config.CAMERA_NAME}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Statistics
        stats = self.get_system_stats()
        stats_text = (f"Frames: {stats['total_frames']} | Detections: {stats['detections']} | "
                     f"Recognition: {stats['recognition_rate']:.1f}%")
        cv2.putText(frame, stats_text, (10, frame.shape[0]-50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Status
        status = "ACTIVE" if self.running else "STOPPED"
        color = (0, 255, 0) if self.running else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def cleanup_resources(self):
        """Cleanup system resources"""
        try:
            # Save final statistics
            if config.ENABLE_DATABASE:
                conn = sqlite3.connect(config.DATABASE_FILE)
                cursor = conn.cursor()
                
                stats = self.get_system_stats()
                cursor.execute('''
                    INSERT INTO system_stats 
                    (date, total_frames, detections, success_rate, avg_processing_time)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().strftime("%Y-%m-%d"),
                    stats['total_frames'],
                    stats['detections'],
                    stats['recognition_rate'],
                    0.0  # Could calculate average processing time
                ))
                
                conn.commit()
                conn.close()
            
            logging.info("Resources cleaned up successfully")
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
    
    def run(self):
        """Enhanced main system loop"""
        self.running = True
        
        print("="*80)
        print(f"    {config.COMPANY_NAME.upper()} - VEHICLE RECOGNITION SYSTEM v{config.SYSTEM_VERSION}")
        print("="*80)
        print(f"📹 Camera: {config.CAMERA_NAME}")
        print(f"🤖 YOLO Confidence: {config.MODEL_CONFIDENCE}")
        print(f"📖 OCR Confidence: {config.OCR_CONFIDENCE}")
        print(f"🐛 Debug Mode: {'Enabled' if config.DEBUG_MODE else 'Disabled'}")
        print(f"💾 Database: {'Enabled' if config.ENABLE_DATABASE else 'Disabled'}")
        print(f"🔗 API Integration: {'Enabled' if config.ENABLE_API else 'Disabled'}")
        print(f"🔍 Multiple OCR Engines: {'Enabled' if config.USE_MULTIPLE_OCR_ENGINES else 'Disabled'}")
        print(f"✅ Plate Validation: {'Enabled' if config.ENABLE_PLATE_VALIDATION else 'Disabled'}")
        print("="*80)
        print("🎯 System Status:")
        print("   • Scanning for license plates...")
        print("   • Detections will appear in real-time")
        print("   • Results automatically copied to clipboard")
        print("="*80)
        print("⌨️ Controls:")
        print("   • Press 'q' to quit")
        print("   • Press 's' for screenshot")
        print("   • Press 't' to test OCR on saved images")
        print("   • Press 'r' to reset statistics")
        print("="*80)
        
        # Open camera
        cap = cv2.VideoCapture(config.CAMERA_URL)
        
        if not cap.isOpened():
            logging.error(f"Failed to open camera: {config.CAMERA_URL}")
            print(f"❌ ERROR: Cannot open camera at {config.CAMERA_URL}")
            print("💡 Check:")
            print("   • Camera URL in config.py")
            print("   • Network connection")
            print("   • Camera permissions")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logging.info("Camera opened successfully")
        print("✅ Camera connected successfully")
        print("🚀 System is now running...")
        print("="*80)
        
        last_process_time = time.time()
        last_cleanup_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    logging.warning("Failed to read frame")
                    print("⚠️ Frame read failed, retrying...")
                    time.sleep(1)
                    continue
                
                # Process frame at intervals
                if time.time() - last_process_time >= config.FRAME_INTERVAL:
                    frame = self.process_frame(frame)
                    last_process_time = time.time()
                
                # Memory cleanup
                if time.time() - last_cleanup_time >= config.MEMORY_CLEANUP_INTERVAL:
                    if len(self.last_detection_time) > 100:
                        # Keep only recent detections
                        cutoff_time = time.time() - config.DUPLICATE_DETECTION_TIMEOUT * 2
                        self.last_detection_time = {k: v for k, v in self.last_detection_time.items() if v > cutoff_time}
                    last_cleanup_time = time.time()
                
                # Draw interface
                frame = self.draw_interface(frame)
                
                # Display frame
                cv2.imshow(f"{config.COMPANY_NAME} - Vehicle Recognition System", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                elif key == ord('t'):
                    print("🧪 Testing OCR on saved images...")
                    # You could implement the test function here
                elif key == ord('r'):
                    print("🔄 Statistics reset")
                    self.processing_stats = {key: 0 for key in self.processing_stats}
                    self.detection_count = 0
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupt received, shutting down...")
            
        finally:
            self.running = False
            self.cleanup_resources()
            cap.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            stats = self.get_system_stats()
            print("\n" + "="*60)
            print("📊 FINAL STATISTICS")
            print("="*60)
            print(f"Total Frames Processed: {stats['total_frames']}")
            print(f"Total Detections: {stats['detections']}")
            print(f"Detection Rate: {stats['detection_rate']:.2f}%")
            print(f"Recognition Rate: {stats['recognition_rate']:.2f}%")
            print(f"OCR Success Rate: {stats['ocr_success_rate']:.2f}%")
            print("="*60)
            
            logging.info("System stopped successfully")
            print("✅ System stopped successfully!")
            print(f"📁 Logs saved to: logs/{config.LOG_FILE}")
            if config.SAVE_DETECTIONS:
                print(f"📁 Detections saved to: {config.DETECTION_FOLDER}/")

def main():
    """Main entry point"""
    print(f"🚀 Starting {config.COMPANY_NAME} Vehicle Recognition System...")
    
    try:
        system = VehicleRecognitionSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        logging.error(f"System error: {e}")
        
        # Emergency error information
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check camera connection and URL in config.py")
        print("2. Ensure all required packages are installed: pip install -r requirements.txt")
        print("3. Check system logs in logs/ folder")
        print("4. Verify camera permissions and network access")

if __name__ == "__main__":
    main()
