"""Image PII Detection - With MediaPipe"""
import cv2
import numpy as np
import os
import re
from PIL import Image, ImageDraw, ImageFilter
import pytesseract
from presidio_analyzer import AnalyzerEngine

# MediaPipe is optional: if it's not installed, the detector gracefully falls
# back to the Haar cascade implementation instead of failing to import.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None
    MEDIAPIPE_AVAILABLE = False

import json
from config import DATA_INPUT, DATA_OUTPUT

class ImagePIIDetector:
    def __init__(self):
        print("Initializing...")
        # Always define face_cascades so the Haar fallback works if MediaPipe
        # fails at runtime (not just at init).
        self.face_cascades = []
        for f in ['haarcascade_frontalface_default.xml', 'haarcascade_frontalface_alt2.xml']:
            try:
                c = cv2.CascadeClassifier(cv2.data.haarcascades + f)
                if not c.empty():
                    self.face_cascades.append(c)
            except:
                pass
        if not self.face_cascades:
            self.face_cascades = None

        # Initialize MediaPipe Face Detection (Tasks API) - dual-model strategy
        self.mediapipe_detectors = []
        self.use_mediapipe = False
        if MEDIAPIPE_AVAILABLE:
            models = [
                ('blaze_face_full_range.tflite', 'full_range'),
                ('blaze_face_short_range.tflite', 'short_range')
            ]
            for model_file, model_type in models:
                try:
                    model_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        'models', model_file
                    )
                    if not os.path.exists(model_path):
                        print(f"⚠ Model not found: {model_file}")
                        continue
                    base_options = mp_python.BaseOptions(model_asset_path=model_path)
                    options = mp_vision.FaceDetectorOptions(
                        base_options=base_options,
                        min_detection_confidence=0.3,
                        min_suppression_threshold=0.2
                    )
                    detector = mp_vision.FaceDetector.create_from_options(options)
                    self.mediapipe_detectors.append((detector, model_type))
                    print(f"✓ MediaPipe {model_type} model initialized")
                except Exception as e:
                    print(f"⚠ MediaPipe {model_type} model failed: {e}")
            if self.mediapipe_detectors:
                self.use_mediapipe = True
                print(f"✓ MediaPipe enabled with {len(self.mediapipe_detectors)} model(s)")
        else:
            print("MediaPipe not installed - using Haar cascades")
        
        try:
            self.text_analyzer = AnalyzerEngine()
        except:
            self.text_analyzer = None
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
    def detect_faces_opencv(self, image):
        """Detect faces using MediaPipe (primary) or Haar cascades (fallback)"""
        if self.use_mediapipe:
            return self._detect_faces_mediapipe(image)
        elif self.face_cascades is not None:
            return self._detect_faces_haar(image)
        else:
            return np.array([])
    
    def _preprocess_for_detection(self, image):
        """Preprocess image with CLAHE to improve face detection in low-light/high-contrast images"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            try:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception:
                return image

    def _detect_faces_mediapipe(self, image):
        """Detect faces using MediaPipe - dual-model strategy with Haar fallback"""
        try:
            img_h, img_w = image.shape[:2]
            print(f"\n=== MediaPipe Detection ({img_w}x{img_h}) ===")

            # Preprocess image (CLAHE) for better detection
            rgb_image = self._preprocess_for_detection(image)

            all_faces = []
            # Try each model (full_range first, then short_range)
            for detector, model_type in self.mediapipe_detectors:
                print(f"\nTrying {model_type} model...")
                try:
                    # Wrap the image for MediaPipe Tasks
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                    results = detector.detect(mp_image)
                except Exception as e:
                    print(f"  {model_type} model error: {e}")
                    continue

                faces = []
                rejected = {"aspect": 0, "size": 0, "area": 0}

                if results.detections:
                    print(f"  Raw detections: {len(results.detections)}")
                    for idx, detection in enumerate(results.detections):
                        # Get bounding box (absolute pixel coordinates)
                        bbox = detection.bounding_box
                        x = int(bbox.origin_x)
                        y = int(bbox.origin_y)
                        w = int(bbox.width)
                        h = int(bbox.height)

                        # Ensure coordinates are within image bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, img_w - x)
                        h = min(h, img_h - y)

                        # Skip invalid boxes
                        if w <= 0 or h <= 0:
                            continue

                        # RELAXED validation filters (fewer false negatives)
                        aspect = w / max(h, 1)
                        if aspect < 0.4 or aspect > 2.2:
                            rejected["aspect"] += 1
                            print(f"    Rejected {idx}: bad aspect ({aspect:.2f})")
                            continue

                        # Allow larger faces (close-up selfies)
                        if w > img_w * 0.85 or h > img_h * 0.85:
                            rejected["size"] += 1
                            print(f"    Rejected {idx}: too large")
                            continue
                        if w * h > img_w * img_h * 0.60:
                            rejected["area"] += 1
                            print(f"    Rejected {idx}: area too large")
                            continue

                        faces.append([x, y, w, h])
                        conf = detection.categories[0].score if detection.categories else 0
                        print(f"    ✓ Accepted {idx}: [{x}, {y}, {w}, {h}] conf={conf:.2f}")

                unique = self.remove_duplicate_faces(faces) if len(faces) > 1 else np.array(faces)
                print(
                    f"  {model_type}: {len(faces)} faces, {len(unique)} unique "
                    f"(rejected: aspect={rejected['aspect']}, size={rejected['size']}, area={rejected['area']})"
                )
                all_faces.extend(unique.tolist())

            # Combine results from all models and deduplicate
            if all_faces:
                final_faces = self.remove_duplicate_faces(all_faces)
                print(f"\n✓ MediaPipe final: {len(final_faces)} unique face(s)")
                return final_faces

            # NO FACES FOUND -> automatic Haar cascade fallback
            print("\n⚠ MediaPipe found 0 faces - trying Haar cascade fallback...")
            if self.face_cascades is not None:
                haar_faces = self._detect_faces_haar(image)
                if len(haar_faces) > 0:
                    print(f"✓ Haar fallback found {len(haar_faces)} face(s)")
                    return haar_faces

            print("✗ No faces detected by any method")
            return np.array([])

        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            import traceback
            traceback.print_exc()
            if self.face_cascades is not None:
                print("Falling back to Haar cascades...")
                return self._detect_faces_haar(image)
            return np.array([])
    def _detect_faces_haar(self, image):
        """Fallback face detection using Haar cascades"""
        if self.face_cascades is None:
            return np.array([])
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_h, img_w = image.shape[:2]
            preprocessed = {
                'eq': cv2.equalizeHist(gray),
                'clahe': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray),
            }
            all_faces = []
            
            max_face_size = min(img_w, img_h) // 2
            for cascade in self.face_cascades:
                for prep_img in preprocessed.values():
                    faces = cascade.detectMultiScale(
                        prep_img, 
                        scaleFactor=1.05, 
                        minNeighbors=3,
                        minSize=(30, 30),
                        maxSize=(max_face_size, max_face_size)
                    )
                    if len(faces) > 0:
                        all_faces.extend(faces)
            
            filtered = []
            for (x, y, fw, fh) in all_faces:
                if fw > img_w * 0.4 or fh > img_h * 0.4:
                    continue
                if fw * fh > img_w * img_h * 0.20:
                    continue
                aspect = fw / max(fh, 1)
                if aspect < 0.6 or aspect > 1.4:
                    continue
                if x < 0 or y < 0 or x + fw > img_w or y + fh > img_h:
                    continue
                filtered.append([x, y, fw, fh])
            
            unique = self.remove_duplicate_faces(filtered)
            print(f"Haar found {len(filtered)} faces, {len(unique)} unique")
            return unique
        except Exception as e:
            print(f"Haar detection error: {e}")
            return np.array([])
    
    def remove_duplicate_faces(self, faces):
        if len(faces) <= 1:
            return np.array(faces)
        faces_list = [list(f) for f in faces]
        unique = []
        for i, (x1, y1, w1, h1) in enumerate(faces_list):
            dup = False
            for j, (x2, y2, w2, h2) in enumerate(faces_list):
                if i != j:
                    ox = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                    oy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                    oa = ox * oy
                    if w1*h1 > 0 and oa / (w1*h1) > 0.5:
                        if w1*h1 <= w2*h2:
                            dup = True
                            break
            if not dup:
                unique.append([x1, y1, w1, h1])
        return np.array(unique) if unique else np.array([])
    
    def blur_faces(self, image, faces):
        if len(faces) == 0:
            return image
        blurred = image.copy()
        for i, (x, y, w, h) in enumerate(faces):
            try:
                # Expand box significantly to cover full face (hair, chin, ears, both eyes)
                # Haar cascades detect inner face, so we need generous padding
                px, py = int(w*0.50), int(h*0.50)  # Increased from 0.35/0.40 to 0.50/0.50
                x1, y1 = max(0, x-px), max(0, y-py)
                x2, y2 = min(image.shape[1], x+w+px), min(image.shape[0], y+h+py)
                region = blurred[y1:y2, x1:x2]
                if region.size > 0:
                    bs = max(51, min(w,h))  # Increased blur size for better coverage
                    if bs % 2 == 0: bs += 1
                    b = cv2.GaussianBlur(region, (bs, bs), 0)
                    b = cv2.GaussianBlur(b, (bs, bs), 0)
                    blurred[y1:y2, x1:x2] = b
            except Exception as e:
                print(f"Blur error: {e}")
        return blurred
    
    def pixelate_faces(self, image, faces, pixel_size=8):
        if len(faces) == 0:
            return image
        image_pixelated = image.copy()
        for i, (x, y, w, h) in enumerate(faces):
            try:
                # Expand box to cover full face (hair, chin, ears)
                padding_x = int(w * 0.30)
                padding_y = int(h * 0.35)
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(image.shape[1], x + w + padding_x)
                y2 = min(image.shape[0], y + h + padding_y)
                face_region = image_pixelated[y1:y2, x1:x2]
                if face_region.size > 0 and (x2-x1) > 0 and (y2-y1) > 0:
                    small_w = max(1, (x2-x1) // pixel_size)
                    small_h = max(1, (y2-y1) // pixel_size)
                    small = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    image_pixelated[y1:y2, x1:x2] = pixelated
            except Exception as e:
                print(f"Pixelate error: {e}")
        return image_pixelated
    
    def _process(self, image, filename, redaction_methods=None, write_output=True, original_path=None):
        """Core detection + redaction pipeline shared by process_image() and
        process_image_bytes(). Runs face detection and/or OCR-based PII
        detection, applies the requested redaction methods, and returns a
        results dictionary consumed by the local server.
        """
        results = {}
        try:
            results['filename'] = filename
            if original_path is not None:
                results['original_path'] = original_path
            results['redaction_methods'] = redaction_methods or ['blur_faces']
            results['image_dimensions'] = f"{image.shape[1]}x{image.shape[0]}"

            redacted_image = image.copy()
            detections = {}
            text_regions = []
            pii_regions = []

            if any(m in redaction_methods for m in ['blur_faces', 'pixelate_faces', 'black_box_faces']):
                faces = self.detect_faces_opencv(image)
                if len(faces) > 0:
                    detections['faces'] = len(faces)
                    if 'blur_faces' in redaction_methods:
                        redacted_image = self.blur_faces(redacted_image, faces)
                    elif 'pixelate_faces' in redaction_methods:
                        redacted_image = self.pixelate_faces(redacted_image, faces)
                    elif 'black_box_faces' in redaction_methods:
                        face_regions = [{'bbox': (x, y, w, h), 'label': 'face'} for x, y, w, h in faces]
                        redacted_image = self.black_box_regions(redacted_image, face_regions, 'face')

            if any(m in redaction_methods for m in ['blur_text', 'pixelate_text', 'redact_text']):
                text_regions = self.extract_text_from_image(image)
                pii_regions = self.analyze_text_for_pii(text_regions)
                if pii_regions:
                    detections['text_pii_regions'] = len(pii_regions)
                    if 'blur_text' in redaction_methods:
                        redacted_image = self.blur_text_regions(redacted_image, pii_regions)
                    elif 'pixelate_text' in redaction_methods:
                        redacted_image = self.pixelate_text_regions(redacted_image, pii_regions)
                    elif 'redact_text' in redaction_methods:
                        text_regions_list = [{'bbox': r['bbox']} for r in pii_regions]
                        redacted_image = self.black_box_regions(redacted_image, text_regions_list, 'text')

            # Format results for the local server response.
            results['faces_detected'] = detections.get('faces', 0)
            results['text_regions'] = text_regions
            results['pii_text_regions'] = pii_regions
            results['redacted_image'] = redacted_image
            results['redacted_versions'] = {}

            # Store redacted image versions
            if redaction_methods:
                for method in redaction_methods:
                    if 'face' in method or 'text' in method:
                        results['redacted_versions'][method] = redacted_image

            if write_output:
                output_filename = f"redacted_{filename}"
                output_path = os.path.join(DATA_OUTPUT, 'images', output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, redacted_image)
                results['output_path'] = output_path

            results['status'] = 'success'

        except Exception as e:
            results['error'] = str(e)
            results['status'] = 'error'

        return results

    def process_image(self, image_path, redaction_methods=None):
        """Detect and redact PII in an image file on disk."""
        results = {}
        try:
            image = cv2.imread(image_path)
            if image is None:
                results['error'] = f"Could not load image: {image_path}"
                results['status'] = 'error'
                return results
            return self._process(
                image,
                os.path.basename(image_path),
                redaction_methods,
                write_output=True,
                original_path=image_path,
            )
        except Exception as e:
            results['error'] = str(e)
            results['status'] = 'error'
            return results

    def process_image_bytes(self, img_bytes, redaction_methods=None, filename='upload.png'):
        """Run the full detection + redaction pipeline on raw image bytes.

        Used by the design site's API (server.py) so images uploaded from the
        browser can be processed without writing them to disk first.
        """
        results = {}
        try:
            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                results['error'] = 'Could not decode image bytes'
                results['status'] = 'error'
                return results
            return self._process(image, filename, redaction_methods, write_output=False)
        except Exception as e:
            results['error'] = str(e)
            results['status'] = 'error'
            return results

    def generate_detection_report(self, results):
        report = f"""
IMAGE PROCESSING REPORT
{'='*60}

FILE: {results.get('filename', 'Unknown')}
DIMENSIONS: {results.get('image_dimensions', 'Unknown')}
STATUS: {results.get('status', 'Unknown').upper()}

"""
        if 'error' in results:
            report += f"ERROR: {results['error']}\n"
            return report
        
        detections = results.get('detections', {})
        if detections:
            report += "PII DETECTED:\n"
            if 'faces' in detections:
                report += f"- Faces detected: {detections['faces']}\n"
            if 'text_pii_regions' in detections:
                report += f"- Text PII regions: {detections['text_pii_regions']}\n"
        else:
            report += "No PII detected.\n"
        
        report += f"\nOUTPUT: {results.get('output_path', 'Not saved')}\n"
        report += f"{'='*60}\n"
        
        return report
    
    def black_box_regions(self, image, regions, region_type="face"):
        image_boxed = image.copy()
        for i, region in enumerate(regions):
            try:
                if region_type == "face":
                    x, y, w, h = region
                else:
                    x, y, w, h = region['bbox']
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
                image_boxed[y1:y2, x1:x2] = (0, 0, 0)
            except Exception as e:
                print(f"Black box error: {e}")
        return image_boxed
    
    def extract_text_from_image(self, image):
        text_regions = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Use image_to_data to get bounding boxes for each text element
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            n_boxes = len(data['text'])
            
            # Collect all text elements with their positions
            elements = []
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                if text and conf > 40:  # Increased from 30 to reduce false positives
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    # Filter out tiny or huge boxes - increased minimum size
                    if w >= 8 and h >= 8 and w < image.shape[1] * 0.8 and h < image.shape[0] * 0.8:
                        elements.append({
                            'text': text,
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'bbox': (x, y, w, h),
                            'confidence': conf / 100.0
                        })
            
            # Group nearby elements into lines/regions to avoid single-character blurring
            if elements:
                # Sort by y then x position
                elements.sort(key=lambda e: (e['y'], e['x']))
                
                # Group elements that are on the same line (similar y coordinate)
                lines = []
                current_line = []
                line_y = elements[0]['y']
                for elem in elements:
                    # If element is within 20px vertically, consider it same line
                    if abs(elem['y'] - line_y) < 20:
                        current_line.append(elem)
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = [elem]
                        line_y = elem['y']
                if current_line:
                    lines.append(current_line)
                
                # Merge each line into a single region
                for line in lines:
                    if len(line) == 1:
                        # Single word/character - only include if it's long enough to avoid false positives
                        elem = line[0]
                        if len(elem['text']) >= 3 and elem['confidence'] > 0.5:  # At least 3 chars and 50% confidence
                            text_regions.append(elem)
                    else:
                        # Multiple elements on same line - merge them
                        min_x = min(e['x'] for e in line)
                        min_y = min(e['y'] for e in line)
                        max_x = max(e['x'] + e['w'] for e in line)
                        max_y = max(e['y'] + e['h'] for e in line)
                        merged_text = ' '.join(e['text'] for e in line)
                        # Only include merged regions if they have meaningful content
                        if len(merged_text.strip()) >= 3:
                            text_regions.append({
                                'text': merged_text,
                                'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                                'confidence': min(e['confidence'] for e in line)
                            })
            
            # Fallback to full image text if no regions found
            if not text_regions:
                text = pytesseract.image_to_string(thresh)
                if text.strip():
                    text_regions.append({
                        'text': text.strip(),
                        'bbox': (0, 0, image.shape[1], image.shape[0]),
                        'confidence': 0.8
                    })
        except Exception as e:
            print(f"Text extraction error: {e}")
        return text_regions
    
    def analyze_text_for_pii(self, text_regions):
        pii_text_regions = []
        pii_patterns = {
            'email': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'phone': [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'],
            'ssn': [r'\b\d{3}-\d{2}-\d{4}\b'],
            'credit_card': [r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'],
        }
        for region in text_regions:
            text = region['text']
            is_pii = False
            detected_types = []
            
            # Skip very short text (single characters, 1-2 chars) to avoid false positives
            # Only check PII patterns for text that's at least 3 characters
            if len(text.strip()) >= 3:
                for pii_type, patterns in pii_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            is_pii = True
                            detected_types.append(pii_type.upper())
                            break
            
            # Only use Presidio for text that's at least 3 characters and with higher threshold
            if len(text.strip()) >= 3 and self.text_analyzer is not None:
                try:
                    pii_results = self.text_analyzer.analyze(text=text, language='en', score_threshold=0.7)  # Increased from 0.5
                    if pii_results:
                        is_pii = True
                        for result in pii_results:
                            if result.entity_type not in detected_types:
                                detected_types.append(result.entity_type)
                except Exception as e:
                    print(f"Presidio analysis failed: {e}")
            if is_pii:
                region['pii_detected'] = True
                region['pii_types'] = list(set(detected_types))
                pii_text_regions.append(region)
        return pii_text_regions
    
    def blur_text_regions(self, image, regions):
        image_blurred = image.copy()
        for region in regions:
            try:
                x, y, w, h = region['bbox']
                padding = 5
                x1, y1 = max(0, x-padding), max(0, y-padding)
                x2, y2 = min(image.shape[1], x+w+padding), min(image.shape[0], y+h+padding)
                text_region = image_blurred[y1:y2, x1:x2]
                if text_region.size > 0:
                    blurred_region = cv2.GaussianBlur(text_region, (25, 25), 0)
                    image_blurred[y1:y2, x1:x2] = blurred_region
            except Exception as e:
                print(f"Text blur failed: {e}")
        return image_blurred
    
    def pixelate_text_regions(self, image, regions, pixel_size=10):
        image_pixelated = image.copy()
        for region in regions:
            try:
                x, y, w, h = region['bbox']
                padding = 3
                x1, y1 = max(0, x-padding), max(0, y-padding)
                x2, y2 = min(image.shape[1], x+w+padding), min(image.shape[0], y+h+padding)
                text_region = image_pixelated[y1:y2, x1:x2]
                if text_region.size > 0 and (x2-x1) > 0 and (y2-y1) > 0:
                    small_w = max(1, (x2-x1) // pixel_size)
                    small_h = max(1, (y2-y1) // pixel_size)
                    small = cv2.resize(text_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    pixelated_region = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    image_pixelated[y1:y2, x1:x2] = pixelated_region
            except Exception as e:
                print(f"Text pixelation failed: {e}")
        return image_pixelated
