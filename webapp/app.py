
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, abort
import sqlite3, os, time, uuid, cv2, datetime
from pathlib import Path
from werkzeug.utils import secure_filename
import sys

# Add the src directory to path for imports
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))

from enhanced_plate_detection import detect_plate_enhanced, crop_plate
from plate_detection import detect_plate_contour  # Keep for fallback
from enhanced_ocr import read_plate_with_confidence, EnhancedOCR
from ocr_recognition import OCRReader
from database import init_db, insert_plate
from smart_parking import SmartParkingSystem, VIOLATION_TYPES, VEHICLE_TYPES
from robust_plate_pipeline import RobustPlatePipeline

try:
    from yolo_detector import YOLODetector
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-here'

# Initialize Smart Parking System
parking_system = SmartParkingSystem()

# Ensure upload directory exists
upload_dir = Path(__file__).resolve().parent / 'uploads'
upload_dir.mkdir(exist_ok=True)

DB = Path(__file__).resolve().parents[1] / 'results' / 'plates.db'
CHALLAN_DIR = Path(__file__).resolve().parents[1] / 'results' / 'challans'
CHALLAN_DIR.mkdir(parents=True, exist_ok=True)
results_dir = Path(__file__).resolve().parents[1] / 'results' / 'detected_images'
results_dir.mkdir(parents=True, exist_ok=True)
# Debug directory to store attempted crops even on failure
DEBUG_SAVE = True
debug_dir = Path(__file__).resolve().parents[1] / 'results' / 'debug'
debug_dir.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_plate_from_image(image_path, detector='yolo', weights=None, ocr_backend='easyocr'):
    """Detect plate from uploaded image and return plate text and processed image path"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None, "Failed to read image"
        # Safety: downscale very large uploads to speed up processing
        try:
            h, w = image.shape[:2]
            max_dim = 1600
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        except Exception:
            pass

        # 1) Try YOLO license-plate model
        plate_img = None
        detection_method_used = None
        yolo_weights = weights or str((Path(__file__).resolve().parents[1] / 'models' / 'yolov8n-license-plate.pt'))
        if _HAS_YOLO and os.path.exists(yolo_weights):
            try:
                yd = YOLODetector(weights_path=yolo_weights, conf=0.25, iou=0.5)
                dets = yd.detect(image)
                if dets:
                    best = max(dets, key=lambda d: d['conf'])
                    x1, y1, x2, y2 = best['xyxy']
                    # Padding
                    ph, pw = image.shape[:2]
                    pad = int(0.12 * max(ph, pw))
                    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                    x2 = min(pw - 1, x2 + pad); y2 = min(ph - 1, y2 + pad)
                    plate_img = image[y1:y2, x1:x2]
                    detection_method_used = 'yolo'
            except Exception as e:
                print(f"YOLO detection failed: {e}")

        # 2) If YOLO missing/failed, use robust contour-based pipeline
        text = ''
        confidence = 0.0
        if plate_img is None:
            try:
                pipeline = RobustPlatePipeline(languages=['en'])
                text, box, plate2, confidence, info = pipeline.process_image(image)
                if box is not None and plate2 is not None:
                    plate_img = plate2
                    detection_method_used = f"robust:{info.get('detection_method','enhanced')}"
            except Exception as e:
                print(f"Robust pipeline failed: {e}")

        # 3) As last resort, enhanced/contour
        if plate_img is None:
            try:
                box = detect_plate_enhanced(image)
                if box is not None:
                    plate_img = crop_plate(image, box)
                    detection_method_used = 'enhanced'
            except Exception:
                pass
            if plate_img is None:
                try:
                    box = detect_plate_contour(image)
                    if box is not None:
                        plate_img = crop_plate(image, box)
                        detection_method_used = 'contour_fallback'
                except Exception:
                    pass
        if plate_img is None:
            return None, None, 'No plate detected in image'

        # OCR (use existing text/confidence if provided by robust)
        ocr_method_used = 'enhanced'
        if not text:
            try:
                text, confidence = read_plate_with_confidence(plate_img)
            except Exception as e:
                print(f"Enhanced OCR failed: {e}")
                try:
                    reader = OCRReader(backend=ocr_backend)
                    text = reader.read_plate(plate_img)
                    confidence = 70.0
                    ocr_method_used = ocr_backend
                except Exception:
                    text, confidence = '', 0.0

        cleaned = (text or '').strip().upper().replace(' ', '')
        if not cleaned or len(cleaned) < 5:
            # Save debug crop
            try:
                dbg_path = debug_dir / f"poor_quality_{int(time.time())}_{os.path.basename(image_path)}"
                cv2.imwrite(str(dbg_path), plate_img)
            except Exception:
                pass
            return None, None, f"Poor quality text detected: '{text}' (confidence: {confidence:.1f}%)"
        text = cleaned

        # Save cropped plate image
        filename = f"detected_{int(time.time())}_{os.path.basename(image_path)}"
        out_path = results_dir / filename
        cv2.imwrite(str(out_path), plate_img)

        # Save to DB
        init_db()
        insert_plate(text, datetime.datetime.now().isoformat(), speed=None, image_path=str(out_path))

        success_message = {
            'text': text,
            'confidence': confidence,
            'detection_method': detection_method_used or 'unknown',
            'ocr_method': ocr_method_used
        }
        return text, str(out_path), None, success_message
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"

def query_db(limit=100):
    if not DB.exists():
        return []
    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()
    cur.execute('SELECT id, plate, timestamp, speed, image_path FROM plates ORDER BY id DESC LIMIT ?', (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No file selected')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{filename}"
            filepath = upload_dir / filename
            file.save(str(filepath))
            
            # Detect plate
            detector = request.form.get('detector', 'yolo')
            ocr_backend = request.form.get('ocr', 'easyocr')
            result = detect_plate_from_image(
                filepath, detector=detector, ocr_backend=ocr_backend
            )
            
            # Handle different return formats
            if len(result) == 4:
                plate_text, processed_image, error, success_info = result
            else:
                plate_text, processed_image, error = result
                success_info = None
            
            if error:
                return render_template('upload.html', error=error, uploaded_image=f'/uploads/{filename}')
            elif plate_text:
                # Create enhanced success message
                if success_info:
                    confidence = success_info.get('confidence', 0)
                    detection_method = success_info.get('detection_method', 'unknown')
                    ocr_method = success_info.get('ocr_method', 'unknown')
                    success_msg = f'Plate detected: {plate_text} (Confidence: {confidence:.1f}%, Detection: {detection_method}, OCR: {ocr_method})'
                else:
                    success_msg = f'Plate detected: {plate_text}'
                    
                return render_template('upload.html', 
                                       success=success_msg,
                                       uploaded_image=f'/uploads/{filename}',
                                       processed_image=f'/static-results/{os.path.basename(processed_image)}',
                                       plate_text=plate_text,
                                       detection_info=success_info)
            else:
                return render_template('upload.html', 
                                       error='No plate detected in the image',
                                       uploaded_image=f'/uploads/{filename}')
        else:
            return render_template('upload.html', error='Invalid file type. Please upload an image file.')
    
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    rows = query_db(100)
    return render_template('dashboard.html', rows=rows)

@app.route('/api/plates', methods=['GET'])
def api_plates():
    rows = query_db(500)
    data = []
    for r in rows:
        data.append({'id': r[0], 'plate': r[1], 'timestamp': r[2], 'speed': r[3], 'image': (r[4].split('/')[-1] if r[4] else None)})
    return jsonify(data)

@app.route('/challan/create', methods=['POST'])
def create_challan():
    # Expected form data: plate, fine_amount (optional), reason (optional)
    plate = request.form.get('plate')
    fine = request.form.get('fine', '0')
    reason = request.form.get('reason', 'Traffic violation')
    if not plate:
        return jsonify({'error': 'plate required'}), 400
    # create a simple challan id and HTML file
    challan_id = str(uuid.uuid4())[:8]
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    challan_html = render_template('challan.html', challan_id=challan_id, plate=plate, fine=fine, reason=reason, timestamp=timestamp)
    path = CHALLAN_DIR / f'challan_{challan_id}.html'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(challan_html)
    return jsonify({'challan_id': challan_id, 'url': f'/challan/{challan_id}'}), 201

@app.route('/challan/<challan_id>')
def view_challan(challan_id):
    path = CHALLAN_DIR / f'challan_{challan_id}.html'
    if not path.exists():
        abort(404)
    return send_from_directory(str(CHALLAN_DIR), path.name)

@app.route('/static-results/<path:filename>')
def static_results(filename):
    results_path = Path(__file__).resolve().parents[1] / 'results' / 'detected_images'
    return send_from_directory(str(results_path), filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(str(upload_dir), filename)

# simple route to stream an output video (served as static file for now)
@app.route('/video/<path:filename>')
def stream_video(filename):
    video_path = Path('results') / filename
    if not video_path.exists():
        abort(404)
    return send_from_directory('results', filename)

@app.route('/parking', methods=['GET'])
def parking_dashboard():
    """Smart parking management dashboard"""
    parking_status = parking_system.get_parking_status()
    
    # Calculate total spots and available spots
    total_spots = sum(zone[1] for zone in parking_status['zones'])
    available_spots = total_spots - parking_status['active_sessions']
    parking_status['total_spots'] = total_spots
    parking_status['available_spots'] = available_spots
    
    # Get active parking sessions
    import sqlite3
    conn = sqlite3.connect(str(PARKING_DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        SELECT ps.session_id, ps.plate_number, ps.entry_time, ps.vehicle_type, 
               ps.vehicle_type, pz.zone_name, psp.spot_number, 
               CASE WHEN ps.entry_time THEN 
                   CAST((julianday('now') - julianday(ps.entry_time)) * 24 AS INTEGER) || 'h ' ||
                   CAST(((julianday('now') - julianday(ps.entry_time)) * 24 * 60) % 60 AS INTEGER) || 'm'
               ELSE 'Active'
               END as duration
        FROM parking_sessions ps
        JOIN parking_zones pz ON ps.zone_id = pz.id
        LEFT JOIN parking_spots psp ON ps.spot_id = psp.id
        WHERE ps.is_active = 1
        ORDER BY ps.entry_time DESC
    ''')
    active_sessions = cursor.fetchall()
    conn.close()
    
    return render_template('parking_dashboard.html', 
                         parking_status=parking_status,
                         active_sessions=active_sessions)

@app.route('/parking/entry', methods=['POST'])
def parking_entry():
    """Handle vehicle entry to parking"""
    plate_number = request.form.get('plate_number')
    zone_name = request.form.get('zone_name')
    vehicle_type = request.form.get('vehicle_type', 'CAR')
    
    image_path = None
    if 'image' in request.files and request.files['image'].filename:
        file = request.files['image']
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"entry_{timestamp}_{filename}"
            filepath = upload_dir / filename
            file.save(str(filepath))
            image_path = str(filepath)
    
    # Process entry
    success, message = parking_system.vehicle_entry(plate_number, zone_name, vehicle_type, image_path)
    
    parking_status = parking_system.get_parking_status()
    total_spots = sum(zone[1] for zone in parking_status['zones'])
    available_spots = total_spots - parking_status['active_sessions']
    parking_status['total_spots'] = total_spots
    parking_status['available_spots'] = available_spots
    
    return render_template('parking_dashboard.html', 
                         parking_status=parking_status,
                         result_message=message,
                         result_success=success)

@app.route('/parking/exit', methods=['POST'])
def parking_exit():
    """Handle vehicle exit from parking"""
    plate_number = request.form.get('plate_number')
    
    image_path = None
    if 'image' in request.files and request.files['image'].filename:
        file = request.files['image']
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"exit_{timestamp}_{filename}"
            filepath = upload_dir / filename
            file.save(str(filepath))
            image_path = str(filepath)
    
    # Process exit
    success, result = parking_system.vehicle_exit(plate_number, image_path)
    
    parking_status = parking_system.get_parking_status()
    total_spots = sum(zone[1] for zone in parking_status['zones'])
    available_spots = total_spots - parking_status['active_sessions']
    parking_status['total_spots'] = total_spots
    parking_status['available_spots'] = available_spots
    
    message = result if isinstance(result, str) else result.get('message', 'Exit processed')
    parking_details = result if isinstance(result, dict) else None
    
    return render_template('parking_dashboard.html', 
                         parking_status=parking_status,
                         result_message=message,
                         result_success=success,
                         parking_details=parking_details)

@app.route('/parking/register', methods=['POST'])
def parking_register():
    """Register a vehicle for parking passes"""
    plate_number = request.form.get('plate_number')
    owner_name = request.form.get('owner_name')
    owner_phone = request.form.get('owner_phone')
    owner_email = request.form.get('owner_email')
    vehicle_type = request.form.get('vehicle_type', 'CAR')
    pass_type = request.form.get('pass_type', 'MONTHLY')
    
    # Set pass duration based on type
    pass_duration = {
        'DAILY': 1,
        'MONTHLY': 30,
        'VIP': 365
    }.get(pass_type, 30)
    
    success, message = parking_system.register_vehicle(
        plate_number, owner_name, owner_phone, owner_email, 
        vehicle_type, pass_type, pass_duration
    )
    
    parking_status = parking_system.get_parking_status()
    total_spots = sum(zone[1] for zone in parking_status['zones'])
    available_spots = total_spots - parking_status['active_sessions']
    parking_status['total_spots'] = total_spots
    parking_status['available_spots'] = available_spots
    
    return render_template('parking_dashboard.html', 
                         parking_status=parking_status,
                         result_message=message,
                         result_success=success)

@app.route('/api/parking/status')
def api_parking_status():
    """API endpoint for parking status"""
    return jsonify(parking_system.get_parking_status())

@app.route('/api/parking/history/<plate_number>')
def api_parking_history(plate_number):
    """API endpoint for vehicle parking history"""
    history = parking_system.get_vehicle_parking_history(plate_number)
    return jsonify({
        'plate_number': plate_number,
        'history': history
    })

from smart_parking import PARKING_DB_PATH

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
