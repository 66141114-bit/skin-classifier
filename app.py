import base64
import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

# แก้ไขปัญหา MediaPipe บน Windows ด้วยการ Import ตรงจาก python solutions
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
except (ImportError, AttributeError):
    try:
        import mediapipe.solutions.face_mesh as mp_face_mesh
    except (ImportError, AttributeError):
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh

app = Flask(__name__, template_folder='.')
CORS(app)

# ==========================================
# Roboflow Configuration
# ==========================================
ROBOFLOW_API_KEY = "LrjGaxGgS3S2Fy1Wndo2"
CLASSIFY_URL = f"https://classify.roboflow.com/facial-skin-classification-q6yym/1?api_key={ROBOFLOW_API_KEY}&confidence=0"

# พิกัด Face Mesh 468 จุดของแต่ละโซนบนใบหน้า
LANDMARK_ZONES = {
    "forehead": [10, 67, 103, 104, 108, 109, 297, 333, 337, 338, 151],
    "cheek_left": [116, 117, 118, 123, 147, 187, 205, 206, 207],
    "cheek_right": [345, 346, 347, 352, 376, 411, 425, 426, 427],
    "chin": [152, 175, 199, 200, 18, 17, 377, 396, 400]
}

# กำหนดตัวตรวจจับ Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def base64_to_cv2(b64_string):
    """แปลง Base64 string ให้เป็น OpenCV BGR Image"""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def cv2_to_base64(image_cv2):
    """แปลง OpenCV BGR Image ให้เป็น Base64 Data URL"""
    _, buffer = cv2.imencode('.jpg', image_cv2, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def crop_zone(image, landmarks, indices, pad_ratio=0.15, target_size=(224, 224)):
    """คำนวณ Bounding Box และตัดชิ้นส่วนผิว พร้อมปรับขนาดตามมาตรฐาน"""
    h, w, _ = image.shape
    pts = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in indices])
    
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)

    bw = x_max - x_min
    bh = y_max - y_min
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    cropped = image[y1:y2, x1:x2]
    if cropped.size > 0:
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    else:
        resized = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    return resized, (x1, y1, x2, y2)

@app.route('/')
def home():
    return render_template('index.html')

# ให้บริการไฟล์รูปภาพโปสเตอร์จากโฟลเดอร์ image/
@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory('image', filename)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "ไม่พบข้อมูลรูปภาพ"}), 400

    img = base64_to_cv2(data['image'])
    if img is None:
        return jsonify({"success": False, "error": "รูปภาพไม่ถูกต้อง"}), 400

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if not results.multi_face_landmarks:
        return jsonify({"success": False, "error": "ไม่พบใบหน้าในภาพ กรุณาจัดหน้าให้อยู่ในกรอบ"}), 200

    landmarks = results.multi_face_landmarks[0].landmark
    annotated_img = img.copy()

    crops_b64 = {}
    zone_colors = {
        "forehead": ((244, 114, 182), "Forehead"),
        "cheek_left": ((219, 39, 119), "Left Cheek"),
        "cheek_right": ((219, 39, 119), "Right Cheek"),
        "chin": ((168, 85, 247), "Chin")
    }

    for zone_name, indices in LANDMARK_ZONES.items():
        cropped_patch, box = crop_zone(img, landmarks, indices)
        crops_b64[zone_name] = cv2_to_base64(cropped_patch)

        color, label = zone_colors[zone_name]
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ส่งภาพแก้มซ้ายไปยัง Roboflow API
    primary_crop_b64 = crops_b64["cheek_left"].split(",")[1]
    roboflow_result = {}

    try:
        rf_response = requests.post(
            CLASSIFY_URL,
            data=primary_crop_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        if rf_response.ok:
            roboflow_result = rf_response.json()
        else:
            roboflow_result = {"error": f"Roboflow HTTP {rf_response.status_code}"}
    except Exception as e:
        roboflow_result = {"error": str(e)}

    return jsonify({
        "success": True,
        "clean_image": cv2_to_base64(img),
        "annotated_image": cv2_to_base64(annotated_img),
        "crops": crops_b64,
        "predictions": roboflow_result
    })

if __name__ == '__main__':
    print("🚀 AuraSkin AI Server เริ่มทำงานแล้ว: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)