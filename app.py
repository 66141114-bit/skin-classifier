import base64
from collections import Counter
import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

# จัดการการ Import MediaPipe Face Mesh ให้รองรับทั้ง Windows และระบบปฏิบัติการอื่น
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
# 1. การตั้งค่า Roboflow API
# ==========================================
ROBOFLOW_API_KEY = "LrjGaxGgS3S2Fy1Wndo2"
CLASSIFY_URL = f"https://classify.roboflow.com/facial-skin-classification-q6yym/1?api_key={ROBOFLOW_API_KEY}&confidence=0"

# พิกัด Face Mesh 468 จุด สำหรับมาร์กและตัดภาพผิวหนัง 4 โซน
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
    """แปลงข้อมูล Base64 string ให้เป็นภาพ OpenCV BGR"""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def cv2_to_base64(image_cv2):
    """แปลงภาพ OpenCV BGR ให้เป็น Base64 Data URL"""
    _, buffer = cv2.imencode('.jpg', image_cv2, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def crop_zone(image, landmarks, indices, pad_ratio=0.15, target_size=(224, 224)):
    """คำนวณ Bounding Box ของกลุ่ม Landmark และตัดภาพผิวเฉพาะจุดตามขนาดมาตรฐาน"""
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

def evaluate_overall_skin(zone_predictions):
    """
    คำนวณสัดส่วนคะแนนภาพรวม (Skin Profile) และวิเคราะห์สรุปสภาพผิวตามแนวโน้มจริง
    """
    scores_sum = {"normal": 0.0, "oily": 0.0, "dry": 0.0, "acne": 0.0}
    
    for zone_name, res in zone_predictions.items():
        if "error" in res:
            continue
        
        raw_preds = res.get("raw_predictions", {})
        found_raw = False
        
        if isinstance(raw_preds, list):
            for item in raw_preds:
                c = str(item.get("class", "")).lower()
                conf = float(item.get("confidence", 0.0))
                for k in scores_sum:
                    if k in c:
                        scores_sum[k] += conf
                        found_raw = True
        elif isinstance(raw_preds, dict) and raw_preds:
            for c, val in raw_preds.items():
                c_str = str(c).lower()
                conf = float(val.get("confidence", 0.0) if isinstance(val, dict) else val)
                for k in scores_sum:
                    if k in c_str:
                        scores_sum[k] += conf
                        found_raw = True

        if not found_raw:
            top_c = str(res.get("predicted_class", "")).lower()
            conf = float(res.get("confidence", 50.0))
            matched = False
            for k in scores_sum:
                if k in top_c:
                    scores_sum[k] += conf
                    matched = True
            if not matched:
                scores_sum["normal"] += conf

    total = sum(scores_sum.values())
    if total > 0:
        skin_profile = {k: round((v / total) * 100, 1) for k, v in scores_sum.items()}
    else:
        skin_profile = {"normal": 50.0, "oily": 20.0, "dry": 15.0, "acne": 15.0}

    # อ่านค่าของแต่ละโซน
    forehead = zone_predictions.get("forehead", {}).get("predicted_class", "").lower()
    chin = zone_predictions.get("chin", {}).get("predicted_class", "").lower()
    left_cheek = zone_predictions.get("cheek_left", {}).get("predicted_class", "").lower()
    right_cheek = zone_predictions.get("cheek_right", {}).get("predicted_class", "").lower()

    classes_detected = [v.get("predicted_class", "").lower() for v in zone_predictions.values() if "predicted_class" in v]
    normal_count = sum(1 for c in classes_detected if "normal" in c or "ปกติ" in c)
    acne_count = sum(1 for c in classes_detected if "acne" in c or "สิว" in c)

    # 1. เงื่อนไข: ผิวปกติชนะเด่นชัด (ตรวจเจอผิวปกติ 3 จุดขึ้นไป หรือมีคะแนน Profile สูงสุดและนำคลาสอื่น)
    if normal_count >= 3 or (skin_profile["normal"] >= 40.0 and skin_profile["normal"] > skin_profile["oily"]):
        return {
            "type_key": "normal",
            "overall_type": "Normal Skin (ผิวปกติ)",
            "summary_th": f"ผิวมีความสมดุลของน้ำและน้ำมันได้ดี สุขภาพผิวแข็งแรง (ตรวจพบผิวปกติ {normal_count}/4 ตำแหน่ง)",
            "recommendation": "ดูแลรักษาความชุ่มชื้นด้วยมอยส์เจอไรเซอร์พื้นฐาน และปกป้องผิวด้วยครีมกันแดดเป็นประจำทุกวัน",
            "profile": skin_profile
        }

    # 2. เงื่อนไข: สิวเด่นชัด (พบสิว 2 จุดขึ้นไป หรือสัดส่วนสิวเกิน 35%)
    if acne_count >= 2 or skin_profile["acne"] >= 35.0:
        return {
            "type_key": "acne",
            "overall_type": "Acne Skin (ผิวมีแนวโน้มเป็นสิวง่าย)",
            "summary_th": f"ตรวจพบปัญหาการเกิดสิวหรือการอักเสบใน {max(acne_count, 1)} ตำแหน่งบนใบหน้า",
            "recommendation": "ควรใช้คลีนเซอร์สูตรอ่อนโยนลดการอุดตัน และใช้ผลิตภัณฑ์แต้มสิวเฉพาะจุด",
            "profile": skin_profile
        }

    # 3. เงื่อนไข: ผิวผสม (ต้องมันทั้งหน้าผากและคางอย่างชัดเจน)
    t_oily = ("oily" in forehead) and ("oily" in chin)
    if t_oily and any(k in left_cheek or k in right_cheek for k in ["dry", "normal", "แห้ง", "ปกติ"]):
        return {
            "type_key": "combination",
            "overall_type": "Combination Skin (ผิวผสม)",
            "summary_th": "มีความมันสะสมบริเวณ T-Zone (หน้าผากและคาง) ขณะที่บริเวณแก้มมีความแห้งหรือปกติ",
            "recommendation": "บำรุงมอยส์เจอไรเซอร์เนื้อเบาบริเวณแก้ม และควบคุมความมันส่วนเกินบริเวณทีโซน",
            "profile": skin_profile
        }

    # 4. กรณีอื่นๆ ยึดตามคลาสที่มีคะแนนสะสมสูงสุด (Majority Vote)
    top_class = max(skin_profile, key=skin_profile.get)
    meta = {
        "oily": ("oily", "Oily Skin (ผิวมัน)", "ผิวหน้ามีความมันวาวและรูขุมขนกว้างทั่วทั้งใบหน้า", "ควรใช้สกินแคร์สูตร Oil-Free คลีนเซอร์เนื้อเจล และทำความสะอาดคราบมันอย่างสม่ำเสมอ"),
        "dry": ("dry", "Dry Skin (ผิวแห้ง)", "ผิวขาดความชุ่มชื้น มีแนวโน้มแห้งกร้านหรือเป็นขุย", "ควรใช้มอยส์เจอไรเซอร์เข้มข้นที่มี Ceramide หรือ Hyaluronic Acid หลีกเลี่ยงน้ำอุ่น"),
        "normal": ("normal", "Normal Skin (ผิวปกติ)", "ผิวมีความสมดุลของน้ำและน้ำมันที่ดี สุขภาพผิวแข็งแรง", "ดูแลด้วยมอยส์เจอไรเซอร์พื้นฐานและทาครีมกันแดดเป็นประจำทุกวัน"),
        "acne": ("acne", "Acne Skin (ผิวเป็นสิว)", "ตรวจพบการระคายเคืองและการเกิดสิวสะสม", "ควรเน้นปลอบประโลมผิว ลดการเสียดสี และรักษาความสะอาดอย่างถูกวิธี")
    }
    key, name, desc, rec = meta.get(top_class, ("normal", "Normal Skin (ผิวปกติ)", "ผิวมีความสมดุลแข็งแรง", "ดูแลด้วยมอยส์เจอไรเซอร์พื้นฐานและทากันแดดทุกวัน"))

    return {
        "type_key": key,
        "overall_type": name,
        "summary_th": desc,
        "recommendation": rec,
        "profile": skin_profile
    }

@app.route('/')
def home():
    return render_template('index.html')

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
        return jsonify({"success": False, "error": "รูปภาพไม่ถูกต้องหรือไม่สามารถประมวลผลได้"}), 400

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

    # ตัดภาพชิ้นส่วนทั้ง 4 โซน
    for zone_name, indices in LANDMARK_ZONES.items():
        cropped_patch, box = crop_zone(img, landmarks, indices)
        crops_b64[zone_name] = cv2_to_base64(cropped_patch)

        color, label = zone_colors[zone_name]
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ส่งวิเคราะห์ครบทั้ง 4 โซนไปยัง Roboflow API
    zone_predictions = {}
    for zone_name, b64_img in crops_b64.items():
        raw_b64 = b64_img.split(",")[1]
        try:
            rf_response = requests.post(
                CLASSIFY_URL,
                data=raw_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10
            )
            if rf_response.ok:
                res_data = rf_response.json()
                top_pred = res_data.get("top", "Unknown")
                conf = res_data.get("confidence", 0.0)
                zone_predictions[zone_name] = {
                    "predicted_class": top_pred,
                    "confidence": round(conf * 100, 1),
                    "raw_predictions": res_data.get("predictions", {})
                }
            else:
                zone_predictions[zone_name] = {"error": f"Roboflow HTTP {rf_response.status_code}"}
        except Exception as e:
            zone_predictions[zone_name] = {"error": str(e)}

    # คำนวณสรุปสภาพผิวภาพรวมและสัดส่วน Profile
    overall_evaluation = evaluate_overall_skin(zone_predictions)

    return jsonify({
        "success": True,
        "clean_image": cv2_to_base64(img),
        "annotated_image": cv2_to_base64(annotated_img),
        "crops": crops_b64,
        "zone_predictions": zone_predictions,
        "overall_evaluation": overall_evaluation
    })

if __name__ == '__main__':
    print("🚀 AuraSkin AI Server เริ่มทำงานแล้ว: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
