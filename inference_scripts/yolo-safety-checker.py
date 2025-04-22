import cv2
import paho.mqtt.client as mqtt
import numpy as np
from ultralytics import YOLO
import time
import subprocess
import config.camera_creds as cam
import config.mqtt_topics as mqtt_top
import config.blue_iris_settings as bi
import utilities.http_requests_bi as bi_request
import config.mqtt_settings as mqtt_set
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Parametry pro sledování a alarmy
ALERT_THRESHOLD = 175          # Počet po sobě jdoucích framů bez správného vybavení, než se řekne: "KURVA, pozor!"
MAX_MISSED_FRAMES = 15         # Pokud stopa zmizí na příliš dlouho, smažeme ji
CENTER_MARGIN = 0.2            # Definice centrální oblasti

# Otevři video zdroj – RTSP kamera nebo soubor
RTSP_CAM_3 = f"rtsp://{cam.USER}:{cam.PASSWORD}@{cam.CAMERA_IP[3]}:{cam.RTSP_PORT}{cam.HIKVISION_MAIN_STREAM_URL}"
PROXY_CAM = "rtsp://pco:VezeBlackTorch4@192.168.88.6:8554/camera1"

# Načti YOLO model (vyměň cestu, jak se ti to hodí)
model = YOLO("/home/sidsm/Projects/YOLO-preprocessing-and-training/datasets/dataset-2/runs/detect/train/weights/best.pt").to("cuda")

cap = cv2.VideoCapture(RTSP_CAM_3, cv2.CAP_FFMPEG)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Snižme to zbytečné cachování, ať se to nabuší
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
cap.set(cv2.CAP_PROP_FPS, 15) 
# Získej rozměry vstupního rámce
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Vstupní rozměry: {frame_width}x{frame_height}")
# outputs 2560x1440

# Spusť FFmpeg, který bude číst surový BGR stream a posílat ho jako RTSP
# Uprav si 'blueiris_username', 'blueiris_password' a 'blueiris_server' podle svého
output_width = 640#1280
output_height = int(frame_height * (output_width / frame_width))
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    #'-vcodec', 'rawvideo',
    #'-pix_fmt', 'bgr24',
    '-pix_fmt', 'yuv420p',
    '-s', f'{output_width}x{output_height}',
    '-r', '15',  # FPS
    '-i', '-',  # Vstupní data z pipe
    '-c:v', 'libx264',  # Použijeme H.264 enkodér
    '-pix_fmt', 'yuv420p',  # Barevný prostor, aby byl kompatibilní všude
    '-preset', 'ultrafast',  # Nejrychlejší enkódování
    #'-probesize', '100M',
    #'-analyzeduration', '100M',
    '-f', 'rtsp',  # Výstupní formát je RTSP
    '-rtsp_transport', 'tcp',  # Použijeme stabilnější TCP transport
    'rtsp://admin:admin@192.168.88.6:8554/ai_cam'  # Streamujeme na náš RTSP server
]
ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Skladujeme sledované osoby podle track_id, který nám přiřídí ByteTrack
tracked_persons = {}

def compute_iou(box1, box2):
    """Spočítej IoU (Intersection over Union) mezi dvěma boxy [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)

def is_equipped(person_box, equipment_boxes):
    """Zjisti, jestli se v zóně osoby nachází nějaký kus vybavení (vest nebo helma)."""
    for eq_box in equipment_boxes:
        if compute_iou(person_box, eq_box) > 0.04:  # Prahová hodnota, případně doladit
            return True
    return False

def is_in_center(bbox, frame_width, frame_height, margin=0.2):
    """Vrátí True, pokud se střed boxu nachází uvnitř centrální oblasti rámce."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (margin * frame_width < cx < (1 - margin) * frame_width) and \
           (margin * frame_height < cy < (1 - margin) * frame_height)

def send_alert(track_id, bbox):
    """Spustí alarm pro daný track. Tady jen vypíšeme zprávu, ale můžeš to rozšířit."""
    print(f"ALERT: Osoba s track ID {track_id} na {bbox} nemá potřebné vybavení! Sakra, tohle je nebezpečný!")
    client.publish(mqtt_top.TopicList.AI_DETECTION, f"Safety equipment is missing")





center_left   = int(CENTER_MARGIN * frame_width)
center_top    = int(CENTER_MARGIN * frame_height)
center_right  = int((1 - CENTER_MARGIN) * frame_width)
center_bottom = int((1 - CENTER_MARGIN) * frame_height)

client = mqtt.Client()

client.connect(mqtt_set.HOST, mqtt_set.PORT, 60)
client.loop_start()
print("MQTT client connected")


while True:
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError("Bad frame detected, skipping.")
    except Exception as e:
        print(f"⚠️ OpenCV Error: {e}. Skipping frame.")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(RTSP_CAM_3, cv2.CAP_FFMPEG)
        continue
    #frame = cv2.resize(frame, (output_width, output_height))
    # Spusť YOLO s ByteTrackem – model.track nám už dá track_id přímo
    results = model.track(frame, conf=0.4, iou=0.5, imgsz=1280, half=False, verbose=False,
                          tracker="bytetrack.yaml", persist=True)

    # Připrav seznamy detekcí pro jednotlivé třídy
    # Předpokládáme: 1 = osoba, 2 = vest, 3 = vest (alternativně) a 0 = helma
    person_detections = []  # (track_id, bbox)
    vest_detections = []
    hardhat_detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        bbox = box.xyxy.cpu().numpy().flatten().tolist()  # [x1, y1, x2, y2]
        if cls_id == 1:  # Detekce osoby
            if box.id is not None:
                track_id = int(box.id.item())
                person_detections.append((track_id, bbox))
            else:
                # Můžeš tu případně logovat varování nebo prostě detekci přeskočit
                print("Varování: Detekce osoby nemá přiřazený track_id!")
        elif cls_id in [2, 3]:  # Vest nebo belt-vest
            vest_detections.append(bbox)
        elif cls_id == 0:  # Hardhat
            hardhat_detections.append(bbox)

    current_track_ids = set()

    # Aktualizuj informace o sledovaných osobách podle nových detekcí
    for track_id, bbox in person_detections:
        current_track_ids.add(track_id)
        if track_id not in tracked_persons:
            tracked_persons[track_id] = {
                "bbox": bbox,
                "missed_frames": 0,
                "alert_count": 0,
                "alerted": False,
                "center_image": False
            }
            print(f"Nová zatracená osoba s id {track_id}!")
        else:
            tracked_persons[track_id]["bbox"] = bbox
            tracked_persons[track_id]["missed_frames"] = 0

        # Zkontroluj vybavení
        has_vest = is_equipped(bbox, vest_detections)
        has_hardhat = is_equipped(bbox, hardhat_detections)  # Pro jednoduchost je hardhat kontrola vypnutá. Pokud chceš, povol ji: is_equipped(bbox, hardhat_detections)
        if not (has_vest and has_hardhat):
            tracked_persons[track_id]["alert_count"] += 1
        else:
            tracked_persons[track_id]["alert_count"] = 0

        # Zjisti, zda je osoba ve středu
        if is_in_center(bbox, frame_width, frame_height, CENTER_MARGIN):
            tracked_persons[track_id]["center_image"] = True
        else:
            tracked_persons[track_id]["center_image"] = False

        # Pokud jsou podmínky pro alarm splněny, vyhoď poplašnou zprávu
        if (tracked_persons[track_id]["alert_count"] >= ALERT_THRESHOLD and 
            not tracked_persons[track_id]["alerted"] and tracked_persons[track_id]["center_image"]):
            send_alert(track_id, bbox)
            tracked_persons[track_id]["alerted"] = True

    # U starých stop, které se v aktuálním frame neobjevily, zvýš počet zmeškaných framů
    for track_id in list(tracked_persons.keys()):
        if track_id not in current_track_ids:
            tracked_persons[track_id]["missed_frames"] += 1
            if tracked_persons[track_id]["missed_frames"] > MAX_MISSED_FRAMES:
                del tracked_persons[track_id]

    # Vykresli boxy a informace o sledovaných osobách
    for track_id, track in tracked_persons.items():
        bbox = track["bbox"]
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        color = (0, 0, 255) if track["alerted"] else (0, 255, 0)
        cv2.rectangle(frame, pt1, pt2, color, 5)
        cv2.rectangle(frame, (center_left, center_top), (center_right, center_bottom), (255, 0, 0), 2)
        cv2.putText(frame, f"ID:{track_id} cnt:{track['alert_count']}", 
                    (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        status_text = "ALERT" if track["alerted"] else "SAFE"
        cv2.putText(frame, status_text, (pt1[0], pt1[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    frame = cv2.resize(frame, (output_width, output_height))
    #frame = cv2.resize(frame, (output_width, int(frame.shape[0] * (output_width / frame.shape[1]))))
    # Pošli zpracovaný frame do FFmpeg
    try:
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

        ffmpeg_process.stdin.write(frame_yuv.tobytes())
    except BrokenPipeError:
        print("FFmpeg pipe se posral. Končím.")
        break
    #cv2.imshow("Frame", frame_resized)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
cv2.destroyAllWindows()

