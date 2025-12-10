import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import base64
import math

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- MediaPipe 初始化 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- 轨迹记录与魔法逻辑 ---
class SpellCaster:
    def __init__(self):
        self.points = [] # 存储指尖轨迹
        self.is_drawing = False
    
    def detect_shape(self, points):
        """ 简单的形状识别算法 """
        if len(points) < 10: return "unknown"
        
        # 转换为 Numpy 数组
        pts = np.array(points, dtype=np.int32)
        
        # 1. 计算外接矩形和凸包
        x, y, w, h = cv2.boundingRect(pts)
        hull = cv2.convexHull(pts)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: return "unknown"

        # 2. 轮廓近似 (用于计算角点)
        epsilon = 0.04 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        corners = len(approx)

        # 3. 形状判定逻辑
        # 圆形: 面积利用率高，且角点多或非常圆滑
        # 三角形: 3个角
        # 正方形: 4个角
        
        # 这是一个简化版的判定，实际项目中可以使用更复杂的几何算法
        if corners == 3:
            return "triangle" # 火球
        elif corners == 4:
            return "square"   # 大爆炸
        elif corners > 5:
            # 检查是否接近圆形 (长宽比接近 1)
            aspect_ratio = float(w)/h
            if 0.8 <= aspect_ratio <= 1.2:
                return "circle" # 冰锥
        
        return "unknown"

caster = SpellCaster()

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # 1. 接收前端传来的 Base64 图片
            data = await websocket.receive_text()
            header, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 2. MediaPipe 处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            response = {"type": "empty"}

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 获取关键点
                    # 8: 食指指尖, 4: 拇指指尖, 0: 手腕
                    idx_finger = hand_landmarks.landmark[8]
                    thumb = hand_landmarks.landmark[4]
                    
                    # 坐标归一化
                    h, w, c = frame.shape
                    cx, cy = int(idx_finger.x * w), int(idx_finger.y * h)
                    
                    # --- 手势逻辑 ---
                    
                    # 1. 检测 OK 手势 (拇指和食指距离很近) -> 触发虚空魔法
                    dist_ok = math.hypot(idx_finger.x - thumb.x, idx_finger.y - thumb.y)
                    if dist_ok < 0.05:
                        response = {
                            "type": "void_charge", 
                            "x": cx, "y": cy
                        }
                        caster.points = [] # 重置画笔
                    
                    # 2. 检测“食指向上” (画画模式)
                    # 简单判断：食指指尖 y 坐标小于 手指根部 y 坐标，且其他手指收起(这里简化处理，默认只要不是OK就是画画)
                    elif dist_ok >= 0.05:
                        # 正在绘制
                        caster.points.append([cx, cy])
                        response = {
                            "type": "drawing",
                            "x": cx, "y": cy
                        }
                        
                        # 如果点太多，判定一次形状并清空 (模拟施法完成)
                        # 在实际交互中，最好加入“握拳”来确认施法，这里为了演示简化为自动触发
                        if len(caster.points) > 30: 
                            shape = caster.detect_shape(caster.points)
                            if shape != "unknown":
                                response = {
                                    "type": "cast_spell",
                                    "spell": shape, # triangle, square, circle
                                    "x": cx, "y": cy
                                }
                                caster.points = [] # 施法后清空
                            elif len(caster.points) > 60:
                                caster.points = [] # 超时重置

            # 3. 发送结果给前端
            await websocket.send_json(response)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    # 注意：在本地运行请使用 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=8000)
