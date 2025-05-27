import cv2
import time
import numpy as np
from det_face_mediapipe import FaceMeshDetector
from Eye_Control import EyeCtrl
import queue

class ServoController:
    def __init__(self, servo_queue, frame_queue, com_port='COM7'):
        self.servo_queue = servo_queue
        self.frame_queue = frame_queue
        self.face_mesh_detector = FaceMeshDetector()
        self.ctrl = EyeCtrl(com_port)

        # 初始化眼球注视点比例位置
        self.eyeball_horizontal_init = self.ctrl.eyeball_horizontal
        self.eyeball_vertical_init = self.ctrl.eyeball_vertical

        # 延迟初始化帧尺寸，后续第一次拿到帧后设定
        self.frame_width = None
        self.frame_height = None
        self.eyeball_px_x = None
        self.eyeball_px_y = None

    def process(self,frame):
        if frame is None:
            return

        # 截取左边画面（如果你用的是1280宽度的左半部分）
        frame = frame[:, :1280]

        if self.frame_width is None or self.frame_height is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.eyeball_px_x = int(self.eyeball_horizontal_init * self.frame_width)
            self.eyeball_px_y = int(self.eyeball_vertical_init * self.frame_height)

        # 转换BGR到RGB，给detector用
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, _, _ = self.face_mesh_detector.get_results(frame_rgb)

        if landmarks:
            try:
                # 计算人脸中心像素坐标
                face_center_x_norm = (landmarks[33].x + landmarks[263].x) / 2
                face_center_y_norm = (landmarks[33].y + landmarks[263].y) / 2
                face_center_x = int(face_center_x_norm * self.frame_width)
                face_center_y = int(face_center_y_norm * self.frame_height)

                dx = face_center_x - self.eyeball_px_x
                dy = face_center_y - self.eyeball_px_y

                dx_norm = -dx / (self.frame_width / 2)
                dy_norm = -dy / (self.frame_height / 2)

                sensitivity = 0.8
                horizontal_offset = np.clip(self.eyeball_horizontal_init + dx_norm * sensitivity, 0.0, 1.0)
                vertical_offset = np.clip(self.eyeball_vertical_init + dy_norm * sensitivity, 0.0, 1.0)

                # 将舵机动作放入队列
                servo_action = np.array([horizontal_offset, vertical_offset], dtype=np.float32)
                try:
                    self.servo_queue.put_nowait(servo_action)  # 非阻塞式放入队列
                except queue.Full:
                    # 队列已满时跳过
                    pass

                # 发送指令到伺服电机
                self.ctrl.eyeball_horizontal = horizontal_offset
                self.ctrl.eyeball_vertical = vertical_offset
                self.ctrl.send()

                print(f"👁️ 控制: 水平={horizontal_offset:.2f}, 竖直={vertical_offset:.2f}")

            except Exception as e:
                print(f"[控制伺服异常] {e}")
        else:                
            print("😐 没检测到人脸 → 不控制舵机")

        time.sleep(0.02)
