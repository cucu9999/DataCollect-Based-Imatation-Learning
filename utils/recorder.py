import cv2
import time
import queue
import asyncio
import display
import threading
from writer_hdf5 import WriteManager_HDF5
from track import ServoController
import sys

class VideoRecorder:
    def __init__(self, target_fps=30, width=640, height=480, camera_index=0, frame_queue_size=260, servo_queue_size=260):
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.camera_index = camera_index
        self.frame_queue_size = frame_queue_size
        self.servo_queue_size = servo_queue_size

        self.cap = None
        self.last_frame = None
        self.recording = False
        self._stop_event = threading.Event()

        self.frame_queue = queue.Queue(maxsize=self.frame_queue_size)  # 共享队列
        self.servo_queue = queue.Queue(maxsize=self.servo_queue_size)
        self.display_manager = display.DisplayManager()
        self.display_manager.register_stop_callback(self.on_quit)    # 注册退出回调

        self.save_path = None

    def initialize(self, save_path=None):
        if save_path:
            self.save_path = save_path
        if not self.save_path:
            print("没有指定保存路径，程序退出。")
            sys.exit(1)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def capture_frames(self):
        self.recording = True
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret or frame is None:
                continue

            try:
                if self.frame_queue.qsize() < self.frame_queue_size:
                    self.frame_queue.put_nowait(frame)  # 将帧放入队列
            except queue.Full:
                pass

            try:
                if not self.frame_queue.empty():
                    display_frame = self.frame_queue.queue[0]
                    self.display_manager.update_frame(display_frame)  # 显示帧
            except queue.Empty:
                pass

            self.last_frame = frame
        self.recording = False

    def on_quit(self):
        print("退出程序")
        self.request_stop()  
        self.release()  

    def request_stop(self):
        self._stop_event.set()

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.display_manager.stop()

    def start(self):
        self.display_manager.start()
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.start()
        return capture_thread

    def save_data(self):
        if self.save_path:
            print(f"数据保存到: {self.save_path}")
        else:
            print("⚠️ 没有设置保存路径")

def control_servo(servo_queue, frame_queue, stop_event):
    servo_controller = ServoController(servo_queue, frame_queue)

    while not stop_event.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            frame = frame_queue.get_nowait()
            servo_controller.process(frame)  # 控制舵机
        time.sleep(0.02)

    print("第二个线程（舵机控制）结束")


async def write_data_to_h5_async(frame_queue, servo_queue, write_manager, stop_event):
    frames_batch = []
    servo_batch = []

    while not stop_event.is_set() or not frame_queue.empty() or not servo_queue.empty():
        if not frame_queue.empty() and not servo_queue.empty():
            try:
                frame = frame_queue.get_nowait()
                servo_action = servo_queue.get_nowait()
                frames_batch.append(frame)
                servo_batch.append(servo_action)

                if len(frames_batch) >= 120 and len(servo_batch) >= 120:
                    # 批量写入
                    await asyncio.to_thread(write_manager.write_top_image_with_timestamp, frames_batch)
                    await asyncio.to_thread(write_manager.write_eye_action_with_timestamp, servo_batch)
                    frames_batch.clear()
                    servo_batch.clear()
            except queue.Empty:
                await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0.01)

    print("第三个线程（数据写入）结束")

def main():
    save_path = "output_data.h5"
    capture_manager = VideoRecorder()

    capture_manager.initialize(save_path=save_path)

    frame_queue = capture_manager.frame_queue
    servo_queue = capture_manager.servo_queue

    write_manager = WriteManager_HDF5(save_path)

    stop_event = threading.Event()

    # 启动显示与捕获线程
    capture_thread = capture_manager.start()

    # 启动舵机控制线程
    control_servo_thread = threading.Thread(target=control_servo, args=(servo_queue, frame_queue, stop_event))
    control_servo_thread.start()

    loop = asyncio.get_event_loop()

    # 启动异步写入任务
    write_data_task = loop.create_task(write_data_to_h5_async(frame_queue, servo_queue, write_manager, stop_event))

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            capture_manager.on_quit()
            break

    stop_event.set()

    # 等待所有线程完成
    capture_thread.join()  
    control_servo_thread.join()  
    loop.run_until_complete(write_data_task) 

    print(f"✅ 数据保存到: {save_path}")
    capture_manager.release()

if __name__ == "__main__":
    main()