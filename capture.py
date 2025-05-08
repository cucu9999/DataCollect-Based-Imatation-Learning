import cv2
import asyncio
import platform

class CaptureManager:
    def __init__(self, target_fps):
        self.target_fps = target_fps
        self.cap = None
        self.last_frame = None
        self.recording = False
        self._stop_event = asyncio.Event()

    async def initialize(self):
        if platform.system() == 'Windows':
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        return True

    async def capture_frames(self, frame_queue):
        self.recording = True
        while not self._stop_event.is_set():
            ret, frame = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.cap.read()
            )

            if not ret or frame is None:
                continue

            await frame_queue.put(frame)
            self.last_frame = frame
            await asyncio.sleep(1/self.target_fps)

        self.recording = False

    def request_stop(self):
        self._stop_event.set()

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()