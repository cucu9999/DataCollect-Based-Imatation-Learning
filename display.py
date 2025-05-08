import cv2 
import threading
import platform

class DisplayManager:
    def __init__(self, window_name="Recording"):
        self.window_name = window_name
        self.last_frame = None
        self.stop_event = threading.Event()
        self.display_thread = None
        self.space_pressed = False
        self.q_pressed = False
        self._external_stop_callback = None

    def update_frame(self, frame):
        self.last_frame = frame

    def register_stop_callback(self, callback):
        """注册必须是同步的回调函数"""
        if callable(callback):
            self._external_stop_callback = callback
        else:
            raise ValueError("回调必须是可调用函数")

    def _display_loop(self):
        if platform.system() != 'Darwin':
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while not self.stop_event.is_set():
            if self.last_frame is not None:
                cv2.imshow(self.window_name, self.last_frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                self.space_pressed = True
            elif key == ord('q'):
                self.q_pressed = True
                if self._external_stop_callback:
                    self._external_stop_callback()  # 同步调用
                break

        cv2.destroyAllWindows()

    def start(self):
        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True
        )
        self.display_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.display_thread:
            self.display_thread.join(timeout=1.0)

    def should_stop(self):
        return self.q_pressed