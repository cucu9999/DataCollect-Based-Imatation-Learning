import cv2
import platform
import threading

class DisplayManager:
    def __init__(self, window_name="Recording"):
        self.window_name = window_name
        self.last_frame = None
        self.stop_event = threading.Event()
        self.display_thread = None
        self.space_pressed = False
        self.q_pressed = False
        self._external_space_callback = None
        self._external_stop_callback = None

    def update_frame(self, frame):
        self.last_frame = frame

    def register_space_callback(self, callback):
        """注册空格键按下后的回调函数（开始/停止录制）"""
        if callable(callback):
            self._external_space_callback = callback
        else:
            raise ValueError("回调必须是可调用函数")

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
                if self._external_space_callback:
                    self._external_space_callback()  # 调用录制开始/停止回调
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

if __name__ == "__main__":
    import numpy as np
    import time

    def on_space():
        print("空格键按下：模拟开始/停止录制")

    def on_quit():
        print("Q 键按下：退出显示")

    display = DisplayManager("测试显示窗口")
    display.register_space_callback(on_space)
    display.register_stop_callback(on_quit)

    display.start()

    try:
        while not display.should_stop():
            dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            display.update_frame(dummy_frame)
            time.sleep(1/30)
    finally:
        display.stop()


