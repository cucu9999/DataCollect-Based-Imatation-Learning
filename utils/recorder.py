import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shutil
import asyncio
from tkinter import Tk, filedialog
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

# 添加 utils 文件夹到模块搜索路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.capture import CaptureManager
from utils.display import DisplayManager
from utils.writer import WriteManager


class VideoRecorder:
    def __init__(self, zarr_path=None, target_fps=30, chunk_size=60, compression_level=1):
        if zarr_path is None:
            
            temp_dir = TemporaryDirectory()
            zarr_path = os.path.join(temp_dir.name, "recording.zarr")
            self._temp_dir = temp_dir  
        self.zarr_path = os.path.abspath(zarr_path)
        self.target_fps = target_fps
        self.chunk_size = chunk_size
        self.compression_level = compression_level

        self.capture = CaptureManager(target_fps)
        self.display = DisplayManager()
        self.writer = WriteManager(self.zarr_path, chunk_size, compression_level)

        self.executor = ThreadPoolExecutor(max_workers=3)
        self.recording = False
        self._should_stop = False

    def _handle_space_request(self):
        """切换录制状态"""
        self.recording = not self.recording
        if self.recording:
            print("开始录制...")
        else:
            print("暂停录制...")

    def _handle_stop_request(self):
        """同步停止处理"""
        self._should_stop = True
        self.capture.request_stop()

    async def _process_frames(self, frame_queue):
        frames_batch = []
        while not self._should_stop or not frame_queue.empty():
            try:
                frame = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                self.display.update_frame(frame)

                if self.recording:
                    frames_batch.append(frame)

                    if len(frames_batch) >= self.chunk_size:
                        await self._write_frames(frames_batch)
                        frames_batch.clear()

            except asyncio.TimeoutError:
                if self._should_stop:
                    break

        if frames_batch:
            await self._write_frames(frames_batch)

    async def _write_frames(self, frames):
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.writer.write_batch,
            frames.copy()
        )

    async def start(self):
        self.display.register_space_callback(self._handle_space_request)
        self.display.register_stop_callback(self._handle_stop_request)
        await self.capture.initialize()

        self.display.start()
        frame_queue = asyncio.Queue()

        capture_task = asyncio.create_task(self.capture.capture_frames(frame_queue))
        process_task = asyncio.create_task(self._process_frames(frame_queue))

        try:
            await asyncio.gather(capture_task, process_task)
        finally:
            self.capture.release()
            self.display.stop()
            self.executor.shutdown(wait=True)

            print("录制完成，准备保存文件...")

            # 弹出保存对话框
            root = Tk()
            root.withdraw()

            save_path = filedialog.asksaveasfilename(
                title="保存 Zarr 文件为...",
                defaultextension=".zarr",
                filetypes=[("Zarr 文件夹", "*.zarr")],
                initialfile="new_recording.zarr"
            )

            if save_path:
                shutil.move(self.zarr_path, save_path)
                print(f"Zarr 文件已保存到: {save_path}")
            else:
                print("用户取消保存，临时文件将自动由操作系统清理。")
                shutil.rmtree(self.zarr_path, ignore_errors=True)

if __name__ == "__main__":
    async def main():
        recorder = VideoRecorder()
        await recorder.start()
        print("调试完成")

    asyncio.run(main())
