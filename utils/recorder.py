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

from utils.asyncio_capture import CaptureManager
from utils.display import DisplayManager
from utils.writer_zarr import WriteManager_Zarr
from utils.writer_hdf5 import WriteManager_HDF5


class VideoRecorder:
    def __init__(self, target_fps=30, chunk_size=60, compression_level=5):
        temp_dir = TemporaryDirectory()
        self._temp_dir = temp_dir
        self.output_path = os.path.join(temp_dir.name, "recording.zarr")  # 临时路径统一使用 zarr 格式存储
        self.target_fps = target_fps
        self.chunk_size = chunk_size
        self.compression_level = compression_level

        self.capture = CaptureManager(target_fps)
        self.display = DisplayManager()

        # 临时写入器统一使用 Zarr 存储帧数据
        self.writer = WriteManager_Zarr(self.output_path, chunk_size, compression_level)
        self.recorded_frames = []

        self.executor = ThreadPoolExecutor(max_workers=3)
        self.recording = False
        self._should_stop = False

    def _handle_space_request(self):
        self.recording = not self.recording
        print("开始录制..." if self.recording else "暂停录制...")

    def _handle_stop_request(self):
        self._should_stop = True
        self.capture.request_stop()

    async def _process_frames(self, frame_queue):
        frames_batch = []
        while not self._should_stop or not frame_queue.empty():
            try:
                frame = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                self.display.update_frame(frame)

                if self.recording:
                    self.recorded_frames.append(frame)
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

            # 保存对话框
            root = Tk()
            root.withdraw()
            filetypes = [("Zarr 文件夹", "*.zarr"), ("HDF5 文件", "*.h5")]

            save_path = filedialog.asksaveasfilename(
                title="保存文件为...",
                defaultextension=".zarr",
                filetypes=filetypes,
                initialfile="new_recording"
            )

            if save_path:
                ext = os.path.splitext(save_path)[1].lower()
                if ext == ".zarr":
                    final_writer = WriteManager_Zarr(save_path, self.chunk_size, self.compression_level)
                elif ext == ".h5":
                    final_writer = WriteManager_HDF5(save_path, self.chunk_size, self.compression_level)
                else:
                    print("不支持的文件扩展名，取消保存。")
                    shutil.rmtree(self.output_path, ignore_errors=True)
                    return

                # 重新写入完整帧数据
                print("正在将数据写入最终文件...")
                final_writer.write_batch(self.recorded_frames)
                print(f"文件已保存到: {save_path}")
            else:
                print("用户取消保存，临时文件将被删除。")
                shutil.rmtree(self.output_path, ignore_errors=True)


if __name__ == "__main__":
    async def main():
        recorder = VideoRecorder()
        await recorder.start()
        print("调试完成")

    asyncio.run(main())
