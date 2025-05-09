import asyncio
from concurrent.futures import ThreadPoolExecutor
from utils.capture import CaptureManager
from utils.display import DisplayManager
from utils.writer import WriteManager

class VideoRecorder:
    def __init__(self, zarr_path, target_fps=30, chunk_size=60, compression_level=1):
        self.zarr_path = zarr_path
        self.target_fps = target_fps
        self.chunk_size = chunk_size
        self.compression_level = compression_level
        
        self.capture = CaptureManager(target_fps)
        self.display = DisplayManager()
        self.writer = WriteManager(zarr_path, chunk_size, compression_level)
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.recording = False
        self._should_stop = False

    def _handle_space_request(self):
        """切换录制状态"""
        self.recording = not self.recording
        if self.recording:
            print("开始录制...")
        else:
            print("停止录制...")

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
