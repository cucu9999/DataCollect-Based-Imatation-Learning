import asyncio
from concurrent.futures import ThreadPoolExecutor
from capture import CaptureManager
from display import DisplayManager
from writer import WriteManager

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
        self._should_stop = False

    def _handle_stop_request(self):
        """同步停止处理"""
        self._should_stop = True
        self.capture.request_stop()

    async def _process_frames(self, frame_queue, video_array):
        frames_batch = []
        while not self._should_stop or not frame_queue.empty():
            try:
                frame = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
                frames_batch.append(frame)
                self.display.update_frame(frame)

                if len(frames_batch) >= self.chunk_size:
                    await self._write_frames(video_array, frames_batch)
                    frames_batch.clear()

            except asyncio.TimeoutError:
                if self._should_stop:
                    break

        if frames_batch:
            await self._write_frames(video_array, frames_batch)

    async def _write_frames(self, video_array, frames):
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.writer.write_batch,
            video_array,
            frames.copy()
        )

    async def start(self):
        self.display.register_stop_callback(self._handle_stop_request)
        await self.capture.initialize()
        
        init_frame = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.capture.cap.read()[1]
        )
        video_array = self.writer.initialize(init_frame)
        
        self.display.start()
        frame_queue = asyncio.Queue(maxsize=300)
        
        capture_task = asyncio.create_task(self.capture.capture_frames(frame_queue))
        process_task = asyncio.create_task(self._process_frames(frame_queue, video_array))
        
        try:
            await asyncio.gather(capture_task, process_task)
        finally:
            self.capture.release()
            self.display.stop()
            self.executor.shutdown(wait=True)