import zarr
import numpy as np
from numcodecs import Zstd
from datetime import datetime
import os

class WriteManager:
    def __init__(self, zarr_path, chunk_size, compression_level):
        self.zarr_path = zarr_path
        self.chunk_size = chunk_size
        self.compression_level = compression_level
        self.written_frames = 0
        self.written_bytes = 0

    def initialize(self, first_frame):
        """初始化Zarr存储"""
        if not os.path.exists(self.zarr_path):
            root = zarr.open(self.zarr_path, mode='w')
            video = root.create_dataset(
                'video',
                shape=(0, *first_frame.shape),
                chunks=(self.chunk_size, *first_frame.shape),
                dtype=np.uint8,
                compressor=Zstd(level=self.compression_level)
            )
            root.attrs.update({
                'created': datetime.now().isoformat(),
                'resolution': f"{first_frame.shape[1]}x{first_frame.shape[0]}"
            })
        else:
            root = zarr.open(self.zarr_path, mode='a')
            video = root['video']
        return video

    def write_batch(self, video_array, frames):
        """写入一批帧"""
        if not frames:
            return

        current_size = video_array.shape[0]
        new_size = current_size + len(frames)
        video_array.resize((new_size, *video_array.shape[1:]))
        video_array[current_size:new_size] = frames

        self.written_frames += len(frames)
        self.written_bytes += sum(f.nbytes for f in frames)