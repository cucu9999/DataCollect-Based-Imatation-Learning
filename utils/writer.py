import zarr
import numpy as np
import os
from datetime import datetime
from numcodecs import Blosc
from zarr.storage import DirectoryStore

class WriteManager:
    def __init__(self, zarr_path, chunk_size=60, compression_level=7):
        self.zarr_path = os.path.abspath(zarr_path)
        self.chunk_size = chunk_size
        self.compression_level = compression_level
        self.video_array = None

    def _initialize_if_needed(self, first_frame):
        """如有必要，初始化 Zarr 文件和数据集（Zarr v2 API）"""
        if self.video_array is not None:
            return

        os.makedirs(os.path.dirname(self.zarr_path), exist_ok=True)

        # 打开或新建 Zarr 文件
        store = DirectoryStore(self.zarr_path)  # 强制 v2 存储格式
        root = zarr.group(store=store, overwrite=False)

        if 'video' not in root:
            print("Zarr文件不存在或未包含 'video' 数据集，正在创建...")

            height, width, _ = first_frame.shape
            chunk_shape = (self.chunk_size, height, width, 3)

            # 使用 Blosc 压缩器（兼容 Zarr v2）
            compressor = Blosc(cname='zstd', clevel=self.compression_level, shuffle=Blosc.BITSHUFFLE)

            self.video_array = root.create_dataset(
                name='video',
                shape=(0, height, width, 3),
                chunks=chunk_shape,
                dtype='uint8',
                compressor=compressor,
                overwrite=False,
            )

            root.attrs.update({
                'created': datetime.now().isoformat(),
                'fps': 30,
                'resolution': f"{width}x{height}",
                'color_space': 'BGR'
            })
            print("已创建新的视频数据集。")
        else:
            print("Zarr文件存在，加载已有数据集。")
            self.video_array = root['video']

    def write_batch(self, frames):
        """将一批帧写入 Zarr 文件"""
        if not frames:
            print("没有帧可写入。")
            return

        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]

        self._initialize_if_needed(frames[0])

        current_frames = self.video_array.shape[0]
        new_total = current_frames + frames.shape[0]

        # 扩展数据集以容纳新的帧
        self.video_array.resize((new_total, *self.video_array.shape[1:]))
        self.video_array[current_frames:new_total] = frames

        print(f"已写入 {frames.shape[0]} 帧，当前总帧数: {new_total}")
