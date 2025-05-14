import os
import h5py
import numpy as np
from datetime import datetime

class WriteManager_HDF5:
    def __init__(self, hdf5_path, chunk_size=60, compression_level=5):
        self.hdf5_path = os.path.abspath(hdf5_path)
        self.chunk_size = chunk_size
        self.compression_level = compression_level
        self.dataset = None
        self.total_written = 0  
        self.hdf5_file = None

    def _initialize_if_needed(self, first_frame):
        if self.dataset is not None:
            return

        os.makedirs(os.path.dirname(self.hdf5_path), exist_ok=True)
        self.hdf5_file = h5py.File(self.hdf5_path, 'a')

        height, width, _ = first_frame.shape
        maxshape = (None, height, width, 3)
        chunks = (self.chunk_size, height, width, 3)

        obs_group = self.hdf5_file.require_group("observations")
        img_group = obs_group.require_group("images")
        self.dataset = img_group.create_dataset(
            'top',
            shape=(0, height, width, 3),
            maxshape=maxshape,
            chunks=chunks,
            dtype='uint8',
            compression='gzip',
            compression_opts=self.compression_level
        )

        # 创建空的 qpos 和 qvel（以零占位，或稍后填充）
        obs_group.create_dataset('qpos', shape=(0,), maxshape=(None,), dtype='float32')
        obs_group.create_dataset('qvel', shape=(0,), maxshape=(None,), dtype='float32')
        # 创建空的 action（也可以后期追加）
        self.hdf5_file.create_dataset('action', shape=(0,), maxshape=(None,), dtype='float32')

        self.hdf5_file.attrs['created'] = datetime.now().isoformat()
        self.hdf5_file.attrs['color_space'] = 'BGR'
        self.hdf5_file.attrs['resolution'] = f"{width}x{height}"
        self.hdf5_file.attrs['fps'] = 30

        print("已创建 HDF5 数据集")

    def write_batch(self, frames):
        """将一批帧写入 HDF5 文件"""
        if not frames:
            print("没有帧可写入。")
            return

        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]

        self._initialize_if_needed(frames[0])

        new_total = self.total_written + frames.shape[0]
        self.dataset.resize((new_total, *self.dataset.shape[1:]))
        self.dataset[self.total_written:new_total] = frames
        self.total_written = new_total  

        print(f"已写入 {frames.shape[0]} 帧，当前总帧数: {self.total_written}")

    def __del__(self):
        if hasattr(self, 'hdf5_file'):
            # 确保文件完全关闭
            self.hdf5_file.flush()
            self.hdf5_file.close()


if __name__ == "__main__":
    import numpy as np, os, h5py
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    frames = [dummy] * 5
    path = "test_writer_debug.h5"

    if os.path.exists(path):
        os.remove(path)  

    WriteManager_HDF5(path).write_batch(frames)

    print("写入成功，已完成 HDF5 文件创建并写入。")

    os.remove(path)
    print("已删除测试 HDF5 文件。")
