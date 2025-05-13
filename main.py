import asyncio
from utils.capture import CaptureManager
from utils.display import DisplayManager
from utils.writer_zarr import WriteManager_Zarr
from utils.writer_hdf5 import WriteManager_HDF5
from utils.recorder import VideoRecorder

async def main():
    recorder = VideoRecorder(
        target_fps=30,
        chunk_size=60,
        compression_level=1
    )
    
    try:
        await recorder.start()
    except asyncio.CancelledError:
        print("录制已停止")
    except Exception as e:
        print(f"录制出错: {str(e)}")
    finally:
        print("录制结束")

if __name__ == "__main__":
    asyncio.run(main())