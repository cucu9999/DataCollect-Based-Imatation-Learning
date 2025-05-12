import asyncio
from utils.recorder import VideoRecorder
from utils.display import DisplayManager
from utils.writer import WriteManager
from utils.capture import CaptureManager

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