import edge_tts
import asyncio
from pathlib import Path


async def text_to_speech(text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural", rate: int = 0, volume: int = 0, pitch: int = 0):
    """
    Convert text to speech using Microsoft Edge TTS

    Args:
        text (str): Text to convert to speech
        output_path (str): Path to save the audio file
        voice (str): Voice to use for TTS (default: zh-CN-XiaoxiaoNeural)
        rate: str = "+0%" 速度
        volume: str = "+0%" 音量
        pitch: str = "+0Hz" 音高
    """
    try:
        communicate = edge_tts.Communicate(
            text, voice, rate=f"+{rate}%", volume=f"+{volume}%", pitch=f"+{pitch}Hz")
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return False


def tts_sync(text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural", rate: int = 0, volume: int = 0, pitch: int = 0):
    """
    Synchronous wrapper for text_to_speech function
    """
    asyncio.run(text_to_speech(text, output_path, voice, rate, volume, pitch))
