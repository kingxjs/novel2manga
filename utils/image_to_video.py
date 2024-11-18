from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *
import numpy as np
import os

from utils.youdao_api.TtsDemo import create_youdao_request


def generate_video_with_subtitles(image_path, text_segments, output_video, chapter_title):
    """
    根据给定的图片、字幕文本和输出文件名生成带有字幕的视频。

    参数:
        image_path (str): 输入图片路径。
        text_segments (list of str): 字幕文本列表，每段文本对应一个字幕段。
        output_video (str): 输出视频文件路径。

    返回:
        bool: 是否成功生成视频。
    """

    # Step 1: 加载图片并转换为RGB模式
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"图片加载失败: {e}")
        return False

    # Step 2: 生成音频文件
    audio_clips = []
    durations = []
    # 新建voices文件夹
    if not os.path.exists("results/" + chapter_title + "/voices"):
        os.makedirs("results/" + chapter_title + "/voices")
    for i, text in enumerate(text_segments):
        audio_path = "results/" + chapter_title + f"/voices/audio_{i}.mp3"
        # 请替换 create_youdao_request 为你实际的音频生成函数
        try:
            create_youdao_request(text, audio_path)  # 生成音频文件
            audio_clip = AudioFileClip(audio_path)
            audio_clips.append(audio_clip)
            durations.append(audio_clip.duration)
        except Exception as e:
            print(f"音频生成或加载失败: {e}")
            return False

    # Step 3: 自动换行函数
    def wrap_text(text, font, max_width):
        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_width, _ = font.getsize(test_line)

            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)
        return lines

    # Step 4: 叠加字幕到图片上，生成每段文字的图像
    font_path = "fonts/HiraginoSansGB.ttc"  # 替换为实际路径
    font = ImageFont.truetype(font_path, 24, index=0)  # 使用第一个字体（index=0）
    image_clips = []
    for i, text in enumerate(text_segments):
        img = image.copy()
        draw = ImageDraw.Draw(img)

        max_width = img.width - 40  # 设置左右边距
        wrapped_text = wrap_text(text, font, max_width)

        total_text_height = len(wrapped_text) * font.getsize(wrapped_text[0])[1]
        y_text = img.height - total_text_height - 30  # 设置文本的垂直位置

        for line in wrapped_text:
            text_width, text_height = draw.textsize(line, font)
            x_text = (img.width - text_width) // 2  # 居中对齐
            draw.text((x_text, y_text), line, (240, 167, 50), font=font)
            y_text += text_height

        img_clip = ImageClip(np.array(img)).set_duration(durations[i])
        image_clips.append(img_clip)

    # Step 5: 合成最终视频
    try:
        final_clip = concatenate_videoclips(
            [image_clips[i].set_audio(audio_clips[i]) for i in range(len(text_segments))])
        final_clip.write_videofile(output_video, fps=24, codec="libx264", audio_codec="aac")
        print("视频生成成功:", output_video)
        return True
    except Exception as e:
        print(f"视频生成失败: {e}")
        return False


if __name__ == '__main__':
    image_path = "../images/true.png"
    text_segments = ["Have you heard?",
                     "Cao Xiong laoshi had just competed with the academy’s number one teacher from the bottom, Teacher Zhang Xuan!"]
    output_video = "output_video.mp4"

    # 调用函数生成视频
    success = generate_video_with_subtitles(image_path, text_segments, output_video)
    if success:
        print("视频生成成功！")
    else:
        print("视频生成失败！")
