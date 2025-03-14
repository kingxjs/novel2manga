from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *
from moviepy.video.fx import fadein, fadeout
import numpy as np
import os
import time

from utils.tts import tts_sync


def generate_video_with_subtitles(data, output_video, chapter_title, voice="zh-CN-XiaoxiaoNeural"):
    """
    根据给定的图片、字幕文本和输出文件名生成带有字幕的视频。

    参数:
        data (str): 输入图片路径。
        output_video (str): 输出视频文件路径。

    返回:
        bool: 是否成功生成视频。
    """
    # 生成音频文件
    audio_clips = []
    durations = []
    # # 新建voices文件夹
    if not os.path.exists("results/" + chapter_title + "/voices"):
        os.makedirs("results/" + chapter_title + "/voices")

    # 自动换行函数
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

    font_path = "fonts/HiraginoSansGB.ttc"  # 替换为实际路径
    font = ImageFont.truetype(font_path, 20, index=0)  # 使用第一个字体（index=0）
    image_clips = []
    text_segments_size = 0
    final_clips = []
    for j, item in enumerate(data):
        image = Image.open(item["image_path"]).convert("RGB")

        text_segments = item["sentences"]
        for i, text in enumerate(text_segments):
            img = image.copy()
            draw = ImageDraw.Draw(img)
            audio_path = "results/" + chapter_title + \
                f"/voices/audio_{j}_{i}.mp3"
            # 请替换 create_youdao_request 为你实际的音频生成函数
            try:
                tts_sync(text, audio_path, voice=voice,
                         rate=20, pitch=10)  # 生成音频文件
                # 判断 audio_path 是否存在 并且 不是0字节
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    # 删除 audio_path
                    os.remove(audio_path)
                    time.sleep(1)  # 等待500ms，避免频繁调用API
                    tts_sync(text, audio_path, voice=voice,
                         rate=20, pitch=10)  # 生成音频文件
                    
                print(f"音频生成成功: {audio_path}")
                time.sleep(0.5)  # 等待500ms，避免频繁调用API
                audio_clip = AudioFileClip(audio_path)
                audio_clips.append(audio_clip)
                durations.append(audio_clip.duration)
            except Exception as e:
                print(f"音频生成或加载失败: {e}")
                return False

            # 叠加字幕到图片上，生成每段文字的图像
            max_width = img.width - 100  # 设置左右边距
            wrapped_text = wrap_text(text, font, max_width)

            total_text_height = len(wrapped_text) * \
                font.getsize(wrapped_text[0])[1]
            y_text = img.height - total_text_height - 30  # 设置文本的垂直位置

            for line in wrapped_text:
                text_width, text_height = draw.textsize(line, font)
                x_text = (img.width - text_width) // 2  # 居中对齐
                draw.text((x_text, y_text), line, (240, 167, 50), font=font)
                y_text += text_height

            img_clip = ImageClip(np.array(img)).set_duration(
                durations[text_segments_size])
            img_clip = img_clip.set_audio(audio_clips[text_segments_size])
            image_clips.append(img_clip)

            # clip = image_clips[text_segments_size].fx(fadein.fadein, 0.1)  # 添加淡入效果
            # if text_segments_size > 0:  # 在第二个及以后的片段添加淡出效果
            #     clip = clip.fx(fadeout.fadeout,  0.1)
            final_clips.append(img_clip)

            text_segments_size += 1

    # 合成最终视频
    try:
        # 使用列表推导式创建最终片段，并添加淡入淡出效果
        # final_clips = []
        # for i in range(len(image_clips)):
        #     clip = image_clips[i].fx(fadein.fadein, 0.1)  # 添加淡入效果
        #     if i > 0:  # 在第二个及以后的片段添加淡出效果
        #         clip = clip.fx(fadeout.fadeout,  0.1)
        #     final_clips.append(clip)
        # final_clip = concatenate_videoclips(
        #     [image_clips[i].set_audio(audio_clips[i]) for i in range(text_segments_size)], method="compose")
        final_clip = concatenate_videoclips(final_clips, method="compose")
        final_clip.write_videofile(
            output_video, fps=24, codec="libx264", audio_codec="aac")
        print("视频生成成功:", output_video)
        return True
    except Exception as e:
        print(e)
        print(f"视频生成失败: {e}")
        return False


# if __name__ == '__main__':
#     image_paths = "../images/true.png"
#     text_segments = ["Have you heard?",
#                      "Cao Xiong laoshi had just competed with the academy’s number one teacher from the bottom, Teacher Zhang Xuan!"]
#     output_video = "output_video.mp4"

#     # 调用函数生成视频
#     success = generate_video_with_subtitles(image_path, text_segments, output_video)
#     if success:
#         print("视频生成成功！")
#     else:
#         print("视频生成失败！")
