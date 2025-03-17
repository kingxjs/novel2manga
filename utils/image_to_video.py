import os
import time
import random
import math
import traceback
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Callable, Optional, Union

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *
import numpy as np
from PIL import Image

from utils.tts import tts_sync

# ---------------------- 日志设置 ----------------------

logger = logging.getLogger("video_generator")

# ---------------------- 缓动函数 ----------------------


@lru_cache(maxsize=1024)
def ease_cubic(t: float) -> float:
    """三次方缓动函数"""
    return t * t * t


@lru_cache(maxsize=1024)
def ease_quad(t: float) -> float:
    """二次方缓动函数"""
    return t * t


@lru_cache(maxsize=1024)
def ease_sine(t: float) -> float:
    """正弦缓动函数"""
    return 1 - math.cos(t * math.pi / 2)


@lru_cache(maxsize=1024)
def ease_segment(t: float) -> float:
    """分段缓动函数"""
    if t < 0.2:
        p = t / 0.2
        return p * p * p
    elif t > 0.8:
        p = (t - 0.8) / 0.2
        return 0.8 + 0.2 * (1 - (1-p) * (1-p) * (1-p))
    else:
        p = (t - 0.2) / 0.6
        return 0.2 + 0.6 * p


# 缓动函数映射
EASING_FUNCTIONS = {
    "cubic": ease_cubic,
    "quad": ease_quad,
    "sine": ease_sine,
    "segment": ease_segment
}


def get_easing_function(ease_type: str = "cubic") -> Callable[[float], float]:
    """
    获取指定类型的缓动函数

    Args:
        ease_type: 缓动类型名称

    Returns:
        对应的缓动函数
    """
    return EASING_FUNCTIONS.get(ease_type, ease_cubic)

# ---------------------- 视频效果函数 ----------------------

# 尝试导入OpenCV，如果不可用则使用PIL
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    from PIL import Image

@lru_cache(maxsize=128)
def get_easing_function(ease_type: str = "cubic"):
    """
    获取指定类型的缓动函数，带缓存
    
    Args:
        ease_type: 缓动类型名称
        
    Returns:
        对应的缓动函数
    """
    # 返回相应的缓动函数
    if ease_type == "cubic":
        return lambda t: t * t * t
    elif ease_type == "quad":
        return lambda t: t * t
    elif ease_type == "sine":
        return lambda t: 1 - np.cos(t * np.pi / 2)
    elif ease_type == "segment":
        def ease_segment(t):
            if t < 0.2:
                p = t / 0.2
                return p * p * p
            elif t > 0.8:
                p = (t - 0.8) / 0.2
                return 0.8 + 0.2 * (1 - (1-p) * (1-p) * (1-p))
            else:
                p = (t - 0.2) / 0.6
                return 0.2 + 0.6 * p
        return ease_segment
    else:
        return lambda t: t  # 线性缓动（无缓动）

def optimized_zoom_effect(clip: VideoClip, zoom_start: float = 1.0, 
                         zoom_end: float = 1.2, ease_type: str = "cubic",
                         use_cache: bool = True) -> VideoClip:
    """
    创建优化的居中平滑放大效果，放大时间与音频时间一致
    
    Args:
        clip: 要缩放的视频片段
        zoom_start: 起始缩放比例
        zoom_end: 结束缩放比例
        ease_type: 缓动类型 ("cubic", "quad", "sine", "segment")
        use_cache: 是否使用缓存提高性能（长视频建议开启）
    
    Returns:
        应用了居中放大效果的视频片段
    """
    duration = clip.duration
    orig_w, orig_h = clip.size
    ease_func = get_easing_function(ease_type)
    
    # 对于长视频，使用缓存可以提高性能
    if use_cache and duration > 10:  # 超过10秒的视频使用缓存
        fps = 24  # 固定帧率
        frame_count = int(duration * fps)
        scale_cache = {}
        
        # 预计算缩放值
        for i in range(frame_count + 1):
            t = i / fps
            progress = t / duration
            eased_progress = ease_func(progress)
            scale = zoom_start + (zoom_end - zoom_start) * eased_progress
            scale_cache[i] = scale
    
    def transform(t):
        # 计算当前时间点的缩放值
        if use_cache and duration > 10:
            # 从缓存获取缩放值
            frame_idx = int(t * 24)
            scale = scale_cache.get(frame_idx, 
                                  zoom_start + (zoom_end - zoom_start) * 
                                  ease_func(t / duration))
        else:
            # 直接计算缩放值
            progress = t / duration
            eased_progress = ease_func(progress)
            scale = zoom_start + (zoom_end - zoom_start) * eased_progress
        
        # 获取原始帧
        frame = clip.get_frame(t)
        
        # 确保帧是正确的格式
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # 如果是灰度图像，转换为RGB
        if len(frame.shape) == 2:
            frame = np.stack((frame,) * 3, axis=-1)
        
        # 计算新尺寸（确保为整数）
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        # 确保新尺寸至少与原始尺寸一样大
        new_w = max(new_w, orig_w)
        new_h = max(new_h, orig_h)
        
        try:
            if CV2_AVAILABLE:
                # 使用OpenCV进行高效缩放
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # 计算裁剪区域（确保居中）
                start_x = (new_w - orig_w) // 2
                start_y = (new_h - orig_h) // 2
                
                # 裁剪回原始尺寸
                cropped = resized[start_y:start_y+orig_h, start_x:start_x+orig_w]
                
                # # 确保颜色空间正确
                # if frame.shape[-1] == 3 and not np.array_equal(frame[0,0], cropped[0,0]):
                #     # 如果颜色通道顺序不匹配，转换回来
                #     cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
                return cropped
            else:
                # 使用PIL进行高质量缩放
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                
                # 计算裁剪区域（确保居中）
                left = (new_w - orig_w) // 2
                top = (new_h - orig_h) // 2
                right = left + orig_w
                bottom = top + orig_h
                
                # 裁剪并转回numpy数组
                cropped = pil_img.crop((left, top, right, bottom))
                return np.array(cropped)
                
        except Exception as e:
            logging.error(f"帧处理错误: {e}, 时间: {t}, 尺寸: {new_w}x{new_h}")
            # 出错时返回原始帧
            return frame
    
    # 创建新的视频片段
    new_clip = VideoClip(lambda t: transform(t), duration=duration)
    
    # 保留原始音频
    if clip.audio is not None:
        new_clip = new_clip.set_audio(clip.audio)
    
    return new_clip

def create_pan_effect(clip: VideoClip, target_size: Tuple[int, int], 
                     direction: str = "vertical", amplitude: float = 0.1, 
                     frequency: float = 1.0, ease_type: str = "sine") -> VideoClip:
    """
    创建平滑的平移效果
    
    Args:
        clip: 要平移的视频片段
        target_size: 目标尺寸 (width, height)
        direction: 平移方向 ("vertical", "horizontal", "circular", "figure8")
        amplitude: 平移幅度，相对于视频高度的比例
        frequency: 平移频率，完整循环次数
        ease_type: 缓动类型 ("sine", "quad", "cubic")
    
    Returns:
        应用了平移效果的视频片段
    """
    target_w, target_h = target_size
    clip_duration = clip.duration
    ease_func = get_easing_function(ease_type)
    
    # 使用固定帧率而不是从clip获取
    fps = 24  # 固定帧率
    frame_count = int(clip_duration * fps)
    position_cache = {}
    
    # 计算中心点
    center_x = target_w / 2
    center_y = target_h / 2
    max_move_y = target_h * amplitude
    max_move_x = target_w * amplitude
    
    for i in range(frame_count + 1):
        t = i / fps
        progress = (t / clip_duration) * frequency
        
        if direction == "vertical":
            # 垂直平移
            offset_y = max_move_y * ease_func(progress % 1.0)
            position_cache[i] = (center_x, center_y + offset_y)
            
        elif direction == "horizontal":
            # 水平平移
            offset_x = max_move_x * ease_func(progress % 1.0)
            position_cache[i] = (center_x + offset_x, center_y)
            
        elif direction == "circular":
            # 圆形平移
            angle = progress * 2 * math.pi
            offset_x = max_move_x * math.cos(angle)
            offset_y = max_move_y * math.sin(angle)
            position_cache[i] = (center_x + offset_x, center_y + offset_y)
            
        elif direction == "figure8":
            # 8字形平移
            angle = progress * 2 * math.pi
            offset_x = max_move_x * math.sin(2 * angle)
            offset_y = max_move_y * math.sin(angle)
            position_cache[i] = (center_x + offset_x, center_y + offset_y)
            
        else:
            # 默认垂直平移
            offset_y = max_move_y * ease_func(progress % 1.0)
            position_cache[i] = (center_x, center_y + offset_y)

    def pan_position(t):
        # 查找缓存中最接近的位置
        frame_idx = int(t * fps)
        return position_cache.get(frame_idx, position_cache[0])  # 如果找不到，返回第一帧位置

    # 应用平移效果
    return clip.set_position(pan_position)


def create_dynamic_effect(clip: VideoClip, effect_type: str = "random", 
                         target_aspect_ratio: float = 4/3) -> VideoClip:
    """
    为视频片段添加动态效果（平移、缩放等）
    
    Args:
        clip: 输入的视频片段
        effect_type: 效果类型: "random", "pan", "zoom", "pan_zoom"
        target_aspect_ratio: 目标宽高比，默认为4:3
    
    Returns:
        应用了动态效果的视频片段
    """
    clip_duration = clip.duration
    orig_w, orig_h = clip.size
    
    # 计算目标尺寸（4:3比例）
    target_h = orig_h
    target_w = int(target_h * target_aspect_ratio)
    
    # 如果原始宽度不够，调整高度
    if orig_w < target_w:
        target_w = orig_w
        target_h = int(target_w / target_aspect_ratio)
    
    # 创建背景（黑色）
    background = ColorClip(size=(target_w, target_h),
                          color=(0, 0, 0), duration=clip_duration)
    
    if effect_type == "random":
        effect_type = random.choice(["pan", "zoom", "pan_zoom"])
    
    try:
        if effect_type == "none":
            # 无效果，保持原样
            final_clip = CompositeVideoClip([background, clip.set_position('center')],
                                           size=(target_w, target_h))
        elif effect_type == "pan":
            # 创建平移效果
            moving_clip = create_pan_effect(
                clip,
                (target_w, target_h),
                direction=random.choice(["vertical", "horizontal", "circular", "figure8"]),
                amplitude=random.uniform(0.05, 0.15),  # 随机平移幅度
                frequency=random.uniform(0.8, 1.2),    # 随机频率
                ease_type=random.choice(["sine", "quad", "cubic"])
            )
            final_clip = CompositeVideoClip([background, moving_clip], size=(target_w, target_h))
            
        elif effect_type == "zoom":
            # 缩放效果
            zoom_start = random.uniform(0.9, 1.1)
            zoom_end = random.uniform(1.1, 1.3)
            zooming_clip = optimized_zoom_effect(
                clip, zoom_start=zoom_start, zoom_end=zoom_end, 
                ease_type=random.choice(["cubic", "sine", "segment"])
            )
            final_clip = CompositeVideoClip([background, zooming_clip.set_position('center')], 
                                           size=(target_w, target_h))
            
        elif effect_type == "pan_zoom":
            # 平移和缩放效果组合
            # 使用固定帧率而不是从clip获取
            fps = 24  # 使用固定的帧率，而不是从clip获取
            frame_count = int(clip_duration * fps)
            frames_cache = {}
            
            for i in range(frame_count + 1):
                t = i / fps
                progress = t / clip_duration
                # 缩放
                scale = 1.0 + 0.2 * math.sin(progress * math.pi)
                # 平移
                max_move = min(target_h, orig_h) * 0.08
                offset_y = max_move * math.sin(progress * 2 * math.pi)
                frames_cache[i] = (scale, offset_y)
            
            def pan_zoom_frame(t):
                frame_idx = int(t * fps)
                scale, offset_y = frames_cache.get(frame_idx, (1.0, 0))
                
                # 先缩放
                zoomed = clip.fx(vfx.resize, scale)
                
                # 计算垂直居中位置
                center_y = (target_h - zoomed.size[1]) / 2
                
                # 设置位置：水平居中，垂直居中+偏移
                return zoomed.set_position(('center', center_y + offset_y)).get_frame(t)
            
            effect_clip = VideoClip(make_frame=pan_zoom_frame, duration=clip_duration)
            final_clip = CompositeVideoClip([background, effect_clip], size=(target_w, target_h))
            
        else:  # 其他值，保持原样
            final_clip = CompositeVideoClip([background, clip.set_position('center')],
                                           size=(target_w, target_h))
    
    except Exception as e:
        logger.error(f"创建动态效果失败: {e}")
        # 出错时返回原始视频（居中放置）
        final_clip = CompositeVideoClip([background, clip.set_position('center')],
                                       size=(target_w, target_h))
    
    return final_clip


# ---------------------- 文本处理函数 ----------------------
@lru_cache(maxsize=100)
def load_font(font_path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    """
    缓存字体加载，避免重复加载

    Args:
        font_path: 字体文件路径
        size: 字体大小
        index: 字体索引

    Returns:
        加载的字体对象
    """
    return ImageFont.truetype(font_path, size, index=index)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """
    自动换行函数，考虑中英文混排情况

    Args:
        text: 需要换行的文本
        font: 使用的字体
        max_width: 最大宽度

    Returns:
        换行后的文本行列表
    """
    # 对于中文，按字符分割可能更合适
    if any(u'\u4e00' <= c <= u'\u9fff' for c in text):
        # 中文文本处理
        lines = []
        current_line = ""

        for char in text:
            test_line = current_line + char
            text_width, _ = font.getsize(test_line)

            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = char

        if current_line:
            lines.append(current_line)
    else:
        # 英文文本处理
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

        if current_line:
            lines.append(current_line)

    return lines


# ---------------------- 音频处理函数 ----------------------
def process_audio_segment(item_idx: int, sentence_idx: int, text: str,
                          chapter_title: str, voice: str,speed:int=30, retry: int = 0) -> Optional[str]:
    """
    处理单个音频段落，便于并行处理

    Args:
        item_idx: 项目索引
        sentence_idx: 句子索引
        text: 文本内容
        chapter_title: 章节标题
        voice: TTS语音类型
        speed: 速度

    Returns:
        生成的音频文件路径，失败时返回None
    """
    audio_path = os.path.join("results", chapter_title,
                              "voices", f"audio_{item_idx}_{sentence_idx}.mp3")

    try:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            try:
                tts_sync(text, audio_path, voice=voice, rate=speed, pitch=10)
            except Exception as e:
                logger.error(f"音频生成失败: {e}")
                time.sleep(2)
                if(retry>=3):
                    return None
                return process_audio_segment(item_idx,sentence_idx, text,chapter_title, voice=voice, speed=speed,retry=retry+1)

            # 验证生成的音频
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return None

        logger.info(f"音频生成成功: {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"音频生成失败: {e}")
        return None


# ---------------------- 主函数 ----------------------
def generate_video_with_subtitles(data: List[Dict], output_video: str,
                                  chapter_title: str, voice: str = "zh-CN-XiaoxiaoNeural", speed:int=30) -> bool:
    """
    根据给定的图片、字幕文本和输出文件名生成带有字幕的视频。

    Args:
        data: 包含图片路径和字幕的数据列表
        output_video: 输出视频文件路径
        chapter_title: 章节标题，用于组织文件
        voice: 文本转语音的声音选择

    Returns:
        是否成功生成视频
    """
    try:
        # 创建目录结构
        voice_dir = os.path.join("results", chapter_title, "voices")
        os.makedirs(voice_dir, exist_ok=True)

        # 加载字体
        font_path = "fonts/HiraginoSansGB.ttc"
        font = load_font(font_path, 20, index=0)

        # 生成所有音频文件（并行处理）
        audio_tasks = []

        for j, item in enumerate(data):
            print(item["sentences"])
            for i, text in enumerate(item["sentences"]):
                audio_tasks.append((j, i, text, chapter_title, voice, speed, 0))

        # 使用线程池并行处理音频生成
        audio_results = {}
        with ThreadPoolExecutor(max_workers=min(3, len(audio_tasks))) as executor:
            futures = {executor.submit(
                process_audio_segment, *task): task for task in audio_tasks}
            for future in futures:
                task = futures[future]
                j, i = task[0], task[1]  # 只解包需要的索引值
                result = future.result()
                audio_results[(j, i)] = result

        # 检查是否所有音频都成功生成
        if len(audio_results) != len(audio_tasks):
            logger.error(
                f"部分音频生成失败 ({len(audio_results)}/{len(audio_tasks)})，继续处理可用片段")

        # 处理每个图像和音频片段
        final_clips = []

        for j, item in enumerate(data):
            try:
                if "image_path" not in item or not os.path.exists(item["image_path"]):
                    continue
                image = Image.open(item["image_path"]).convert("RGB")
            except Exception as e:
                logger.error(f"无法打开图像文件 {item['image_path']}: {e}")
                continue

            text_segments = item["sentences"]

            for i, text in enumerate(text_segments):
                # 获取对应的音频文件
                audio_path = audio_results.get((j, i))
                if not audio_path:
                    logger.warning(f"跳过片段 {j}_{i}，无法找到对应音频：{text}")
                    continue
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    logger.warning(f"跳过片段 {j}_{i}，无法找到对应音频：{text}")
                    continue
                # 加载音频
                try:
                    audio_clip = AudioFileClip(audio_path)
                    # 缩短音频时间，平滑过渡
                    adjusted_duration = max(0.1, audio_clip.duration - 0.08)
                except Exception as e:
                    logger.error(f"音频加载失败 {audio_path}，{text}: {e}")
                    continue

                # 创建图像副本并添加动态效果
                img = image.copy()
                img_clip = ImageClip(np.array(img)).set_duration(
                    adjusted_duration)
                img_clip = create_dynamic_effect(img_clip, "none")
                img_clip = img_clip.set_audio(audio_clip)

                # 获取图像尺寸
                orig_w, orig_h = img_clip.size

                # 创建字幕
                max_width = orig_w - 100  # 设置左右边距
                wrapped_text = wrap_text(text, font, max_width)

                # 创建空白透明图像作为字幕背景
                text_img = Image.new('RGBA', (orig_w, orig_h), (0, 0, 0, 0))
                text_draw = ImageDraw.Draw(text_img)

                # 计算字幕位置
                line_height = font.getsize("测试")[1]  # 使用固定高度计算
                total_text_height = len(wrapped_text) * line_height
                y_text = orig_h - total_text_height - 30  # 设置文本的垂直位置

                # 绘制字幕文本
                for line in wrapped_text:
                    text_width, text_height = text_draw.textsize(line, font)
                    x_text = (orig_w - text_width) // 2  # 居中对齐

                    # 添加文本阴影增强可读性
                    text_draw.text((x_text+2, y_text+2), line,
                                   (0, 0, 0, 180), font=font)  # 阴影
                    text_draw.text((x_text, y_text), line,
                                   (240, 167, 50), font=font)      # 文本

                    y_text += line_height

                # 创建字幕视频轨道
                text_clip = ImageClip(
                    np.array(text_img)).set_duration(adjusted_duration)

                # 合成图像轨道和字幕轨道
                composed_clip = CompositeVideoClip([img_clip, text_clip])

                # 添加淡入淡出效果
                # composed_clip = composed_clip.fadein(0.2).fadeout(0.2)

                final_clips.append(composed_clip)
        # 合成最终视频
        if final_clips:
            final_clip = concatenate_videoclips(final_clips, method="compose")
            # 确保固定4:3尺寸
            w, h = final_clip.size
            if abs(w/h - 4/3) > 0.01:  # 如果不是精确的4:3比例
                # 计算4:3比例的新尺寸
                if w/h > 4/3:  # 太宽了，保持高度
                    new_width = int(h * 4/3)
                    new_height = h
                else:  # 太高了，保持宽度
                    new_width = w
                    new_height = int(w * 3/4)
                final_clip = final_clip.resize((new_width, new_height))

            final_clip.write_videofile(
                output_video, fps=24, codec="libx264", audio_codec="aac")
            print("视频生成成功:", output_video)
        return True
    except Exception as e:
        traceback.print_exc()
        return False


if __name__ == '__main__':

    data = [
        {
            "text": "背景：南京城钟山脚下，紫禁城内\n氛围：灯火如昼的紫禁城\n时间：黄昏将近\n描述：连绵的琉璃屋脊宛如长龙，一直延伸至紫禁城一角的东 宫\n视觉元素：灯火辉煌的宫殿，琉璃屋脊\n人物：无\n图片风格：动漫风格",
            "image_prompt": "masterpiece, best quality, 4k, anime style, best lighting, depth of field, detailed environment, vast and majestic Forbidden City, glowing lanterns, expansive roof ridges shining under the bright lights, the Eastern Palace courtyard rising and falling in a solemn and mysterious atmosphere, suburb landscape, intricate details of the tiled roof, twilight sky with warm tones, sprawling architectural complex, grand scale, serene and majestic environment. ,comic style, very detailed, ultra high resolution, 2K, masterpiece,",
            "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-0-LibLib.png",
            "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/fca38834-3ef6-479b-aeeb-1b9374c3927e_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936202&Signature=b10kq5Bo3q%2B%2B3P1bPt0wJxyoFIo%3D",
            "sceneContent": "紫禁城内灯火通明，琉璃屋脊 在灯光下闪耀，东宫的院落起伏，显得庄严而神秘。",
            "sentences": [
                "紫禁城内灯火通明，琉璃屋脊在灯光下闪耀，东宫的院落起伏，显得庄严而神秘。"
            ]
        },
        {
            "text": "背景：东宫的院落中\n氛围：紧张而急促的气氛\n时间：黄昏时分\n描述：有人急得要跺脚，口里叫着：“站住，站住……”\n视觉元素：急促奔  跑的身影，灯火下的院落\n人物：穿着衮服的人\n图片风格：动漫风格",
            "image_prompt": "A man in a royal robe (衮服) rushing through the courtyard of the Eastern Palace, shouting \"Stop! Stop!\", with a tense and hurried atmosphere, the courtyard illuminated by lanterns, the setting  sun casting shadows on the ground, anime style, masterpiece, best quality, 4k, illustration style, best lighting, depth of field, detailed character, detailed environment. ,comic style, very detailed, ultra high resolution, 2K, masterpiece,",
            "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-1-LibLib.png",
            "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/8408f782-8ef5-41a3-9fd2-e1717417267f_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936196&Signature=WUbc2Sye5pL0aPcQTQSGW9r%2Bcwg%3D",
            "sceneContent": "一个穿着衮服的人在东宫的院落中急促奔跑，口中不断呼喊着：“站住，站住……”",
            "sentences": [
                "一个穿着衮服的人在东宫的院落中急促  奔跑，口中不断呼喊着：“站住，站住……”"
            ]
        },
        # {
        #     "text": "背景：月洞附近\n氛围：紧张的对峙\n时间：黄昏时分\n描述：少年从月洞探出脑袋，一脸心有余  悸地道：“姐夫若不打我，我便不跑。”\n视觉元素：月洞，少年的惊慌表情\n人物：十二三岁的少年\n图片风格：动漫风格",
        #     "image_prompt": "A 12-13 year old boy peeks out from a moon-shaped hole, his face showing panic and fear, speaking to someone pursuing him, \"If my brother-in-law doesn't hit me, I won't run.\" The scene is set at dusk, creating a tense atmosphere of confrontation. The environment includes the moon-shaped hole, with visible shadows and details. The boy's facial expression is emphasized for clarity. The image is drawn in an anime style,  featuring detailed characters and environments, masterpiece, best quality, 4k, anime illustration style, best lighting, depth of field. ,comic style, very detailed, ultra high resolution, 2K, masterpiece,",
        #     "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-2-LibLib.png",
        #     "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/2f7c9aef-8677-4920-8634-360787ab1a70_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936202&Signature=o6YJuiyBmXQewa1FDMXZRhhLJYo%3D",
        #     "sceneContent": "十二三岁的少年从月洞探  出头，脸上带着惊慌，对追赶他的人说道：“姐夫若不打我，我便不跑。”",
        #     "sentences": [
        #         "十二三岁的少年从月洞探出头，脸上带着惊慌，对追赶他的人说 道：“姐夫若不打我，我便不跑。”"
        #     ]
        # },
        # {
        #     "text": "背景：月洞附近\n氛围：紧张的对话\n时间：黄昏时分\n描述：穿着衮服的人瞪大眼睛，咬牙切齿地道：“子不教，父之过，今日若不狠狠教训你，明日你岂不还要上房揭瓦？”\n视觉元素：穿着衮服的人愤怒的表情\n人物：穿着衮服的人\n图片风格：动漫风格",
        #     "image_prompt": "A highly detailed anime-style illustration of a man in traditional royal robes (衮服), showing an intense and angry expression, wide eyes, gritted teeth, as he sternly scolds a young boy: \"If a child is not taught, it is the father's fault. If I don't discipline you severely today, tomorrow you'll be climbing roofs and causing havoc!\" The scene takes place near a moon-shaped archway during dusk, creating a tense atmosphere. The background features a serene yet ominous environment with soft twilight colors. Masterpiece, best quality, 4k, anime style, dramatic lighting, depth of field, highly detailed character expressions, detailed background. ,comic style, very  detailed, ultra high resolution, 2K, masterpiece,",
        #     "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-3-LibLib.png",
        #     "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/b32bb2f3-cfb8-461e-9b35-197f0a1d5fea_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936200&Signature=nuMm9zSOGrbHyBYnxPGSUSwFgMc%3D",
        #     "sceneContent": "穿着衮服的人愤怒地瞪大眼睛，对少年吼道： “子不教，父之过，今日若不狠狠教训你，明日你岂不还要上房揭瓦？”",
        #     "sentences": [
        #         "穿着衮服的人愤怒地瞪大眼睛，对少年吼道：“子不教，父之过，  今日若不狠狠教训你，明日你岂不还要上房揭瓦？”"
        #     ]
        # },
        # {
        #     "text": "背景：月洞附近\n氛围：紧张的对话\n时间：黄昏时分\n描述：少年立即高声反驳：“可  你不是我爹啊。”\n视觉元素：少年的倔强表情\n人物：十二三岁的少年\n图片风格：动漫风格",
        #     "image_prompt": "A 12-year-old boy with a stubborn expression, shouting \"But you're not my father!\" with a tense atmosphere, standing near a moonlit cave during the dusk, anime style, detailed character, detailed environment, masterpiece, best quality, 4k, illustration style, best lighting, depth of field. ,comic style, very  detailed, ultra high resolution, 2K, masterpiece,",
        #     "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-4-LibLib.png",
        #     "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/6573154a-de6d-4ed3-885b-47e1b653b563_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936197&Signature=K1RWQWFmbZJVjow3y3np9XkUIZw%3D",
        #     "sceneContent": "少年立即高声反驳：“可你不是我爹啊。”",
        #     "sentences": [
        #         "少年立即高声反驳：“可你不是我爹啊。”"
        #     ]
        # },
        # {
        #     "text": "背景：月洞附近\n氛围：紧张的对话\n时间：黄昏时分\n描述：穿着衮服的人大喝  道：“我是你姐夫！”\n视觉元素：穿着衮服的人的愤怒表情\n人物：穿着衮服的人\n图片风格：动漫风格",
        #     "image_prompt": "masterpiece, best quality, 4k, anime style, dramatic lighting, depth of field, detailed character, detailed environment,  \n(angry man in traditional royal robe), shouting intensely with a furious expression, \"I am your brother-in-law!\"  \nduring dusk, near a moon-shaped arch, tense and dramatic atmosphere ,comic style, very detailed, ultra high resolution, 2K, masterpiece,",
        #     "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-5-LibLib.png",
        #     "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/3ff275aa-fb68-4be2-aa44-5185e09c02da_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936230&Signature=lf%2FAp9ZucCYbQCIKosq1rPt0Qoc%3D",
        #     "sceneContent": "穿着衮服的人  大喝道：“我是你姐夫！”",
        #     "sentences": [
        #         "穿着衮服的人大喝道：“我是你姐夫！”"
        #     ]
        # },
        # {
        #     "text": "背景：月洞附近\n氛围：紧张的抓捕\n时间：黄昏时分\n描述：几个宦官蹑手蹑脚地靠近，其中一个如恶狗扑食一般，一把将少年拽住。\n视觉元素：宦官们的动作，少年的挣扎\n人物：宦官，十二三岁的少年\n  图片风格：动漫风格",
        #     "image_prompt": "masterpiece, best quality, 4k, anime style, dramatic lighting, intense atmosphere, sunset glow, detailed character, detailed environment, (eunuchs) sneaking quietly, one eunuch suddenly pouncing on a young boy around 12-13 years old, the  boy struggling, other eunuchs quickly surrounding and restraining the boy firmly, intense expression on eunuchs' faces, the boy's fear and desperation, moonlit cave entrance in the background, shadows cast by the setting sun, intricate clothing details of eunuchs and the boy, dynamic action poses, cinematic composition. ,comic style, very detailed, ultra high resolution, 2K, masterpiece,",
        #     "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-6-LibLib.png",
        #     "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/99186a6f-3ecc-4e1a-b6c9-1cfc59c3cc51_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936224&Signature=oHaIcgiS51y2m4mnWxPKOtfV%2BH0%3D",
        #     "sceneContent": "宦官们悄悄靠近，其中一个猛然抓住少年，其他宦官迅速围上来，将少年牢牢控制住。",
        #     "sentences": [
        #         "宦官们悄悄靠近，  其中一个猛然抓住少年，其他宦官迅速围上来，将少年牢牢控制住。"
        #     ]
        # },
        # {
        #     "text": "背景：月洞附近\n氛围：紧张后的放松\n时间：黄昏时分\n描述：被称  为太子的穿着衮服的人长舒一口气，背着手，气定神闲地踱步上前。\n视觉元素：太子的悠然姿态\n人物：太子\n图片风格：动漫风格",
        #     "image_prompt": "A dignified prince wearing a ceremonial robe, exhaling deeply with hands clasped behind his back, leisurely walking forward, regaining composure and majesty, near a moon-shaped cave entrance, twilight atmosphere, tranquil after tension, anime style, masterpiece, best quality, 4k, illustration style, best lighting, depth of field, detailed character, detailed environment. ,comic style, very detailed, ultra high  resolution, 2K, masterpiece,",
        #     "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-7-LibLib.png",
        #     "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/86f59fa0-1d41-4395-874c-e4380e04fe3d_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936222&Signature=OFzLGQOfaIe14%2BGSU1W1WiOlguI%3D",
        #     "sceneContent": "太子长舒一口气，背着手，慢慢踱步上前，恢复了平静与威严。",
        #     "sentences": [
        #         "太子长舒一口气，背着手，慢慢踱步上前，恢复了平静与威严。"
        #     ]
        # },
        # {
        #     "text": "背景：月洞附近\n氛围：反抗与呼救\n时间：黄昏时分\n描 述：少年不服气地叫道：“你们偷袭，混账东西，回头我收拾你们。”\n视觉元素：少年的愤怒表情\n人物：十二三岁的少年\n图片风格：动漫风格",
        #     "image_prompt": "(masterpiece), (best quality), (4k), (anime style), (best lighting), (cinematic composition), (depth of field), \n\nA 12-13-year-old boy in a rage, with a furious expression on his face, struggling against restraints, shouting \"You sneaked up on me, you scoundrels,  I'll get you back!\", \n\nnear the moon gate, dusk atmosphere with a hint of rebellion and distress, detailed character, intricate facial  expressions, dynamic pose, intricate clothing, \n\nbackground with intricate moon gate details, warm and dim twilight lighting, slight lens flare, detailed environment. ,comic style, very detailed, ultra high resolution, 2K, masterpiece,",
        #     "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-8-LibLib.png",
        #     "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/1452ecc2-0bdb-438f-853b-9b4ba0fefcfd_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936232&Signature=gFdQKpaI%2FbHWZ163DYo1RkfQTHI%3D",
        #     "sceneContent": "少年愤怒地挣扎，口中不服气地喊道：“你们偷袭，混账东西，回头我收拾你们。”",
        #     "sentences": [
        #         "少年愤怒地挣扎，口中不服气地喊道：“  你们偷袭，混账东西，回头我收拾你们。”"
        #     ]
        # },
        # {
        #     "text": "背景：月洞附近\n氛围：呼救与挣扎\n时间：黄昏时分\n描述：少年继续挣扎，口中喊道：“阿姐  ，阿姐，救命啊！”\n视觉元素：少年的呼救动作\n人物：十二三岁的少年\n图片风格：动漫风格",
        #     "image_prompt": "masterpiece, best quality, 4k, anime style, dramatic lighting, depth of field, detailed character, detailed environment, sunset background, moonlight through a cave entrance, a twelve or thirteen-year-old boy in distress, hands flailing, mouth wide open in a desperate cry, \"Sister, Sister, save me!\", intense emotional expression, shadows cast by the cave entrance, faint glow of the setting sun outside, surrounded by an eerie atmosphere, with subtle hints of struggle and despair in the air. ,comic style, very detailed, ultra high resolution, 2K, masterpiece,",
        #     "image_path": "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/images/1741932577-9-LibLib.png",
        #     "image_url": "https://sc-maas.oss-cn-shanghai.aliyuncs.com/outputs/27b2f74c-a5d3-45b9-bade-3f725d005d46_0.png?OSSAccessKeyId=LTAI5tQnPSzwAnR8NmMzoQq4&Expires=1741936227&Signature=k6wu%2FIDcK%2BuXQLJnuaYtkCeST6I%3D",
        #     "sceneContent": "少年拼命挣扎，高声呼喊：“阿姐，阿姐，救命啊！”",
        #     "sentences": [
        #         "少年拼命挣扎，高声呼喊：“阿姐，阿姐，救命啊 ！”"
        #     ]
        # }
    ]
    output_video = "results/7d2ff4c4-5de8-46d0-94f9-ac193e180407/videos/1741932577-LibLib.mp4"
    chapter_title = "7d2ff4c4-5de8-46d0-94f9-ac193e180407"
    voice = "zh-HK-HiuGaaiNeural"

    generate_video_with_subtitles(
        data=data, output_video=output_video, chapter_title=chapter_title, voice=voice)
