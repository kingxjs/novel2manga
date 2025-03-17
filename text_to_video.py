import time
from concurrent.futures import ThreadPoolExecutor
from logging import handlers
from logging.handlers import RotatingFileHandler

import requests
import os
from dotenv import load_dotenv
import re
from api.llm_bk import generate_prompt, take_prompt, text2imageToChat, text2image
from api.text2img_liblib import Text2img
from utils.cache_utils import get_cache_key, get_cache
from utils.image_to_video import generate_video_with_subtitles
# -*- coding: utf-8 -*-
import logging

FIXED_NUM_THREADS = 5  # 固定线程数

# 1、设置全局的日志格式和级别
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# 2、获取logger （给日志器起个名字 "__name__"）
# __name__内置变量模块名称，轻松地识别出哪个模块产生了哪些日志消息（主程序模块）
logger = logging.getLogger(__name__)
if not os.path.exists("logs"):
    os.makedirs("logs")
# 3、创建文件处理器，指定日志文件和日志级别（局部）---文件输出FileHandle（输出到指定文件 logs/text_to_video.log）
file_handler = RotatingFileHandler(
    'logs/text_to_video.log',
    maxBytes=1024*1024*5,  # 5MB
    backupCount=3,         # 保留3个备份
    encoding='utf-8',
    delay=True             # 延迟打开文件，直到第一次写入
)
file_handler.setLevel(logging.INFO)
# 设置日志格式
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', '%m/%d/%Y %H:%M:%S'))

# 添加文件处理器到logger
logger.addHandler(file_handler)

# logging.debug('debug级别，一般用来打印一些调试信息，级别最低')
# logging.info('info级别，一般用来打印一些正常的操作信息')
# logging.warning('waring级别，一般用来打印警告信息')
# logging.error('error级别，一般用来打印一些错误信息')
# logging.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')

load_dotenv('.env', override=True)

# 尝试加载本地开发环境变量文件
load_dotenv('.local.env', override=True)

# 获取当前脚本所在的目录
current_directory = os.getcwd()

# 读取环境变量
api_token = os.getenv('API_TOKEN')

headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}


# auto try
def generateImage(model, prompt, chapter_title, timeStamp, controlImage=None):
    if not os.path.exists("results/"+chapter_title+"/images"):
        os.makedirs("results/"+chapter_title+"/images")
    imagePath = "results/"+chapter_title+"/images/" + timeStamp + \
                "-" + model.split("/")[-1] + ".png"
    # liblib = Text2img(interval=10)
    # # 简易模式：旗舰版任务，如果不需要请注释
    # progress = liblib.ultra_text2img(prompt, controlImage=controlImage)
    # # 记录响应体-包含生成图片的状态、图片地址、随机种子等信息
    # logger.info(f"生成图片响应体: {progress}")
    # # 新建results/{chapter_title}/images文件夹

    # # 将图片写入到 images 目录下，每个图片使用(时间戳+model).png 来命名
    # imagePath = "results/"+chapter_title+"/images/" + timeStamp + \
    #             "-" + model.split("/")[-1] + ".png"
    # imageUrl = ''
    # if progress['data'].get('generateStatus') == 5:
    #     # 设置下载失败尝试次数
    #     download_retry = 3
    #     imageUrl = progress['data']['images'][0]['imageUrl']
    #     logger.info(f"图片地址: {imageUrl}")
    #     # 得到https://liblibai-tmp-image.liblib.cloud/img/c7bb262ad8c84ddc87c3835f55a85137/6e473d6798281d4fc305b8e46ef3035525f6b26741ed41d72a8eda393c489849.png这样地址
    #     # 通过requests.get获取图片
    #     while download_retry > 0:
    #         try:
    #             response = requests.get(imageUrl)
    #             with open(imagePath, 'wb') as f:
    #                 f.write(response.content)
    #             break
    #         except Exception as e:
    #             print(f"下载图片失败: {e}")
    #             download_retry -= 1
    imageUrl = text2imageToChat(prompt, model, imagePath=controlImage)
    logger.info(f"图片地址: {imageUrl}")
    download_retry = 3
    # 得到https://liblibai-tmp-image.liblib.cloud/img/c7bb262ad8c84ddc87c3835f55a85137/6e473d6798281d4fc305b8e46ef3035525f6b26741ed41d72a8eda393c489849.png这样地址
    # 通过requests.get获取图片
    while download_retry > 0:
        try:
            response = requests.get(imageUrl)
            with open(imagePath, 'wb') as f:
                f.write(response.content)
            return imagePath, imageUrl
        except Exception as e:
            print(f"下载图片失败: {e}")
            imageUrl = text2imageToChat(prompt, model, imagePath=controlImage)
            logger.info(f"图片地址: {imageUrl}")
            download_retry -= 1
    return "", imageUrl


def clear_folder(folder_path):
    """清空指定文件夹中的文件"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def chunk_max_string(string, charactersDefiningChunking=["。", "！", "？", "；", "?", "!", "; ", "\n\n"], maxNumberOfCharactersPerChunk=20):
    """
    根据指定的字符定义分块，并确保每个块的字符数不超过最大字符数。

    :param string: 待分块的字符串
    :param charactersDefiningChunking: 定义分块的字符列表 ["，", "。", "！", "？", "；", ",", "?", "!", "; "]
    :param maxNumberOfCharactersPerChunk: 每个块的最大字符数，默认为10
    :return: 分块后的字符串列表
    """
    chunks = []  # 存储分块后的字符串列表
    last_chunk_end = 0  # 上一个块的结束位置，初始化为0表示从字符串的开头开始

    # 遍历整个字符串
    # 终止条件改为len(string)+1，确保最后一个字符也能被考虑
    for char_index in range(1, len(string) + 1):
        # 如果当前块的字符数已经达到了最大字符数
        if char_index - last_chunk_end == maxNumberOfCharactersPerChunk:
            found_chunk_end = False  # 标记是否找到了合适的分块位置
            chars_away = 1  # 用于在当前字符的前后扩展搜索区间，初始为1

            # 扩展搜索区间，尝试寻找一个合适的分块位置
            while not found_chunk_end and chars_away < maxNumberOfCharactersPerChunk:
                # 遍历所有定义为分块字符的标点符号
                for chunk_char in charactersDefiningChunking:
                    # 在当前位置前后 `chars_away` 个字符内寻找分隔符
                    chunk_char_index = string[char_index -
                                              chars_away:char_index + chars_away + 1].find(chunk_char)

                    # 如果找到了分隔符
                    if chunk_char_index > -1:
                        # 计算当前分块的结束位置
                        chunk_end = char_index - chars_away + chunk_char_index + 1
                        # 如果分块后的长度超出限制，则不做分块
                        if chunk_end - last_chunk_end > maxNumberOfCharactersPerChunk:
                            continue  # 如果超出最大字符数，跳过当前分块

                        # 将分块内容添加到结果列表中
                        chunks.append(string[last_chunk_end:chunk_end])
                        # 更新上一个块的结束位置
                        last_chunk_end = chunk_end
                        # 标记已经找到了分块的位置
                        found_chunk_end = True
                        break  # 找到分块位置后跳出内层循环

                # 扩展搜索区间，增加一个字符距离
                chars_away += 1

            # 如果在扩展区间内仍然没找到合适的分块位置，直接切分
            if not found_chunk_end:
                # 如果当前块的长度已达到最大限制，则直接添加该块
                chunks.append(string[last_chunk_end:char_index])
                last_chunk_end = char_index  # 更新上一个块的结束位置

    # 如果字符串剩余部分的长度小于最大块大小，则直接将剩余部分加入最后一个块
    if last_chunk_end < len(string):
        chunks.append(string[last_chunk_end:])

    # 返回分块后的字符串列表
    return chunks


def split_sentences(text):
    """
    将文本分割成字幕句子，适当处理标点符号。
    该函数确保每个字幕是一个完整的想法表达，正确处理嵌套的引号和括号。
    """
    # 替换常见的换行符和多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)

    # 使用栈跟踪引号和括号的状态
    quote_stack = []  # 用于跟踪引号
    bracket_stack = []  # 用于跟踪括号
    sentence_breaks = []  # 记录句子分割点

    for i, char in enumerate(text):
        # 跟踪引号状态
        if char in '"\'""''':
            # 检查是否匹配栈顶的引号（闭合引号）
            if quote_stack and quote_stack[-1] == char:
                quote_stack.pop()
            else:
                quote_stack.append(char)

        # 跟踪括号状态
        elif char in '([{【（《':
            bracket_stack.append(char)
        elif char in ')]}】）》':
            if bracket_stack:  # 确保栈不为空
                bracket_stack.pop()

        # 只有当引号和括号都正确闭合时，才考虑在句号等标点后分割
        elif char in '。！？!?' and not quote_stack and not bracket_stack:
            # 记录句子结束位置（包括标点）
            sentence_breaks.append(i + 1)

        # 处理省略号
        elif char == '.' and i + 2 < len(text) and text[i:i+3] == '...' and not quote_stack and not bracket_stack:
            # 跳过接下来的两个点（已经处理了第一个点）
            i += 2
            sentence_breaks.append(i + 1)

        elif char == '…' and not quote_stack and not bracket_stack:
            sentence_breaks.append(i + 1)

    # 根据句子分割点分割文本
    sentences = []
    start = 0
    for end in sentence_breaks:
        sentences.append(text[start:end].strip())
        start = end

    # 添加最后一个句子（如果有）
    if start < len(text):
        sentences.append(text[start:].strip())

    # 处理句子列表，合并过短的片段
    processed_sentences = []
    current_sentence = ""

    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue

        # 如果只是标点符号或非常短（少于2个字符），与相邻句子合并
        if len(stripped) < 2 or re.match(r'^[\s\.\,\!\?\;\:\"\'\"\'\、\。\，\！\？\；\：\（\）\【\】\《\》\—\…\[\]\(\)\s]+$', stripped):
            if current_sentence:
                # 附加到前一句
                current_sentence += " " + stripped
            elif processed_sentences:
                # 附加到最后一个处理过的句子
                processed_sentences[-1] += " " + stripped
        else:
            # 如果有正在构建的当前句子，完成它
            if current_sentence:
                processed_sentences.append(current_sentence)

            # 开始新句子
            current_sentence = stripped

    # 添加最后一个句子（如果有）
    if current_sentence:
        processed_sentences.append(current_sentence)

    # 最终清理 - 删除任何多余的空格
    processed_sentences = [re.sub(r'\s+', ' ', s).strip()
                           for s in processed_sentences]
    print(processed_sentences)
    return processed_sentences

# 清空上一本书的生成结果


def clear_results():
    # 清空 images 文件夹
    if not os.path.exists("images"):
        os.makedirs("images")
    clear_folder("images")
    # 清空 voices 文件夹
    if not os.path.exists("voices"):
        os.makedirs("voices")
    clear_folder("voices")
    if not os.path.exists("videos"):
        os.makedirs("videos")
    clear_folder("videos")


def convertTextToImage(model, data, chapter_title):
    pervText = ""
    if "prevScene" in data and data["prevScene"]:
        pervText = data["prevScene"]["text"] + \
            "\n剧本台词："+data["prevScene"]["sceneContent"]

    prompt = generate_prompt(
        data["text"]+"\n剧本台词："+data["sceneContent"], pervText=pervText) + " ,comic style, very detailed, ultra high resolution, 2K, masterpiece,"

    logger.info(f"生成图片提示词: {prompt}")
    image_path, imageUrl = generateImage(
        model, prompt, chapter_title, str(int(time.time())), data["control_image"] if "control_image" in data else None)
    result = {
        "image_prompt": prompt,
    }
    if image_path:
        result["image_path"] = image_path
        result["image_url"] = imageUrl
    return result


def convertTextToVideo(model, texts, chapter_title, use_cache=True, voice: str = "", speed: int = 30):

    # 如果使用缓存，尝试从缓存中获取分句结果，跳过该分镜
    # if use_cache:
    #     cached_result = get_cache(text, model, chapter_title)
    #     if cached_result:
    #         print(f"[+] 从缓存中获取结果: {cached_result}")
    #         return cached_result

    # 记录当前分镜处理的句子
    # logger.info(f"分镜段落(150字左右): {text}")

    # 为输入段落生成图片
    timeStamp = str(int(time.time()))
    cache_key = timeStamp
    # 生成场景
    # texts = take_prompt(text,num=20)
    print(f"分镜数: {len(texts)}")
    results = []
    for i, item in enumerate(texts):
        if i > 0:
            item["prevScene"] = texts[i-1]

    # 使用线程池并行处理图像生成
    with ThreadPoolExecutor(max_workers=FIXED_NUM_THREADS) as executor:
        def process_scene(item_with_index):
            i, item = item_with_index
            logger.info(f"分镜段落: {item}")
            if "image_prompt" in item:
                item["index"] = i
                item["sentences"] = chunk_max_string(item["sceneContent"])
                return item
            pervText = ""
            if "prevScene" in item and item["prevScene"]:
                pervText = item["prevScene"]["text"] + \
                    "\n剧本台词："+item["prevScene"]["sceneContent"]

            # text拼接提示修饰词very detailed, ultra high resolution, 32K UHD, best quality, masterpiece
            prompt = generate_prompt(
                item["text"]+"\n剧本台词："+item["sceneContent"], pervText=pervText) + " ,comic style, very detailed, ultra high resolution, 2K, masterpiece,"
            logger.info(f"生成图片提示词: {prompt}")

            # # 获取控制图像（如果有的话）
            # control_image = None
            # if results and len(results) > 0:
            #     control_image = results[len(results)-1]["image_url"]

            # image_path, imageUrl = generateImage(model, prompt, chapter_title, cache_key+"-"+str(i), control_image)

            # if image_path:
            #     return {
            #         "text": item["text"],
            #         "image_prompt": prompt,
            #         "image_path": image_path,
            #         "image_url": imageUrl,
            #         "sceneContent": item["sceneContent"],
            #         "sentences": split_sentences(item["sceneContent"]),
            #         "index": i  # 保存原始索引以便保持顺序
            #     }

            return {
                "text": item["text"],
                "image_prompt": prompt,
                "image_path": "",
                "image_url": "",
                "sceneContent": item["sceneContent"],
                "sentences": chunk_max_string(item["sceneContent"]),
                "index": i  # 保存原始索引以便保持顺序
            }

        # 将索引与项目配对，以便我们可以在处理后正确排序结果
        scene_results = list(executor.map(process_scene, enumerate(texts)))

        # 过滤掉None结果并按原始索引排序
        scene_results = [
            result for result in scene_results if result is not None]
        scene_results.sort(key=lambda x: x.pop("index"))  # 移除临时索引并排序

        # 添加到结果列表
        results.extend(scene_results)

        for i, result in enumerate(results):
            if "image_path" in result and result["image_path"] and os.path.exists(result["image_path"]):
                continue


            # 获取控制图像（如果有的话）
            control_image = None
            if results and len(results) > 0 and "image_path" in results[len(results)-1] and results[len(results)-1]["image_path"] and os.path.exists(results[len(results)-1]["image_path"]):
                control_image = results[len(results)-1]["image_path"]

            image_path, imageUrl = generateImage(
                model, result["image_prompt"], chapter_title, cache_key+"-"+str(i), control_image)

            if image_path:
                result["image_path"] = image_path
                result["image_url"] = imageUrl

    # 新建video文件夹
    if not os.path.exists("results/"+chapter_title+"/videos"):
        os.makedirs("results/"+chapter_title+"/videos")
    # 调用函数生成视频
    output_video_path = "results/"+chapter_title+"/videos/" + cache_key + \
                        "-" + model.split("/")[-1] + ".mp4"
    success = generate_video_with_subtitles(
        results, output_video_path, chapter_title, voice=voice, speed=speed)
    if success:
        return output_video_path, results
    return "", results


def batchConvertTextToVideo(model, file_path, chapter_example, num_threads: int = FIXED_NUM_THREADS):
    # 对小说文件按章节进行切分，将章节对应的分镜话语切分出来
    if split_novel_auto(file_path, chapter_example):
        # 读取切分后的分镜文件夹
        chapter_dir = os.path.join(os.getcwd(), "results")
        # 遍历每个章节文件夹
        for chapter in os.listdir(chapter_dir):
            chapter_path = os.path.join(chapter_dir, chapter)
            # 遍历每个分镜文件
            for segment in os.listdir(chapter_path):
                segment_path = os.path.join(chapter_path, segment)
                # 读取分镜内容
                with open(segment_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # 多线程执行 convertTextToVideo
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    def process_segment(text):
                        # 去掉segment文件后缀
                        return convertTextToVideo(model, text, chapter+"/"+segment.split(".")[0])
                    video_path = list(executor.map(process_segment, [text]))
                    logger.info(f"生成视频路径: {video_path}")
    # 所有分镜视频生成后，合并所有分镜视频成完整的一个视频
    # merge_videos(video_paths)


def infer_chapter_pattern(chapter_example):
    """
    根据用户提供的章节标题示例推测正则表达式。
    """
    # 定义常见的章节标题模式
    patterns = [
        r'^\d+\s+[\u4e00-\u9fa5]+',  # "1 天才的陨落" 类似数字开头 + 中文标题
        r'(第[一二三四五六七八九十百千]+章)',  # "第一章" 类似中文数字 + 章
        r'(Chapter\s+\d+)',  # "Chapter 1" 类似英文
        # Chapter One
        r'(Chapter\s+[A-Z][a-z]+)',
        r'^\d+\.',  # "1. 天才的陨落" 数字加点
        r'^\d+\s*[、.]?\s*[\u4e00-\u9fa5]+',  # 允许各种分隔符，如空格、顿号、点号
    ]

    for pattern in patterns:
        if re.match(pattern, chapter_example.strip()):
            return pattern

    # 默认回退匹配任意数字 + 空格 + 中文标题
    return None


def split_novel_auto(novel_file, chapter_example, min_length=100, max_length=150):
    """
    根据示例自动推测章节正则，切分章节和分镜。
    """
    # 推测章节正则表达式
    chapter_pattern = infer_chapter_pattern(chapter_example)
    if chapter_pattern is None:
        print("无法推测章节正则表达式，请手动指定。")
        return None

    print(f"推测章节正则表达式: {chapter_pattern}")

    # 读取小说文件
    with open(novel_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按章节切分
    matches = list(re.finditer(chapter_pattern, content))
    chapters = []
    chapter_titles = []

    # 遍历正则匹配结果，提取章节标题和对应的内容
    for i, match in enumerate(matches):
        start = match.end()
        chapter_titles.append(match.group().strip())  # 提取章节标题
        if i + 1 < len(matches):
            end = matches[i + 1].start()
            chapters.append(content[start:end].strip())  # 提取章节内容
        else:
            chapters.append(content[start:].strip())  # 最后一章到结尾部分

    # 遍历每章进行分镜切分
    for i, chapter in enumerate(chapters):
        chapter_title = chapter_titles[i] if i < len(
            chapter_titles) else f"Chapter {i + 1}"
        chapter_dir = os.path.join(
            os.getcwd(), "results", chapter_title.strip())
        os.makedirs(chapter_dir, exist_ok=True)  # 创建章节文件夹

        # 切分为分镜
        # sentences = re.split(r'([。！？\n])', chapter)  # 按标点或换行切分
        # sentences = [s.strip() for s in sentences if s.strip()]
        sentences = split_sentences(chapter)
        segments = []
        segment = ""

        for sentence in sentences:
            if len(segment) + len(sentence) <= max_length:
                segment += sentence
            else:
                if len(segment) >= min_length:
                    segments.append(segment)
                segment = sentence  # 开启新分镜段

        if segment:  # 添加最后一个段落
            segments.append(segment)

        # 将分镜内容保存到文件
        for j, segment in enumerate(segments):
            file_path = os.path.join(
                chapter_dir, f"storyboard--{j + 1:03d}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(segment)
    # 返回写入成功bool值
    return True


def find_file_name_without_extension(file_path):
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    return file_name_without_extension


def convert_time_to_seconds(time):
    hours, minutes, seconds = time.split(':')
    seconds, milliseconds = seconds.split('.')
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)
    total_seconds = (hours * 3600) + (minutes * 60) + \
        seconds + (milliseconds / 1000)
    return total_seconds


if __name__ == '__main__':
    #     text_test = '''
    #    Have you heard? Cao Xiong laoshi had just competed with the academy’s number one teacher from the bottom, Teacher Zhang Xuan!
    # '''
    #     convertTextToVideo("stable-diffusion-v1-5", text_test)

    # batchConvertTextToVideo("liblib", "file/novel.txt", "Chapter One")
    split_sentences("前头跑的人从月洞探出了脑袋来，却是个十二三岁的少年。这少年一脸心有余悸的样子道：“姐夫若不打我，我便不跑。”")
