import shutil
import time
import requests
import json
import cv2
import os
import textwrap

from PIL import Image
from dotenv import load_dotenv
import numpy as np
import subprocess
import re

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from add_text_to_image import add_text_to_image
from api.text2img_liblib import Text2img
from draw_prompt import generate_prompt
from utils.image_to_video import generate_video_with_subtitles

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
def generateImage(model, prompt,timeStamp):
    liblib = Text2img()
    # 简易模式：旗舰版任务，如果不需要请注释
    progress = liblib.ultra_text2img(prompt)
    # 将图片写入到 images 目录下，每个图片使用(时间戳+model).png 来命名
    imagePath = "images/" + timeStamp + \
        "-" + model.split("/")[-1] + ".png"
    if progress['data'].get('generateStatus') == 5:
        # 设置下载失败尝试次数
        download_retry = 3
        image = progress['data']['images'][0]['imageUrl']
        # 得到https://liblibai-tmp-image.liblib.cloud/img/c7bb262ad8c84ddc87c3835f55a85137/6e473d6798281d4fc305b8e46ef3035525f6b26741ed41d72a8eda393c489849.png这样地址
        # 通过requests.get获取图片
        while download_retry > 0:
            try:
                response = requests.get(image)
                with open(imagePath, 'wb') as f:
                    f.write(response.content)
                break
            except Exception as e:
                print(f"下载图片失败: {e}")
                download_retry -= 1
    return imagePath

def clear_folder(folder_path):
    """清空指定文件夹中的文件"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def split_sentences(text):
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)  # 英文省略号
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)  # 中文省略号
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    sentences = text.split("\n")
    # 移除空白的句子
    sentences = [sentence.strip()
                 for sentence in sentences if sentence.strip()]
    return sentences


def convertTextToVideo(model, text):
    # 将文本段落进行分句
    sentences = split_sentences(text)

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

    # 为输入段落生成图片
    timeStamp = str(int(time.time()))
    # text拼接提示修饰词very detailed, ultra high resolution, 32K UHD, best quality, masterpiece
    text = text + " ,very detailed, ultra high resolution, 2K, best quality, masterpiece"
    image_path = generateImage(model, text,timeStamp)
    # 调用函数生成视频
    output_video_path = "videos/" + timeStamp + \
        "-" + model.split("/")[-1] + ".mp4"
    success = generate_video_with_subtitles(image_path, sentences, output_video_path)
    if success:
        return output_video_path


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
    text_test = '''
   Have you heard? Cao Xiong laoshi had just competed with the academy’s number one teacher from the bottom, Teacher Zhang Xuan!
'''
    convertTextToVideo("stable-diffusion-v1-5", text_test)


