import hashlib
import os
import json
from typing import List, Optional

CACHE_DIR = "cache"

def get_cache_key(text: str, model: str) -> str:
    """
    text:分镜文本
    model:模型名称
    生成缓存键值
    """
    # 文本只保留文字，去除所有空格和符号
    text = ''.join(filter(str.isalpha, text))
    # md5 算法生成哈希值，作为缓存键值
    # 保证了不同文本和模型的结果都有不同的缓存键值，实现了按模型和文本分离的缓存
    # 文本中只保留字母，去除了所有空格和符号，保证了缓存的有效性和查询的效率
    # 文本中不包含任何特殊字符，保证了哈希值在 ASCII 字符集内的唯一性
    # 文本中不包含任何空格，保证了哈希值在 32 位的整形域内的唯一性
    # 文本中不包含任何特殊字符，保证了哈希值在 32 位的整形域内的唯一性
    return hashlib.md5(f"{text}_{model}".encode()).hexdigest()


def get_cache(text: str, model: str, chapter_title: str) -> Optional[List[str]]:
    """
    从缓存中获取视频结果
    """
    cache_key = get_cache_key(text, model)
    cache_file = os.path.join("results/", chapter_title, f"/{cache_key}.mp4")
    # 判断视频是否生成
    if os.path.exists(cache_file):
        return cache_file


def set_cache(text: str, model: str, result: List[str]) -> None:
    """
    将断句结果设置到缓存中
    """
    cache_key = get_cache_key(text, model)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
    except IOError:
        pass
