import os

from dotenv import load_dotenv
from openai import OpenAI  # 修改导入
import json
import time
import re

load_dotenv('.env', override=True)

# 尝试加载本地开发环境变量文件
load_dotenv('.local.env', override=True)

# 读取环境变量
api_key = os.getenv('OPEN_API_KEY')  # 变量名可以保持不变或改为更通用的名称
base_url = os.getenv('OPEN_AI_BASE_URL')  # 如果有自定义API端点需要添加这个变量
image_api_key = os.getenv('IMAGE_OPEN_API_KEY', api_key)  # 变量名可以保持不变或改为更通用的名称
image_base_url = os.getenv('IMAGE_OPEN_AI_BASE_URL',
                           base_url)  # 如果有自定义API端点需要添加这个变量

DEFAULT_OPENAI_MODEL = os.getenv('DEFAULT_OPENAI_MODEL')  # 如果有自定义API端点需要添加这个变量
DEFAULT_OPENAI_IMAGE_MODEL = os.getenv(
    'DEFAULT_OPENAI_MODEL')  # 如果有自定义API端点需要添加这个变量

client = OpenAI(
    api_key=api_key,  # 使用api_key而非api_token参数
    base_url=base_url,  # 如果需要自定义API端点
)


imageClient = OpenAI(
    api_key=image_api_key,  # 使用api_key而非api_token参数
    base_url=image_base_url,  # 如果需要自定义API端点
)
# 默认模型名称修改为OpenAI的模型


async def reinvent_prompt(text, model: str = None):
    if not model:
        model = DEFAULT_OPENAI_MODEL
    MIDJOURNEY_PROMPT = '''ROLE DEFINITION: I am the Storyteller, responsible for telling stories in a vivid and engaging narrative that emphasizes emotion and atmosphere, taking the listener deeper into the story's situation.

    TASK REQUIREMEN T: Please re-create this novel in the style of a storyteller, focusing on the following areas:

    Create an engaging atmosphere that captures the listener's attention.
    Enhance the drama and depth of the story with rhythmic language and subtle emotional portrayals.
    Output the result directly without any other explanation.
    '''
    response_stream = client.chat.completions.create(
        stream=True,
        model=model,
        messages=[
            {"role": "system",
             "content": MIDJOURNEY_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    # 流式返回结果
    for chunk in response_stream:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            content = chunk.choices[0].delta.content
            if content:
                yield content

# 封装上述代码成工具函数


def generate_prompt(text, model: str = None, retry: int = 0) -> str:  # 默认模型名称修改为OpenAI的模型
    if not model:
        model = DEFAULT_OPENAI_MODEL
    #     MIDJOURNEY_PROMPT = '''Hello, you are an expert in generating midjourney prompt words, users give you a description, you generate the corresponding prompt words!

    #     Keep in mind that the prompt words you are generating are in English and may be in a format similar to:

    #     As the sun gently kissed the horizon, casting a warm golden glow across the bustling city streets, a captivating sight unfolded before my eyes. A graceful woman, adorned in an elegant dress that fluttered in the gentle breeze, held a porcelain cup of steaming coffee in her delicate hands.

    #    Instead of interpreting any input from the user, without any other nonsense.
    #     '''
    MIDJOURNEY_PROMPT = '''作为 Stable Diffusion Prompt 提示词专家，您将从关键词中创建提示，通常来自 Danbooru 等数据库。

        提示通常描述图像，使用常见词汇，按重要性排列，并用逗号分隔。避免使用"-"或"."，但可以接受空格和自然语言。避免词汇重复。

        为了强调关键词，请将其放在括号中以增加其权重。例如，"(flowers)"将'flowers'的权重增加1.1倍，而"(((flowers)))"将其增加1.331倍。使用"(flowers:1.5)"将'flowers'的权重增加1.5倍。只为重要的标签增加权重。

        提示包括三个部分：**前缀**（质量标签+风格词+效果器）+ **主题**（图像的主要焦点）+ **场景**（背景、环境）。

        *   前缀影响图像质量。像"masterpiece"、"best quality"、"4k"这样的标签可以提高图像的细节。像"illustration"、"lensflare"这样的风格词定义图像的风格。像"bestlighting"、"lensflare"、"depthoffield"这样的效果器会影响光照和深度。

        *   主题是图像的主要焦点，如角色或场景。对主题进行详细描述可以确保图像丰富而详细。增加主题的权重以增强其清晰度。对于角色，描述面部、头发、身体、服装、姿势等特征。

        *   场景描述环境。没有场景，图像的背景是平淡的，主题显得过大。某些主题本身包含场景（例如建筑物、风景）。像"花草草地"、"阳光"、"河流"这样的环境词可以丰富场景。你的任务是设计图像生成的提示。请按照以下步骤进行操作：

        1.  我会发送给您一个图像场景。需要你生成详细的图像描述
        2.  图像描述必须是英文，输出为Positive Prompt。

        示例：

        我发送：二战时期的护士。
        您回复只回复：
        A WWII-era nurse in a German uniform, holding a wine bottle and stethoscope, sitting at a table in white attire, with a table in the background, masterpiece, best quality, 4k, illustration style, best lighting, depth of field, detailed character, detailed environment.
        '''
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system",
                       "content": MIDJOURNEY_PROMPT}, {"role": "user", "content": text}]
        )
        return response.choices[0].message.content
    except Exception as e:
        if retry > 2:
            print(f"Error in generate_prompt: {e}")
            return ""
        return generate_prompt(text, model, retry+1)


# 默认模型名称修改为OpenAI的模型
async def take_prompt_stream(text, model: str = None, num: int = 10, retry: int = 0):
    if not model:
        model = DEFAULT_OPENAI_MODEL
    MIDJOURNEY_PROMPT = '''# 你是一个视觉小说创作者，现在给你一个小说片段，帮我重新创作，并返回多个场景。
    ## 创作要求：
    - 营造引人入胜的氛围，吸引听众的注意力。
    - 用富有节奏感的语言和细腻的情感描写，增强故事的戏剧性和深度。
    - 需要拆分的细一点（场景多一点，至少{num}个，每个场景一句话），尽量保证场景连贯性。
    - 对于当前场景没有的描述，需要联系上下文，补充完整，不要输出无、空等，并且不要脱离故事情节。
    - 每个场景的内容要有连贯性，图片风格要相对一致，不要出现突兀的场景，并且保持动漫风格。
    ## 场景包含：
    - 背景
    - 时间
    - 氛围
    - 描述
    - 视觉元素
    - 人物
    - 图片风格
    - 场景内容（一句话即可，尽量简短，突出重点，可以包含对话）
    '''
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_result",
                "description": "根据小说或剧本提供的场景描述，将其拆分为多个场景。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "contexts": {
                                        "type": "string",
                                        "description": "背景"
                                    },
                                    "timing": {
                                        "type": "string",
                                        "description": "时间"
                                    },
                                    "milieu": {
                                        "type": "string",
                                        "description": "氛围"
                                    },
                                    "descriptive": {
                                        "type": "string",
                                        "description": "描述"
                                    },
                                    "visualElement": {
                                        "type": "string",
                                        "description": "视觉元素"
                                    },
                                    "character": {
                                        "type": "string",
                                        "description": "人物（性别、年龄、体貌特征）"
                                    },
                                    "pictureStyle": {
                                        "type": "string",
                                        "description": "图片风格"
                                    },
                                    "sceneContent": {
                                        "type": "string",
                                        "description": "场景内容（一段即可，尽量简短，突出重点）"
                                    }
                                },
                                "required": ["contexts", "timing", "milieu", "descriptive", "visualElement", "character", "sceneContent"]
                            },
                            "description": "场景列表"
                        }
                    },
                    "required": ["results"]
                }
            }
        }
    ]

    try:
        response_stream = client.chat.completions.create(
            stream=True,
            model=model,
            messages=[{"role": "system",
                       "content": MIDJOURNEY_PROMPT}, {"role": "user", "content": text}],
            tools=tools,
            tool_choice={"type": "function",
                         "function": {"name": "extract_result"}}
        )
        # 流式返回结果
        for chunk in response_stream:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                tool_calls = chunk.choices[0].delta.tool_calls
                if tool_calls:
                    content = tool_calls[0].function.arguments
                    yield content

    except Exception as e:
        if retry > 2:
            print(f"Error in generate_prompt: {e}")
            return
        time.sleep(2)
        take_prompt_stream(text, model, retry+1)


def take_prompt(text, model: str = None, num: int = 10, retry: int = 0):  # 默认模型名称修改为OpenAI的模型
    if not model:
        model = DEFAULT_OPENAI_MODEL
    MIDJOURNEY_PROMPT = '''# 你是一个视觉小说创作者，现在给你一个小说片段，帮我重新创作，并返回多个场景。
    ## 创作要求：
    - 营造引人入胜的氛围，吸引听众的注意力。
    - 用富有节奏感的语言和细腻的情感描写，增强故事的戏剧性和深度。
    - 需要拆分的细一点（每个场景一句话），尽量保证场景连贯性。
    - 对于当前场景没有的描述，需要联系上下文，补充完整，不要输出无、空等，并且不要脱离故事情节。
    - 每个场景的内容要有连贯性，图片风格要相对一致，不要出现突兀的场景，并且保持动漫风格。
    ## 场景包含：
    - 背景
    - 时间
    - 氛围
    - 描述
    - 视觉元素
    - 人物
    - 图片风格
    - 场景内容
    ## 要求：
    - 至少输出{num}个场景
    '''
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_result",
                "description": "根据小说或剧本提供的场景描述，将其拆分为多个场景。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "contexts": {
                                        "type": "string",
                                        "description": "背景"
                                    },
                                    "timing": {
                                        "type": "string",
                                        "description": "时间"
                                    },
                                    "milieu": {
                                        "type": "string",
                                        "description": "氛围"
                                    },
                                    "descriptive": {
                                        "type": "string",
                                        "description": "描述"
                                    },
                                    "visualElement": {
                                        "type": "string",
                                        "description": "视觉元素"
                                    },
                                    "character": {
                                        "type": "string",
                                        "description": "人物（性别、年龄、表情、身穿衣物、动作等体貌特征）"
                                    },
                                    "pictureStyle": {
                                        "type": "string",
                                        "description": "图片风格"
                                    },
                                    "sceneContent": {
                                        "type": "string",
                                        "description": "场景内容（一段即可，尽量简短，突出重点，可以包含对话）"
                                    }
                                },
                                "required": ["contexts", "timing", "milieu", "descriptive", "visualElement", "character", "sceneContent"]
                            },
                            "description": "场景列表"
                        }
                    },
                    "required": ["results"]
                }
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system",
                       "content": MIDJOURNEY_PROMPT}, {"role": "user", "content": text}],
            tools=tools,
            tool_choice={"type": "function",
                         "function": {"name": "extract_result"}}
        )

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            function_response = json.loads(tool_calls[0].function.arguments)
            results = function_response["results"]
            texts = []
            for result in results:
                text = f'''
                背景：{result["contexts"]}
                时间：{result["timing"]}
                氛围：{result["milieu"]}
                描述：{result["descriptive"]}
                视觉元素：{result["visualElement"]}
                人物：{result["character"]}
                图片风格：{result["pictureStyle"]}
                '''
                texts.append({
                    "text": text,
                    "sceneContent": result["sceneContent"]
                })
            return texts
        else:
            # 如果没有返回工具调用，则返回一个默认的空数组
            return []
    except Exception as e:
        if retry > 2:
            print(f"Error in generate_prompt: {e}")
            return []
        time.sleep(2)
        return take_prompt(text, model, retry+1)


def extract_markdown_image_urls(text):
    """
    从 Markdown 文本中提取图片 URL

    参数:
        text (str): 包含 Markdown 图片语法的文本

    返回:
        list: 提取的图片 URL 列表
    """
    # 正则表达式匹配 ![任意文本](URL) 格式
    pattern = r'!\[(.*?)\]\((.*?)\)'

    # 查找所有匹配项
    matches = re.findall(pattern, text)

    # 提取 URL 部分 (第二个捕获组)
    urls = [match[1] for match in matches]

    return urls


def text2imageToChat(prompt, model: str = None, size: str = "1024x1024", imagePath=None, retry: int = 0):
    if not model:
        model = DEFAULT_OPENAI_IMAGE_MODEL
    """
    使用OpenAI的DALL-E模型将文本提示转换为图片

    Args:
        prompt (str): 图片生成提示词
        client: OpenAI客户端实例，如果为None则尝试创建新实例
        size (str): 图片尺寸，可选值："1024x1024", "1792x1024", "1024x1792"
        quality (str): 图片质量，可选值："standard", "hd"
        style (str): 图片风格，可选值："vivid", "natural"
        n (int): 生成图片数量

    Returns:
        dict: 包含生成图片URL的结果字典或错误信息
    """
    try:
        response = imageClient.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        urls = extract_markdown_image_urls(response.choices[0].message.content)
        return urls[0] if urls and len(urls) > 0 else ""
    except Exception as e:
        if retry > 2:
            print(f"Error in text2image: {e}")
            return ""
        print(f"重试: {retry+1}")
        time.sleep(2)
        return text2imageToChat(text, model, size, imagePath, retry+1)


def text2image(prompt, model: str = None, size: str = "1024x1024", imagePath=None, retry: int = 0):

    if not model:
        model = DEFAULT_OPENAI_IMAGE_MODEL
    """
    使用OpenAI的DALL-E模型将文本提示转换为图片

    Args:
        prompt (str): 图片生成提示词
        client: OpenAI客户端实例，如果为None则尝试创建新实例
        size (str): 图片尺寸，可选值："1024x1024", "1792x1024", "1024x1792"
        quality (str): 图片质量，可选值："standard", "hd"
        style (str): 图片风格，可选值："vivid", "natural"
        n (int): 生成图片数量

    Returns:
        dict: 包含生成图片URL的结果字典或错误信息
    """
    try:
        if imagePath:
            with open(imagePath, "rb") as image_file:
                response = imageClient.images.create_variation(
                    model=model,
                    image=image_file,
                    prompt=prompt,
                    n=8,
                    size=size
                )
        else:
            # 调用OpenAI的图像生成API
            response = imageClient.images.generate(
                model=model,
                prompt=prompt,
                n=8,
                size=size
            )

        # 提取生成的图片URL
        result = {
            "success": True,
            "images": [item.url for item in response.data],
            "revised_prompt": response.data[0].revised_prompt if hasattr(response.data[0], 'revised_prompt') else None
        }

        return [item.url for item in response.data]

    except Exception as e:
        if retry > 2:
            print(f"Error in text2image: {e}")
            return []
        print(f"重试: {retry+1}")
        time.sleep(2)
        return text2image(text, model, size, imagePath, retry+1)


if __name__ == '__main__':
    #     text = '''
    # 长康疗养院的三楼，是一片沉寂的植物人病房区。夜色如墨，寒风萧瑟，走廊上的灯光昏黄而幽冷，仿佛连空气都凝固了。最西边的病房里，只有一张孤零零的病床，床上躺着一位女子，名叫徐星光。她的生命，如同一盏微弱的灯，仅靠呼吸机的机械呼吸维持着，三年前的那场意外，让她成了植物人，至今未曾醒来。

    # 窗外，风声呜咽，似乎在诉说着某种不为人知的秘密。那扇半开的窗，让寒风肆意侵入，吹乱了徐星光额前的发丝，仿佛在召唤着什么。突然，奇迹般的变化在这寂静的夜晚悄然发生——她紧闭了三年的双眼，竟微微颤动起来。

    # 徐星光费力地睁开眼，模糊的视线还未聚焦，一阵缓慢而沉重的脚步声便传入她的耳中。哒、哒、哒，那声音在走廊里回荡，像是从地狱深处传来的召唤，令人不寒而栗。脚步声在她的病房门口戛然而止，仿佛犹豫了一下，又像是等待着最佳时机。

    # 徐星光的心跳加快了，她的灵魂在十世轮回中积累了对危险的直觉。她感觉到，来者并非善意。于是，她闭上双眼，继续装睡，等待着对方的下一步动作。

    # 吱嘎一声，房门被推开，一个身影悄无声息地走到她的病床旁。那人低头凝视着她，沉默不语，仿佛在权衡着什么。突然，一道低沉而浑厚的男声打破了寂静：“星光。”那声音陌生却又带着一丝熟悉，仿佛从遥远的记忆中传来。

    # 是爸爸！徐星光的心中涌起一股暖流，她几乎要忍不住睁开眼睛，给这位久违的父亲一个惊喜。然而，就在她即将睁开眼的瞬间，徐泽清接下来的话却如一盆冷水，将她心中的喜悦浇灭：“星光啊，你别怪爸爸心狠手辣。你是植物人，这么活着也是受苦，还不如用你的死，给咱们徐家换个荣华富贵...”

    # 徐星光的身体瞬间僵硬，心中的暖流变成了冰冷的寒意。她感觉自己的心跳仿佛停止了，耳边的风声也变得遥远而模糊。原来，她一直以为的亲情，竟是一场精心编织的谎言。她闭着眼，眼角却有泪水无声滑落，融入这冰冷的夜色中。

    # 病房里，只剩下那道低沉的声音在空气中回荡，像是一场无声的审判，宣告着她命运的终结。
    # '''
    #     message = take_prompt(text)
    #     print(message)
    text = '''
A desolate hospital room on the third floor of Changkang Sanatorium, winter night moonlight slicing through half-open curtains. Cardiac monitor's green glow illuminates porcelain-skinned woman (Xu Xingguang) lying motionless with ventilator tubes, black hair swirling in icy drafts from the window. Extreme close-up of twitching eyelids breaking three-year stillness, medical equipment casting grid-like shadows across starched white bedding. Dark silhouette of middle-aged man (Xu Zeqing) holding pillow, face half-obscured in shadows with polished leather shoes gleaming under cold neon lights. Dramatic diagonal composition emphasizing tension between life-support machinery and threatening figure, breath fog visible in frigid air with heart rate monitor beginning erratic spikes. Sinister atmosphere enhanced by fluorescent hallway lights bleeding under door, reflections of bare tree branches clawing at windows like skeletal fingers. ,comic style, very detailed, ultra high resolution, 2K, masterpiece,
'''
    result = text2imageToChat(text)
    print(result)
