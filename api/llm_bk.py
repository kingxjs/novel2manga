import os

from dotenv import load_dotenv
from openai import OpenAI  # 修改导入
import json

load_dotenv('.env', override=True)

# 尝试加载本地开发环境变量文件
load_dotenv('.local.env', override=True)

# 读取环境变量
api_key = os.getenv('OPEN_API_KEY')  # 变量名可以保持不变或改为更通用的名称
base_url = os.getenv('OPEN_AI_BASE_URL')  # 如果有自定义API端点需要添加这个变量
client = OpenAI(
    api_key=api_key,  # 使用api_key而非api_token参数
    base_url=base_url,  # 如果需要自定义API端点
)


# 默认模型名称修改为OpenAI的模型
async def reinvent_prompt(text, model: str = "deepseek-ai/DeepSeek-V2.5"):

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


def generate_prompt(text, model: str = "deepseek-ai/DeepSeek-V2.5", retry: int = 0) -> str:  # 默认模型名称修改为OpenAI的模型
    MIDJOURNEY_PROMPT = '''Hello, you are an expert in generating midjourney prompt words, users give you a description, you generate the corresponding prompt words!

    Keep in mind that the prompt words you are generating are in English and may be in a format similar to:

    As the sun gently kissed the horizon, casting a warm golden glow across the bustling city streets, a captivating sight unfolded before my eyes. A graceful woman, adorned in an elegant dress that fluttered in the gentle breeze, held a porcelain cup of steaming coffee in her delicate hands.

   Instead of interpreting any input from the user, without any other nonsense.
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


def take_prompt(text, model: str = "deepseek-ai/DeepSeek-V2.5", retry: int = 0):  # 默认模型名称修改为OpenAI的模型
    MIDJOURNEY_PROMPT = '''You're a professional visual novel script writer who helps me break down multiple scenes based on a novel or script I've provided.Please follow these requirements closely when creating and returning data in JSON format:
    # The output is formatted so that each paragraph contains the following
    [
    '背景：
    时间：
    氛围：
    描述：
    视觉元素：
    人物（标注性别）：',
    ''
    ]
    
    
    Keep the background uniform in similar scenes.Up to 6 scenes.
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
                            "items": {"type": "string"},
                            "description": "必须包含：背景,时间,氛围,描述,视觉元素,人物（需标注性别）"
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
            return function_response["results"]
        else:
            # 如果没有返回工具调用，则返回一个默认的空数组
            return []
    except Exception as e:
        if retry > 2:
            print(f"Error in generate_prompt: {e}")
            return []
        return take_prompt(text, model, retry+1)


if __name__ == '__main__':
    text = '''
长康疗养院的三楼，是一片沉寂的植物人病房区。夜色如墨，寒风萧瑟，走廊上的灯光昏黄而幽冷，仿佛连空气都凝固了。最西边的病房里，只有一张孤零零的病床，床上躺着一位女子，名叫徐星光。她的生命，如同一盏微弱的灯，仅靠呼吸机的机械呼吸维持着，三年前的那场意外，让她成了植物人，至今未曾醒来。

窗外，风声呜咽，似乎在诉说着某种不为人知的秘密。那扇半开的窗，让寒风肆意侵入，吹乱了徐星光额前的发丝，仿佛在召唤着什么。突然，奇迹般的变化在这寂静的夜晚悄然发生——她紧闭了三年的双眼，竟微微颤动起来。

徐星光费力地睁开眼，模糊的视线还未聚焦，一阵缓慢而沉重的脚步声便传入她的耳中。哒、哒、哒，那声音在走廊里回荡，像是从地狱深处传来的召唤，令人不寒而栗。脚步声在她的病房门口戛然而止，仿佛犹豫了一下，又像是等待着最佳时机。

徐星光的心跳加快了，她的灵魂在十世轮回中积累了对危险的直觉。她感觉到，来者并非善意。于是，她闭上双眼，继续装睡，等待着对方的下一步动作。

吱嘎一声，房门被推开，一个身影悄无声息地走到她的病床旁。那人低头凝视着她，沉默不语，仿佛在权衡着什么。突然，一道低沉而浑厚的男声打破了寂静：“星光。”那声音陌生却又带着一丝熟悉，仿佛从遥远的记忆中传来。

是爸爸！徐星光的心中涌起一股暖流，她几乎要忍不住睁开眼睛，给这位久违的父亲一个惊喜。然而，就在她即将睁开眼的瞬间，徐泽清接下来的话却如一盆冷水，将她心中的喜悦浇灭：“星光啊，你别怪爸爸心狠手辣。你是植物人，这么活着也是受苦，还不如用你的死，给咱们徐家换个荣华富贵...”

徐星光的身体瞬间僵硬，心中的暖流变成了冰冷的寒意。她感觉自己的心跳仿佛停止了，耳边的风声也变得遥远而模糊。原来，她一直以为的亲情，竟是一场精心编织的谎言。她闭着眼，眼角却有泪水无声滑落，融入这冰冷的夜色中。

病房里，只剩下那道低沉的声音在空气中回荡，像是一场无声的审判，宣告着她命运的终结。
'''
    message = take_prompt(text)
    print(message)
