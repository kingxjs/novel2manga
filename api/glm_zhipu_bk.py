import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv('../.env', override=True)

# 尝试加载本地开发环境变量文件
load_dotenv('.local.env', override=True)

# 读取环境变量
api_token = os.getenv('GLM_API_KEY')


# 封装上述代码成工具函数
def generate_slogan(text, model: str = "glm-4-flash") -> str:
    client = ZhipuAI(api_key=api_token)  # 请填写您自己的APIKey
    # content = f"""
    # I will give you a paragraph of text. The text is: {text}.
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Divide this text into no more than 15 phrases for text-to-image prompts, divided into two parts: detailed description + generation standards.
    # An example is as follows:(1 girl, elf,  (1ong green gradient hair:1.3), green brooch, frilly shirt, very detailed, ultra high resolution, 32K UHD, best quality, masterpiece,)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Details description requirements are as follows:
    # Include: Features of the main character (clothing, facial features, facial expressions, etc.), characteristics of the scene (indoor or outdoor settings, internal details, overall scene, etc.), scene setting (day or night, weather conditions, direction of light, etc.), adjectives (beautiful, happy, etc.)
    #
    # Generation standards requirements are as follows:
    # Include: Image quality prompts (High quality, Clearness, Wallpaper, etc.), style prompts (comic).
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    # It is also important to note the usage of prompt word weights:
    # 1. Parentheses represent a weight of 1.1 times, for example, Exquisite Crown adding (Exquisite Crown) means that the weight of the word "crown" becomes 1.1 times.
    # 2. [pink|blond] long hair, using square brackets to separate the colors, during rendering, it alternates between pink and gold, resulting in an adjusted pink-gold color. The square brackets serve to blend the colors.
    # """

    MIDJOURNEY_PROMPT = '''Hello, you are an expert in generating midjourney prompt words, users give you a description, you generate the corresponding prompt words!

    Keep in mind that the prompt words you are generating are in English and may be in a format similar to:

    As the sun gently kissed the horizon, casting a warm golden glow across the bustling city streets, a captivating sight unfolded before my eyes. A graceful woman, adorned in an elegant dress that fluttered in the gentle breeze, held a porcelain cup of steaming coffee in her delicate hands.

   Instead of interpreting any input from the user, without any other nonsense.
    '''
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": MIDJOURNEY_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    text = "“Have you heard? Cao Xiong laoshi had just competed with the academy’s number one teacher from the " \
           "bottom, Teacher Zhang Xuan!”“They competed? Then, wouldn’t Zhang Xuan laoshi have lost for sure?” "
    message = generate_slogan(text)
    print(message)

