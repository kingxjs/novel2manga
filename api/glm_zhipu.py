import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv('.env', override=True)

# 尝试加载本地开发环境变量文件
load_dotenv('.local.env', override=True)

# 读取环境变量
api_token = os.getenv('GLM_API_KEY')


# 封装上述代码成工具函数
def generate_prompt_zhipu(text, model: str = "glm-4-flash") -> str:
    client = ZhipuAI(api_key=api_token)  # 请填写您自己的APIKey
    # midjourney_prompt = f"""
    # I will give you a paragraph of text. The text is: {text}.
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Use no more than 10 phrases to describe the passage, separated by commas
    # An example is as follows: girl, elf,  (1ong green gradient hair:1.3), green brooch, frilly shirt)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    # """

    midjourney_prompt = '''Hello, you are an expert in generating midjourney prompt words, users give you a description, you generate the corresponding prompt words!

     Keep in mind that the prompt words you are generating are in English and may be in a format similar to:

     As the sun gently kissed the horizon, casting a warm golden glow across the bustling city streets, a captivating sight unfolded before my eyes. A graceful woman, adorned in an elegant dress that fluttered in the gentle breeze, held a porcelain cup of steaming coffee in her delicate hands.

    Instead of interpreting any input from the user, without any other nonsense.
     '''
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": midjourney_prompt},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    text = "“Have you heard? Cao Xiong laoshi had just competed with the academy’s number one teacher from the " \
           "bottom, Teacher Zhang Xuan!”“They competed? Then, wouldn’t Zhang Xuan laoshi have lost for sure?” "
    message = generate_prompt_zhipu(text)
    print(message)

