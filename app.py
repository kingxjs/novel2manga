from flask import Flask, Response, request, render_template, jsonify, send_from_directory, stream_with_context

from text_to_video import convertTextToVideo
import uuid
import asyncio
from configs import models
from api.llm_bk import reinvent_prompt

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/reinvent', methods=['POST'])
def reinvent():
    data = request.json
    novel = data.get('text', '')

   # 创建一个同步生成器来包装异步函数
    def generate():
        # 创建一个新的事件循环来运行异步代码
        loop = asyncio.new_event_loop()
        try:
            # 获取异步生成器
            async_gen = reinvent_prompt(novel)
            
            # 运行直到第一个结果
            while True:
                try:
                    # 在事件循环中运行异步代码，获取下一个结果
                    chunk = loop.run_until_complete(anext_async(async_gen))
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
    
    # 使用同步生成器创建流式响应
    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream'
    )

# 辅助函数，用于获取异步生成器的下一个值
async def anext_async(agen):
    return await agen.__anext__()

@app.route('/convert', methods=['POST'])
def convert_text_to_video():
    # 获取请求中的文本数据
    text = request.form.get('text')
    model = request.form.get('model')
    # 在此处调用您的文本转视频的代码，并生成视频文件
    # videos/1686290882-anything-v4.0.mp4
    
    chapter_title = str(uuid.uuid4())
    video_path = convertTextToVideo(validate_model(model), text,chapter_title)
    return jsonify({'video_path': video_path})


@app.route('/models', methods=['GET'])
def get_available_models():
    return jsonify(models)


@app.route('/videos/<path:filename>')
def get_video(filename):
    return send_from_directory('./', filename)


def validate_model(model):
    if model in models:
        return model
    else:
        return models[0]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
