# 小说转视频简单版

本工具将用户上传的文本文件转换为视频并保存到本地，旨在实现小说的可视化阅读功能

效果图如下：

<img src="demo/demo.png" alt="效果图" style="width: 100%;" />

## 方法设计

- 将文本进行分段，通过标点符号句号分段，分成一个个的句子。
- 选取多个句子为一个分镜，控制150字~200字左右为一个分镜，通过句子生成图片，生成声音，图片，本方案采用 liblib文生图，语言转文字使用 有道tts
- **通过大模型生成midjourney类的提示词，然后通过huggingface的模型/文生图api生成图片**。
- 在通过 moviepy 将图片合并为视频，目前输出 mp4 格式的视频，句子作为字幕贴到视频内容的底部区域。


## Docker 一键启动

```shell
docker-compose up --build
```


pip install -r requirements.txt

## 支持生成绘图提示词来提高绘图质量

需要配置 智谱API 的 api key，支持代理

```shell
GLM_API_KEY="your open ai api key"
```

## 生成 相关的key

智谱API 申请地址：https://bigmodel.cn/usercenter/apikeys

有道语音合成 申请地址：https://ai.youdao.com/doc.s#guide

LibLib 申请地址：https://www.liblib.art/apis

密钥 写入到 .env 文件里面
`GLM_API_KEY="your  api key"`


## 开始使用

```python
#网页访问：
python app.py
#http://127.0.0.1:5001/

# main方法启动：
python text_to_video.py
```

## License: MIT

本项目采用 MIT 许可证授权。