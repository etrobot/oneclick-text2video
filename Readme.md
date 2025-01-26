
# OneClick Text2Video

OneClick Text2Video 是一个将文本脚本转换为视频的应用程序。它使用 OpenAI 和 Azure 的服务来生成音频和视频内容。

## 安装

1. 克隆此存储库到你的本地机器。
2. 确保你已经安装了 Python 及其依赖项。你可以使用以下命令安装依赖项：

   ```bash
   poetry install
   ```

3. 配置 `.env` 文件以设置必要的 API 密钥。

## 配置

在项目根目录下创建一个 `.env` 文件，并添加以下内容：

```plaintext:/Users/tfproduct01/Documents/DEV/AI/oneclick-text2video/.env
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_API_KEY=你的SiliconFlow API密钥
LLM_MODEL='deepseek-ai/DeepSeek-V2.5'
AZURE_TTS=你的Azure TTS订阅密钥
```

请确保将上述占位符替换为你的实际 API 密钥。

## 使用

1. 运行应用程序：

   ```bash
   streamlit run app.py
   ```

2. 在浏览器中打开 Streamlit 应用程序，输入你的文本脚本，然后点击“生成视频”按钮。

3. 生成的视频将保存在 `output/video` 文件夹中。

## 功能

- 将文本转换为音频。
- 根据文本生成视频片段。
- 合并多个视频片段。
- 生成视频缩略图。

## 贡献

欢迎贡献！请提交问题或请求合并。

## 许可证

此项目使用 MIT 许可证。