import os
import time
import json
import logging
import datetime
import sqlite3
import pandas as pd
import openai
import azure.cognitiveservices.speech as speechsdk
import glob
import re

# sqlite_log 函数

def sqlite_log(task_id, step_description, result_type, result):
    logging.info(f"记录日志: task_id={task_id}, step={step_description}, result_type={result_type}, result={result}")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'task_id': task_id,
        'step': step_description,
        'result_type': result_type,
        'result': result
    }
    try:
        df = pd.DataFrame([log_entry])
        with sqlite3.connect('task_logs.db') as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    timestamp TEXT,
                    task_id TEXT,
                    step TEXT,
                    result_type TEXT,
                    result TEXT
                )
            ''')
            df.to_sql('logs', conn, if_exists='append', index=False)
            conn.commit()
    except Exception as e:
        print(f"写入日志失败: {e}")


# DebugLogger 类

class DebugLogger:
    def __init__(self, debug=False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)

    def log(self, level, message):
        if self.debug:
            if level == 'error':
                self.logger.error(message)
            elif level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            else:
                self.logger.debug(message)

    def log_entry(self, func_name, args=None, kwargs=None):
        if self.debug:
            args_str = f", args: {args}" if args else ""
            kwargs_str = f", kwargs: {kwargs}" if kwargs else ""
            self.logger.info(f"Entering {func_name}{args_str}{kwargs_str}")

    def log_exit(self, func_name, result=None):
        if self.debug:
            result_str = f", returned: {result}" if result is not None else ""
            self.logger.info(f"Exiting {func_name}{result_str}")


# log_method 装饰器

def log_method(debug=True):
    logger = DebugLogger(debug=debug)
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log_entry(func.__name__, args, kwargs)
            try:
                result = func(*args, **kwargs)
                logger.log_exit(func.__name__, result)
                return result
            except Exception as e:
                logger.log('error', f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator


# llm_gen_json 函数

def llm_gen_json(llm: openai.Client, model: str, query: str, format: dict, debug=False, max_retries: int = 20) -> dict:
    logger = DebugLogger(debug=True)  # 强制开启调试日志
    logger.log('info', f"开始生成JSON，查询内容: {query}")
    prompt = f"\noutput in json format :\n{str(format)}\n"
    retry = max_retries
    while retry > 0:
        try:
            logger.log('info', f"尝试请求 LLM (剩余重试次数: {retry})")
            llm_response = llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query + prompt}],
                response_format={ "type": "json_object" }
            )
            result = json.loads(llm_response.choices[0].message.content)
            logger.log('info', f"收到响应: {result}")

            if not isinstance(result, dict):
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    result = result[0]
                    logger.log('info', "将列表结果转换为字典")
                else:
                    logger.log('error', f"无效的响应格式，将重试\n{result}\n")
                    retry -= 1
                    continue
            if not all(k in result for k in format):
                logger.log('error', f"响应缺少必要字段，将重试\n{result}\n")
                retry -= 1
                continue
            logger.log('info', f"成功生成JSON结果: {result}")
            return result
        except Exception as e:
            logger.log('error', f"发生错误: {str(e)}")
            time.sleep(2)
            retry -= 1
            continue
    logger.log('error', "达到最大重试次数，返回 None")
    return None


# get_next_available_id 辅助函数

def get_next_available_id():
    audio_files = glob.glob('output/audio/id*.mp3')
    if not audio_files:
        return 1
    ids = [int(re.search(r'id(\d+)\.mp3', f).group(1)) for f in audio_files]
    return max(ids) + 1


# text2voice 函数

@log_method()

def text2voice(text: str, auto_increment=True, id=None, voice_name='zh-CN-guangxi-YunqiNeural'):
    logger = logging.getLogger(__name__)
    logger.info(f"开始文本转语音，文本内容: {text}")

    azure_key = os.getenv('AZURE_TTS')
    if not azure_key:
        logger.error("未找到 AZURE_TTS 环境变量")
        raise ValueError("AZURE_TTS environment variable is not set")
    logger.info("成功获取 AZURE_TTS key")

    os.makedirs('output/audio', exist_ok=True)
    if auto_increment:
        file_id = get_next_available_id()
    elif id is not None:
        file_id = id
    else:
        raise ValueError("必须指定id或启用auto_increment")

    base_filename = f'id{file_id:02d}'
    filename = os.path.join('output/audio', base_filename + '.mp3')
    logger.info(f"输出文件路径: {filename}")

    try:
        logger.info("开始配置 Speech 服务...")
        speech_config = speechsdk.SpeechConfig(subscription=azure_key, region="eastasia")
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3)
        logger.info(f"语音配置: 区域=eastasia, 声音={voice_name}")
        speech_config.speech_synthesis_voice_name = voice_name

        audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)
        logger.info(f"音频输出配置已创建，输出文件: {filename}")

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        logger.info("语音合成器已创建")

        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        ssml = f"""<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" xml:lang=\"zh-CN\">\n            <voice name=\"{voice_name}\">{text}</voice>\n        </speak>"""
        logger.info(f"生成的SSML: {ssml}")

        word_boundaries = []
        def handle_word_boundary(evt):
            word_boundaries.append({
                'text': evt.text,
                'audio_offset': evt.audio_offset / 10000000,
                'duration': evt.duration.total_seconds(),
            })
            logger.debug(f"Word boundary: {evt.text}, offset: {evt.audio_offset}, duration: {evt.duration}")

        def handle_cancellation(evt):
            logger.error(f"语音合成被取消: {evt.result.error_details}")

        speech_synthesizer.synthesis_word_boundary.connect(handle_word_boundary)
        speech_synthesizer.synthesis_canceled.connect(handle_cancellation)

        logger.info("开始执行语音合成...")
        result = speech_synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"语音合成成功完成")
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                logger.info(f"音频文件成功生成，大小: {os.path.getsize(filename)} bytes")
            else:
                logger.error(f"音频文件生成失败或大小为0: {filename}")
                raise Exception("Audio file generation failed")
        else:
            logger.error(f"语音合成失败: {result.reason}")
            if hasattr(result, 'error_details'):
                logger.error(f"错误详情: {result.error_details}")
            raise Exception(f"Speech synthesis failed: {result.reason}")

        return filename, word_boundaries, base_filename
    except Exception as e:
        logger.error(f"语音合成过程中发生错误: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        raise

@log_method()
def prepare_assets_with_videos(asset):
    logger = logging.getLogger(__name__)
    # 打印入参信息
    logger.info(f"入参 asset: {json.dumps({k: str(v) if not isinstance(v, (list, dict)) else v for k, v in asset.items()}, ensure_ascii=False, indent=2)}")
    
    audio = AudioFileClip(asset['audio_file'])
    audio_duration = audio.duration
    asset['duration'] = audio_duration
    asset['videos'] = []
    
    text = asset['text']
    keywords = extract_keywords(text)
    
    sf = SiliconFlow()
    os.makedirs('videos', exist_ok=True)
    
    total_video_duration = 0
    keyword_index = 0
    max_attempts = 10  # 防止无限循环
    attempts = 0

    while total_video_duration < audio_duration and attempts < max_attempts:
        keyword = keywords[keyword_index % len(keywords)]
        keyword_index += 1
        logger.info(f"当前处理: 关键词={keyword}, 已生成视频时长={total_video_duration:.2f}秒, 目标音频时长={audio_duration:.2f}秒, 尝试次数={attempts + 1}")
            
        image_url = sf.generate_image(keyword)
        if not image_url:
            attempts += 1
            continue

        video_url = sf.generate_video(keyword, image_url)
        if not video_url:
            attempts += 1
            continue

        try:
            response = requests.get(video_url)
            temp_video_path = f'videos/temp_{int(time.time())}.mp4'
            with open(temp_video_path, 'wb') as f:
                f.write(response.content)
            
            video_clip = VideoFileClip(temp_video_path)
            video_duration = video_clip.duration
            video_clip.close()
            
            total_video_duration += video_duration
            asset['videos'].append(temp_video_path)
            logger.info(f"成功添加视频: 路径={temp_video_path}, 时长={video_duration:.2f}秒, 累计视频时长={total_video_duration:.2f}秒")
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            attempts += 1
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            continue

    # 在返回之前检查是否成功生成了视频
    if not asset['videos']:
        raise Exception("未能成功生成任何视频素材")
    
    audio.close()
    
    # 打印出参信息
    logger.info(f"出参 asset: {json.dumps({k: str(v) if not isinstance(v, (list, dict)) else v for k, v in asset.items()}, ensure_ascii=False, indent=2)}")
    return asset


@log_method(debug=False)
def generate_thumbnail(video_path, thumbnail_path=None):
    """
    从视频文件生成缩略图
    Args:
        video_path: 视频文件路径
        thumbnail_path: 指定的缩略图保存路径
    Returns:
        str: 缩略图文件路径，如果生成失败则返回None
    """
    try:
        if thumbnail_path is None:
            os.makedirs('output/thumbnails', exist_ok=True)
            thumbnail_path = os.path.join('output/thumbnails', 
                                        os.path.splitext(os.path.basename(video_path))[0] + '.jpg')
        
        video = VideoFileClip(video_path)
        frame = video.get_frame(0)
        img = Image.fromarray(frame)
        img.save(thumbnail_path, 'JPEG', quality=85)
        video.close()
        return thumbnail_path
    except Exception as e:
        logger.error(f"生成缩略图时发生错误: {str(e)}")
        return None
 