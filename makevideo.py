import requests
import os
import azure.cognitiveservices.speech as speechsdk
import numpy as np
from moviepy import vfx
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip, TextClip
from urllib.parse import quote
import random
import openai
import time
from openai import OpenAI
from dotenv import load_dotenv,find_dotenv
import json
import logging
from PIL import Image
import io
from silicon_flow import SiliconFlow,get_llm_config
import glob
import re
import pandas as pd
import sqlite3
import datetime

load_dotenv(find_dotenv())


def sqlite_log(task_id, step_description, result_type, result):
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 构造日志字典
    log_entry = {
        'timestamp': timestamp,
        'task_id': task_id,
        'step': step_description,
        'result_type': result_type,
        'result': result
    }
    
    try:
        # 将日志字典转换为DataFrame并存入sqlite数据库
        df = pd.DataFrame([log_entry])
        with sqlite3.connect('task_logs.db') as conn:
            # 确保表存在
            conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    timestamp TEXT,
                    task_id TEXT,
                    step TEXT,
                    result_type TEXT,
                    result TEXT
                )
            ''')
            # 使用 to_sql 写入数据
            df.to_sql('logs', conn, if_exists='append', index=False)
            conn.commit()
    except Exception as e:
        print(f"写入日志失败: {e}")

# Add DebugLogger class before llm_gen_json function
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

def llm_gen_json(llm:openai.Client,model:str,query:str,format:dict,debug=False,max_retries:int=20)->dict:
    logger=DebugLogger(debug=True)  # 强制开启调试日志
    logger.log('info', f"开始生成JSON，查询内容: {query}")
    prompt= f"\noutput in json format :\n{str(format)}\n"
    retry=max_retries
    while retry>0:
        try:
            logger.log('info', f"尝试请求 LLM (剩余重试次数: {retry})")
            llm_response = llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query+prompt}],
                response_format={ "type": "json_object" }
            )
            result=json.loads(llm_response.choices[0].message.content)
            logger.log('info', f"收到响应: {result}")
            
            if not isinstance(result, dict):
                if isinstance(result, list) and len(result)>0 and isinstance(result[0], dict):
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
            time.sleep(2)  # 固定等待时间，避免过长延迟
            retry-=1
            continue
    logger.log('error', "达到最大重试次数，返回 None")
    return None

@log_method(debug=True)  # 开启调试日志
def extract_keywords(text):
    logger = logging.getLogger(__name__)
    logger.info(f"开始提取关键词，文本内容: {text}")
    
    api_key, base_url = get_llm_config('siliconflow')
    keywords_format = {
        "keywords": ["keyword1", "keyword2", "keyword3"]
    }
    client = openai.Client(api_key=api_key, base_url=base_url)
    prompt = f'''
    generate 3-5 short stable diffusion prompts in English for the following video script with the structure as location+role+action+object like "in the office, a ol is taking to the screen, the video script is":
    text: {text}
    '''
    
    logger.info("开始调用 LLM 生成关键词")
    keywords = llm_gen_json(client, os.getenv('LLM_MODEL'), prompt, keywords_format)
    
    if not keywords:
        logger.warning("生成关键词失败，使用默认关键词")
        return ["happy celebration", "festive atmosphere", "red lantern"]
        
    result = keywords.get('keywords', [])
    logger.info(f"生成的关键词: {result}")
    
    # 过滤非ASCII字符的关键词
    filtered_keywords = [k for k in result if k.isascii()]
    logger.info(f"过滤后的关键词: {filtered_keywords}")
    
    if not filtered_keywords:
        logger.warning("过滤后没有有效关键词，使用默认关键词")
        return ["happy celebration", "festive atmosphere", "red lantern"]
        
    return filtered_keywords[:3]  # 限制返回3个关键词

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


@log_method()
def create_video_from_assets(asset, output_path, vertical=False):
    logger = logging.getLogger(__name__)
    logger.info(f"开始处理视频资源，视频数量: {len(asset['videos'])}")
    
    """
    将音频文件和对应的视频序列合成为视频，支持横屏(720p)和竖屏(1080x1920)
    """
    # 设置视频尺寸
    video_width = 1080 if vertical else 1280
    video_height = 1920 if vertical else 720
    
    # 计算字幕宽度（确保是整数）
    text_width = int(video_width * 0.9)
    
    audio = AudioFileClip(asset['audio_file'])
    audio_duration = asset['duration']
    
    # 创建字幕clips列表
    subtitle_clips = []
    text = asset['text']
    word_boundaries = asset['word_boundaries']
    
    # 根据标点符号分割文本为句子
    sentences = []
    current_sentence = []
    current_start = word_boundaries[0]['audio_offset']
    
    for i, word_info in enumerate(word_boundaries):
        word = word_info['text']
        if word not in ['.', '!', '?', ',','，','？','。']:
            current_sentence.append(word)
        # 检查是否需要结束当前句子
        word_end_pos = text.find(word) + len(word)
        next_char = text[word_end_pos:word_end_pos+1] if word_end_pos < len(text) else ''
        if (next_char in ['.', '!', '?', ',','，','？','。'] or i == len(word_boundaries) - 1):
            if current_sentence:
                sentence_text = " ".join(current_sentence)
                sentences.append({
                    'text': sentence_text,
                    'start': current_start,
                    'end': word_info['audio_offset'] + word_info['duration']
                })
            if i < len(word_boundaries) - 1:
                current_start = word_boundaries[i + 1]['audio_offset']
            current_sentence = []
    
    # 为每个句子创建字幕 (去除高亮逻辑，统一使用白色)
    for sentence in sentences:
        subtitle_duration = sentence['end'] - sentence['start']
        text_clip = (TextClip(text=sentence['text'], 
                          font='Hiragino Sans GB',
                          font_size=45,
                          size=(text_width, None),
                          color='white',
                          stroke_color='black',
                          stroke_width=2,
                          method='caption')
                   .with_position(('center', video_height * 0.85))
                   .with_start(sentence['start'])
                   .with_duration(subtitle_duration))
        subtitle_clips.append(text_clip)
    
    video_clips = []
    current_time = 0
    
    # 创建一个固定的黑色背景
    background = ColorClip(size=(video_width, video_height), color=(0,0,0))
    
    # 处理所有视频片段
    for video_path in asset['videos']:
        logger.info(f"处理视频片段: {video_path}, 当前时间点: {current_time:.2f}秒")
        video = VideoFileClip(video_path)
        
        # 修改视频缩放逻辑，确保视频填满画面
        width_ratio = video_width / video.w
        height_ratio = video_height / video.h
        scale_ratio = max(width_ratio, height_ratio)
        
        new_width = int(video.w * scale_ratio)
        new_height = int(video.h * scale_ratio)
        
        video = video.resized(width=new_width, height=new_height)
        
        # 计算居中位置
        x_center = (video_width - new_width) // 2
        y_center = (video_height - new_height) // 2
        
        # 如果当前视频素材时长超出剩余音频时长，则裁剪
        remaining_duration = audio_duration - current_time
        if video.duration > remaining_duration:
            video = video.subclip(0, remaining_duration)
            logger.info(f"裁剪视频到 {remaining_duration:.2f}秒")
        
        # 设置视频开始时间和位置
        video = video.with_position((x_center, y_center)).with_start(current_time)
        logger.info(f"设置视频位置: ({x_center}, {y_center}), 开始时间: {current_time:.2f}秒, 时长: {video.duration:.2f}秒")
        
        video_clips.append(video)
        current_time += video.duration
    
    # 设置背景持续时间与音频相同
    background = background.with_duration(audio_duration)
    
    # 将背景放在第一位，然后是所有的视频素材片段和字幕
    all_clips = [background] + video_clips + subtitle_clips
    
    # 合并所有视频片段
    logger.info(f"开始合并视频片段，总片段数: {len(all_clips)}")
    final_video = CompositeVideoClip(all_clips, size=(video_width, video_height))
    
    # 添加音频
    final_video = final_video.with_audio(audio)
    
    # 导出最终视频
    final_video.write_videofile(
        output_path,
        fps=24,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=False
    )
    
    # 清理资源
    audio.close()
    final_video.close()
    for clip in video_clips:
        clip.close()

    asset['video_file'] = output_path
    return asset


@log_method(debug=False)
def get_next_available_id():
    """获取下一个可用的ID"""
    audio_files = glob.glob('output/audio/id*.mp3')
    if not audio_files:
        return 1
    # 从文件名中提取ID号并找到最大值
    ids = [int(re.search(r'id(\d+)\.mp3', f).group(1)) for f in audio_files]
    return max(ids) + 1

@log_method(debug=True)
def text2voice(text:str, auto_increment=True, id=None, voice_name='zh-CN-guangxi-YunqiNeural'):
    logger = logging.getLogger(__name__)
    logger.info(f"开始文本转语音，文本内容: {text}")
    
    # 验证环境变量
    azure_key = os.getenv('AZURE_TTS')
    if not azure_key:
        logger.error("未找到 AZURE_TTS 环境变量")
        raise ValueError("AZURE_TTS environment variable is not set")
    logger.info("成功获取 AZURE_TTS key")
    
    # 确保输出目录存在
    os.makedirs('output/audio', exist_ok=True)
    
    if auto_increment:
        file_id = get_next_available_id()
    elif id is not None:
        file_id = id
    else:
        raise ValueError("必须指定id或启用auto_increment")
        
    # 统一文件命名格式
    base_filename = f'id{file_id:02d}'
    filename = os.path.join('output/audio', base_filename + '.mp3')
    logger.info(f"输出文件路径: {filename}")
    
    try:
        # 创建 SpeechConfig 时添加更多日志
        logger.info("开始配置 Speech 服务...")
        speech_config = speechsdk.SpeechConfig(subscription=azure_key, region="eastasia")
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3)
        
        # 验证语音配置
        logger.info(f"语音配置: 区域=eastasia, 声音={voice_name}")
        speech_config.speech_synthesis_voice_name = voice_name
        
        # 创建音频配置
        audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)
        logger.info(f"音频输出配置已创建，输出文件: {filename}")
        
        # 创建合成器
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        logger.info("语音合成器已创建")
        
        # 清理SSML中的特殊字符
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # 简化SSML结构
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
            <voice name="{voice_name}">{text}</voice>
        </speak>"""
        logger.info(f"生成的SSML: {ssml}")
        
        word_boundaries = []
        def handle_word_boundary(evt):
            word_boundaries.append({
                'text': evt.text,
                'audio_offset': evt.audio_offset / 10000000,
                'duration': evt.duration.total_seconds(),
            })
            logger.debug(f"Word boundary: {evt.text}, offset: {evt.audio_offset}, duration: {evt.duration}")
        
        # 添加取消事件处理
        def handle_cancellation(evt):
            logger.error(f"语音合成被取消: {evt.result.error_details}")
            
        speech_synthesizer.synthesis_word_boundary.connect(handle_word_boundary)
        speech_synthesizer.synthesis_canceled.connect(handle_cancellation)
        
        # 执行语音合成
        logger.info("开始执行语音合成...")
        result = speech_synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"语音合成成功完成")
            # 验证文件是否成功生成
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
            os.remove(filename)  # 清理可能的空文件
        raise

@log_method()
def text2video(videoscript:str, task_id:str=None):
    """
    将文本脚本转换为视频
    Args:
        videoscript: 包含多行文本的字符串，每行将被转换为一个视频片段
        task_id: 可选的任务ID，用于日志记录
    Returns:
        dict: 包含所有生成资产信息的字典
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始处理视频脚本: {videoscript}")
    if task_id:
        logger.info(f"任务ID: {task_id}")
    
    # 创建所有必要的输出目录
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/video', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)
    os.makedirs('output/thumbnails', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    
    script_lines = videoscript.split('\n')
    assets = {}
    id = 1
    for line in script_lines:
        if not line.strip():
            continue
            
        logger.info(f"处理第 {id} 行文本: {line}")
        
        # 记录任务进度
        if task_id:
            sqlite_log(task_id, f"处理第{id}行文本", "text", line)
        
        # 使用 text2voice 生成音频和字幕信息
        audio_file, word_boundaries, base_filename = text2voice(line, auto_increment=True)
        if task_id:
            sqlite_log(task_id, f"第{id}行音频生成", "audio", audio_file)
            
        video_path = os.path.join('output/video', f"{base_filename}.mp4")
        
        assets[id] = {
            'audio_file': audio_file,
            'text': line,
            'word_boundaries': word_boundaries,
            'video_file': video_path,
            'base_filename': base_filename
        }
        
        # 准备视频素材
        logger.info(f"开始准备视频素材")
        assets[id] = prepare_assets_with_videos(assets[id])
        if task_id:
            sqlite_log(task_id, f"第{id}行视频素材准备完成", "info", str(len(assets[id]['videos'])) + "个视频片段")
        
        # 创建最终视频
        logger.info(f"开始创建最终视频: {video_path}")
        assets[id] = create_video_from_assets(assets[id], video_path)
        if task_id:
            sqlite_log(task_id, f"第{id}行视频生成完成", "video", video_path)
        
        # 生成缩略图
        logger.info(f"生成视频缩略图")
        thumbnail = generate_thumbnail(video_path)
        assets[id]['thumbnail'] = thumbnail
        if task_id and thumbnail:
            sqlite_log(task_id, f"第{id}行缩略图生成", "thumbnail", thumbnail)
        
        logger.info(f"完成第 {id} 行文本的处理")
        id += 1
    
    if task_id:
        sqlite_log(task_id, "任务完成", "info", f"共处理{id-1}个片段")
    
    return assets

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

@log_method()
def merge_videos(video_files, output_path):
    """
    合并多个视频文件为一个视频
    """
    clips = []
    try:
        for file in video_files:
            clip = VideoFileClip(file)
            clips.append(clip)
        
        final_video = concatenate_videoclips(clips)
        final_video.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
    finally:
        # 清理资源
        for clip in clips:
            clip.close()
        if 'final_video' in locals():
            final_video.close()
