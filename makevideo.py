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
from silicon_flow import SiliconFlow,get_llm_config
from utils import sqlite_log, DebugLogger, log_method, llm_gen_json, get_next_available_id, text2voice, text2video

load_dotenv(find_dotenv())

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
            video = video.subclipped(0, remaining_duration)
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
