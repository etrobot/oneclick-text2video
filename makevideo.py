import os
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip, TextClip
import openai
from dotenv import load_dotenv,find_dotenv
import logging
from PIL import Image
from utils import sqlite_log, log_method, text2voice, prepare_assets_with_videos, generate_thumbnail

load_dotenv(find_dotenv())
# text2video 函数

@log_method()
def text2video(videoscript: str, task_id: str = None):
    logger = logging.getLogger(__name__)
    logger.info(f"开始处理视频脚本: {videoscript}")
    if task_id:
        logger.info(f"任务ID: {task_id}")

    os.makedirs('output', exist_ok=True)
    os.makedirs('output/video', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)
    os.makedirs('output/thumbnails', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    
    script_lines = videoscript.split('\n')
    assets = {}
    id_counter = 1

    for line in script_lines:
        if not line.strip():
            continue
        logger.info(f"处理第 {id_counter} 行文本: {line}")
        if task_id:
            sqlite_log(task_id, f"处理第{id_counter}行文本", "text", line)
        audio_file, word_boundaries, base_filename = text2voice(line, auto_increment=True)
        if task_id:
            sqlite_log(task_id, f"第{id_counter}行音频生成", "audio", audio_file)
        video_path = os.path.join('output/video', f"{base_filename}.mp4")
        assets[id_counter] = {
            'audio_file': audio_file,
            'text': line,
            'word_boundaries': word_boundaries,
            'video_file': video_path,
            'base_filename': base_filename
        }
        logger.info("开始准备视频素材")
        assets[id_counter] = prepare_assets_with_videos(assets[id_counter])
        if task_id:
            sqlite_log(task_id, f"第{id_counter}行视频素材准备完成", "info", str(len(assets[id_counter]['videos'])) + "个视频片段")
        logger.info(f"开始创建最终视频: {video_path}")
        assets[id_counter] = create_video_from_assets(assets[id_counter], video_path)
        if task_id:
            sqlite_log(task_id, f"第{id_counter}行视频生成完成", "video", video_path)
        logger.info("生成视频缩略图")
        thumbnail = generate_thumbnail(video_path)
        assets[id_counter]['thumbnail'] = thumbnail
        if task_id and thumbnail:
            sqlite_log(task_id, f"第{id_counter}行缩略图生成", "thumbnail", thumbnail)
        logger.info(f"完成第 {id_counter} 行文本的处理")
        id_counter += 1
    if task_id:
        sqlite_log(task_id, "任务完成", "info", f"共处理{id_counter-1}个片段")
    return assets


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
