"""
视频处理相关功能
"""

import os
import logging
import time
import requests
import json
from moviepy import VideoFileClip, AudioFileClip
from silicon_flow import SiliconFlow
from PIL import Image
from .debug_logger import log_method
from .llm import get_llm_config, llm_gen_json

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
    """准备视频资产
    
    Args:
        asset: 包含音频文件和文本信息的资产字典
        
    Returns:
        dict: 更新后的资产字典
    """
    logger = logging.getLogger(__name__)
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

    if not asset['videos']:
        raise Exception("未能成功生成任何视频素材")
    
    audio.close()
    
    logger.info(f"出参 asset: {json.dumps({k: str(v) if not isinstance(v, (list, dict)) else v for k, v in asset.items()}, ensure_ascii=False, indent=2)}")
    return asset

@log_method(debug=False)
def generate_thumbnail(video_path, thumbnail_path=None):
    """从视频文件生成缩略图
    
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