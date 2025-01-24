import requests
import os,time
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

load_dotenv(find_dotenv())

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
    logger=DebugLogger(debug=debug)
    prompt= f"\noutput in json format :\n{str(format)}\n"
    retry=max_retries
    while retry>0:
        try:
            llm_response = llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query+prompt}],
                response_format={ "type": "json_object" }
            )
            result=json.loads(llm_response.choices[0].message.content)
            if not isinstance(result, dict):
                if isinstance(result, list) and len(result)>0 and isinstance(result[0], dict):
                    result = result[0]
                else:
                    logger.log('error', f"Invalid action received, will retry\n{result}\n")
                    continue
                if not all(k in result for k in format):
                    logger.log('error', f"Invalid action received, will retry\n{result}\n")
                    continue
            return result
        except Exception as e:
            logger.log('error', e)
            time.sleep((max_retries-retry)*10)
            retry-=1
            continue
    return None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_editor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@log_method(debug=False)  # 将 debug 设置为 False
def text2voice(text:str, audioFile='result',voice_name='zh-CN-guangxi-YunqiNeural'):
    # 创建音频输出目录
    os.makedirs('output/audio', exist_ok=True)
    filename = os.path.join('output/audio', audioFile + '.mp3')
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_TTS'), region="eastasia")
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)

    speech_config.speech_synthesis_voice_name = voice_name

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    ssml = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="zh-CN">
        <voice name="{voice_name}">
         <lexicon uri="https://raw.githubusercontent.com/etrobot/azure-tts-lexicon-cn/refs/heads/main/lexicon.xml"/>
            <mstts:express-as style="cheerful">
                <prosody rate="+20%" volume="-1%">
                    {text}
                </prosody>
            </mstts:express-as>
        </voice>
    </speak>
    """
    word_boundaries = []
    def handle_word_boundary(evt):
        # 移除调试打印
        word_boundaries.append({
            'text': evt.text,
            'audio_offset': evt.audio_offset / 10000000,
            'duration': evt.duration.total_seconds(),
        })
    
    speech_synthesizer.synthesis_word_boundary.connect(handle_word_boundary)

    result = speech_synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format(text))
        stream = speechsdk.AudioDataStream(result)
        stream.save_to_wav_file(filename)
    
    return filename, word_boundaries


@log_method()
def prepare_assets_with_videos(asset):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting to prepare assets with videos for text: {asset['text']}")
    audio = AudioFileClip(asset['audio_file'])
    audio_duration = audio.duration
    asset['duration'] = audio_duration
    asset['videos'] = []
    
    text = asset['text']
    keywords = extract_keywords(text)
    
    sf = SiliconFlow()
    os.makedirs('videos', exist_ok=True)
    
    total_video_duration = 0
    
    for keyword in keywords:
        logger.info(f"Processing keyword: {keyword}")
        if total_video_duration >= audio_duration:
            logger.info("Total video duration reached audio duration, stopping")
            break
            
        image_url = sf.generate_image(keyword)
        if image_url:
            video_url = sf.generate_video(keyword, image_url)
            if video_url:
                try:
                    # 下载视频到本地临时文件
                    response = requests.get(video_url)
                    temp_video_path = f'videos/temp_{int(time.time())}.mp4'
                    with open(temp_video_path, 'wb') as f:
                        f.write(response.content)
                    
                    video_clip = VideoFileClip(temp_video_path)
                    video_duration = video_clip.duration
                    video_clip.close()
                    
                    total_video_duration += video_duration
                    asset['videos'].append(video_url)
                    
                    # 清理临时文件
                    os.remove(temp_video_path)
                except Exception as e:
                    logger.error(f"Error processing video: {str(e)}")
                    continue
    
    # 在返回之前检查是否成功生成了视频
    if not asset['videos']:
        raise Exception("未能成功生成任何视频素材")
    
    audio.close()
    return asset

@log_method(debug=False)  # 将 debug 设置为 False
def extract_keywords(text):
    api_key, base_url = get_llm_config('siliconflow')
    keywords_format = {
    "keywords": ["keyword1", "keyword2", "keyword3"]
}
    client = openai.Client(api_key=api_key, base_url=base_url)
    # text = client.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content":text+'for this video script, generate as many fun gif names as possible, output in English'}]
    # ).choices[0].message.content
    prompt= f'''
    generate short stable diffusion prompts for the following video script:
    text: {text}
    '''
    while True:
        keywords = llm_gen_json(client,os.getenv('LLM_MODEL'),prompt,keywords_format)['keywords']
        if all(keyword.isascii() for keyword in keywords):
            break
    # 移除 print(keywords)
    return keywords

@log_method()
def create_video_from_assets(asset, output_path, vertical=False):
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
    
    # 根据标点符号分割文本
    sentences = []
    current_sentence = []
    current_start = word_boundaries[0]['audio_offset']
    
    for i, word_info in enumerate(word_boundaries):
        word = word_info['text']
        if word not in ['.', '!', '?', ',','，','？','。']:  # 如果不是标点，添加到当前句子
            current_sentence.append(word)
            
        # 检查原始文本中这个词后面是否跟着标点符号
        word_end_pos = text.find(word) + len(word)
        next_char = text[word_end_pos:word_end_pos+1] if word_end_pos < len(text) else ''
        
        # 如果下一个字符是标点或者是最后一个词，结束当前句子
        if (next_char in ['.', '!', '?', ',','，','？','。'] or 
            i == len(word_boundaries) - 1):
            sentences.append({
                'text': ' '.join(current_sentence),
                'start': current_start,
                'end': word_info['audio_offset'] + word_info['duration']
            })
            if i < len(word_boundaries) - 1:
                current_start = word_boundaries[i + 1]['audio_offset']
            current_sentence = []
    
    # 为每个字创建带动画效果的TextClip
    for word_info in word_boundaries:
        word = word_info['text']
        start_time = word_info['audio_offset']
        duration = word_info['duration']
        
        # 创建逐字字幕
        text_clip = (TextClip(text=word, 
                            font='Hiragino Sans GB',
                            font_size=45,
                            size=(text_width, None),
                            color='white',
                            stroke_color='black',
                            stroke_width=2,
                            method='caption',
                            text_align='center')
                    .with_position(('center', video_height * 0.85))
                    .with_start(start_time)
                    .with_duration(duration))
        
        subtitle_clips.append(text_clip)

    video_clips = []
    current_time = 0
    
    # 创建一个固定的黑色背景
    background = ColorClip(size=(video_width, video_height), color=(0,0,0))
    
    for video_path in asset['videos']:
        video = VideoFileClip(video_path)
        
        # 修改视频缩放逻辑，确保视频填满画面
        # 计算缩放比例，使视频的短边至少等于目标尺寸的对应边
        width_ratio = video_width / video.w
        height_ratio = video_height / video.h
        scale_ratio = max(width_ratio, height_ratio)
        
        new_width = int(video.w * scale_ratio)
        new_height = int(video.h * scale_ratio)
        
        video = video.resized(width=new_width, height=new_height)
        
        # 计算居中位置，确保视频居中显示
        x_center = (video_width - new_width) // 2
        y_center = (video_height - new_height) // 2
        
        # 如果当前视频素材时长超出剩余音频时长，则裁剪
        remaining_duration = audio_duration - current_time
        if video.duration > remaining_duration:
            video = video.subclipped(0, remaining_duration)
        
        # 添加视频转场和动画效果
        # video = (video
        #         .with_effects([vfx.Resize(lambda t: 1 + 0.05 * np.sin(t * np.pi))])
        #         .with_position((x_center, y_center)).with_start(current_time))

        # 设置视频素材的开始时间和位置
        current_time += video.duration
        video_clips.append(video)
    
    # 设置背景持续时间与音频相同
    background = background.with_duration(audio_duration)
    
    # 将背景放在第一位，然后是所有的视频素材片段
    all_clips = [background] + video_clips + subtitle_clips
    
    # 合并所有视频片段
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

@log_method(debug=False)
def text2voice(text:str, auto_increment=True, id=None, voice_name='zh-CN-guangxi-YunqiNeural'):
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
    
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_TTS'), region="eastasia")
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)

    speech_config.speech_synthesis_voice_name = voice_name

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    ssml = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="zh-CN">
        <voice name="{voice_name}">
         <lexicon uri="https://raw.githubusercontent.com/etrobot/azure-tts-lexicon-cn/refs/heads/main/lexicon.xml"/>
            <mstts:express-as style="cheerful">
                <prosody rate="+20%" volume="-1%">
                    {text}
                </prosody>
            </mstts:express-as>
        </voice>
    </speak>
    """
    word_boundaries = []
    def handle_word_boundary(evt):
        # 移除调试打印
        word_boundaries.append({
            'text': evt.text,
            'audio_offset': evt.audio_offset / 10000000,
            'duration': evt.duration.total_seconds(),
        })
    
    speech_synthesizer.synthesis_word_boundary.connect(handle_word_boundary)

    result = speech_synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format(text))
        stream = speechsdk.AudioDataStream(result)
        stream.save_to_wav_file(filename)
    
    return filename, word_boundaries, base_filename

def text2video(videoscript:str):
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
            
        # Unpack three values from text2voice
        audio_file, word_boundaries, base_filename = text2voice(line, auto_increment=True)
        video_path = os.path.join('output/video', f"{base_filename}.mp4")
        
        assets[id] = {
            'audio_file': audio_file,
            'text': line,
            'word_boundaries': word_boundaries,
            'video_file': video_path,
            'base_filename': base_filename
        }
        
        assets[id] = prepare_assets_with_videos(assets[id])
        assets[id] = create_video_from_assets(assets[id], video_path)
        
        thumbnail = generate_thumbnail(video_path)
        assets[id]['thumbnail'] = thumbnail
        
        id += 1
    
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

def text2video(videoscript:str):
    # 创建视频输出目录
    os.makedirs('output/video', exist_ok=True)
    
    script_lines = videoscript.split('\n')
    assets = {}
    id=1
    for line in script_lines:
        # Skip empty lines or lines with only whitespace
        if not line.strip():
            continue
        audio_file, word_boundaries = text2voice(line,f"output_audio_{id}.mp3")
        output_path = os.path.join('output/video', f"output_video_{id}.mp4")
        
        assets[id] = {
            'audio_file': audio_file,
            'text': line,
            'word_boundaries': word_boundaries,
            'video_file': output_path
        }
        
        assets[id] = prepare_assets_with_videos(assets[id])
        assets[id] = create_video_from_assets(assets[id], output_path)
        
        thumbnail = generate_thumbnail(output_path)
        assets[id]['thumbnail'] = thumbnail
        
        id+=1
    
    return assets

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
