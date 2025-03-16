"""
文本转语音功能
"""

import os
import logging
import glob
import re
import azure.cognitiveservices.speech as speechsdk
from ..logging.debug_logger import log_method

def get_next_available_id():
    """获取下一个可用的音频文件ID
    
    Returns:
        int: 下一个可用的ID
    """
    audio_files = glob.glob('output/audio/id*.mp3')
    if not audio_files:
        return 1
    ids = [int(re.search(r'id(\d+)\.mp3', f).group(1)) for f in audio_files]
    return max(ids) + 1

@log_method()
def text2voice(text: str, auto_increment=True, id=None, voice_name='zh-CN-guangxi-YunqiNeural'):
    """将文本转换为语音
    
    Args:
        text: 要转换的文本
        auto_increment: 是否自动递增ID
        id: 指定的ID
        voice_name: Azure TTS 声音名称
        
    Returns:
        tuple: (音频文件路径, 单词边界信息, 基础文件名)
    """
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