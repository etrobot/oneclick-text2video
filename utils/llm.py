"""
LLM 相关工具函数
"""

import json
import time,os
import openai
from debug_logger import DebugLogger

def llm_gen_json(llm: openai.Client, model: str, query: str, format: dict, debug=False, max_retries: int = 20) -> dict:
    """
    使用 LLM 生成指定格式的 JSON 响应
    
    Args:
        llm: OpenAI 客户端实例
        model: 模型名称
        query: 查询内容
        format: 期望的 JSON 格式
        debug: 是否开启调试日志
        max_retries: 最大重试次数
        
    Returns:
        dict: 生成的 JSON 数据，失败则返回 None
    """
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

def get_llm_config(scheme='openai'):
    apikey=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("OPENAI_BASE_URL")
    if scheme == 'siliconflow':
        apikey=os.getenv("SILICONFLOW_API_KEY")
        base_url=os.getenv("SILICONFLOW_BASE_URL")
    return apikey,base_url

def get_llm_client(scheme='openai'):
    apikey,base_url=get_llm_config(scheme)
    return openai.Client(api_key=apikey, base_url=base_url)

