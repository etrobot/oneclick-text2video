import requests
import random,time
import os,json
import logging
from openai import OpenAI


def get_llm_config(scheme='openai'):
    apikey=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("OPENAI_BASE_URL")
    if scheme == 'siliconflow':
        apikey=os.getenv("SILICONFLOW_API_KEY")
        base_url=os.getenv("SILICONFLOW_BASE_URL")
    return apikey,base_url

class SiliconFlow:
    def __init__(self):
        api_key, base_url = get_llm_config('siliconflow')
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _generate_seed(self):
        return random.randint(10000000, 99999999)
    
    def describe_image(self, image_url):
        image_prompt = 'make a stable diffusion prompt with this image, only output the prompt \n'
        response = self.client.chat.completions.create(
            model="OpenGVLab/InternVL2-26B",
            messages=[{"role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": image_prompt}]
            }],
            stream=False
        )
        return response.choices[0].message.content

    def generate_image(self, prompt, seed=None, style='realistic'):
        logger = logging.getLogger(__name__)
        logger.info(f"Generating image for prompt: {prompt}")
        
        realistic_prmt=f'{prompt}, Masterful photography, crisp details, soft bokeh, shot by Annie Leibovitz, cinematic lighting, sharp focus'
        scifi_prmt=f'Epic sci-fi movie scene, {prompt}, Blade Runner aesthetic, Denis Villeneuve cinematography, cyberpunk cityscape, neon lights, high contrast, dramatic lighting, hyper-realistic details, 8K resolution'
        prmt_dict={
            'realistic':realistic_prmt,
            'scifi':scifi_prmt
        }
        
        payload = {
            "model": "stabilityai/stable-diffusion-3-5-large",
            "prompt": prmt_dict[style],
            "image_size": "1024x576",
            "batch_size": 1,
            "seed": seed if seed else self._generate_seed(),
            "num_inference_steps": 25,
            "guidance_scale": 5,
            "prompt_enhancement": False
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        sd_api_url = "https://api.siliconflow.cn/v1/images/generations"
        generation_response = requests.request("POST", sd_api_url, json=payload, headers=headers)
        response_data = json.loads(generation_response.text)
        logger.info(f"Image generation response: {response_data}")
        
        if "images" in response_data and len(response_data["images"]) > 0:
            image_url = response_data["images"][0]["url"]
            logger.info(f"Generated image URL: {image_url}")
            return image_url
        logger.error("Failed to generate image")
        return None

    def generate_video(self, prompt, image_url, seed=None):
        logger = logging.getLogger(__name__)
        logger.info(f"Generating video for prompt: {prompt}")
        
        payload = {
            "model": "Lightricks/LTX-Video",
            "prompt": prompt,
            "image": image_url,
            "seed": seed if seed else self._generate_seed()
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        video_api_url = "https://api.siliconflow.cn/v1/video/submit"
        response = requests.request("POST", video_api_url, json=payload, headers=headers)
        response_data = json.loads(response.text)
        
        if "requestId" in response_data:
            request_id = response_data["requestId"]
            time.sleep(30)
            while True:
                status_response = self.check_video_status(request_id)
                logger.info(f"Video status check response: {status_response}")
                
                if status_response["status"] == "Succeed":
                    video_url = status_response["results"]["videos"][0]["url"]
                    logger.info(f"Video generation succeeded: {video_url}")
                    return video_url
                elif status_response["status"] in ["Failed", "Cancelled"]:
                    logger.error(f"Video generation failed: {status_response.get('reason')}")
                    return None
                time.sleep(10)
        
        logger.error("Failed to get request ID for video generation")
        return None

    def check_video_status(self, request_id):
        status_url = "https://api.siliconflow.cn/v1/video/status"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"requestId": request_id}
        response = requests.request("POST", status_url, json=payload, headers=headers)
        return json.loads(response.text)