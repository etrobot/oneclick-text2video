import requests
from silicon_flow import get_llm_config
import subprocess
import os
import browser_cookie3  # 添加新的导入


class biliScrapper:
    def __init__(self, header_string=None):
        base_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.bilibili.com'
        }

        print("[log] 正在从Chrome获取bilibili.com的cookies...")
        try:
            # 从Chrome获取bilibili.com的cookies
            chrome_cookies = browser_cookie3.chrome(domain_name='.bilibili.com')
            cookie_dict = {cookie.name: cookie.value for cookie in chrome_cookies}
            
            if not cookie_dict:
                print("[warning] 未从Chrome找到bilibili.com的cookies，尝试从文件读取...")
                # 如果Chrome中没有找到cookies，尝试从文件读取
                if os.path.exists('cookies/bilibili.txt'):
                    with open('cookies/bilibili.txt', 'r') as f:
                        header_string = f.read().strip()
                        for item in header_string.split(';'):
                            if '=' in item:
                                key, value = item.strip().split('=', 1)
                                cookie_dict[key] = value
                else:
                    print("[warning] cookies文件不存在，将使用空cookies")
            
            print(f"[log] 成功获取到cookies，包含 {len(cookie_dict)} 个键值对")
            base_headers['Cookie'] = '; '.join([f'{k}={v}' for k, v in cookie_dict.items()])
        except Exception as e:
            print(f"[error] 获取cookies失败: {str(e)}")
            print("[warning] 将使用空cookies继续")
            base_headers['Cookie'] = ''
        
        self.headers = base_headers
        self.headers.update({
            'Origin': 'https://www.bilibili.com',
            'Sec-Fetch-Site': 'same-site',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Dest': 'empty',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9'
        })
        
        # 更新BV转换相关的常量
        self.XOR_CODE = 23442827791579
        self.MASK_CODE = 2251799813685247
        self.MAX_AID = 1 << 51
        self.ALPHABET = "FcwAPNKTMug3GV5Lj7EJnHpWsx4tb8haYeviqBz6rkCy12mUSDQX9RdoZf"
        self.ENCODE_MAP = (8, 7, 0, 5, 1, 3, 2, 4, 6)
        self.DECODE_MAP = tuple(reversed(self.ENCODE_MAP))
        self.BASE = len(self.ALPHABET)
        self.PREFIX = "BV1"
        self.PREFIX_LEN = len(self.PREFIX)
        self.CODE_LEN = len(self.ENCODE_MAP)

    def bilisub2srt(self,j):
        """
        Convert Bilibili subtitle JSON to SRT format.
        """
        def float2hhmmss(num):
            int_ = int(num)
            frac = int((num - int_) * 1000)
            hr, min_, sec = int_ // 3600, int_ % 3600 // 60, int_ % 60
            return f'{hr}:{min_:02d}:{sec:02d}.{frac:03d}'
        
        subs = j['body']
        srts = []
        for i, sub in enumerate(subs, start=1):
            st = float2hhmmss(sub['from'])
            ed = float2hhmmss(sub['to'])
            txt = sub['content']
            srtpt = f'{i}\n{st} --> {ed}\n{txt}'
            srts.append(srtpt)

        srt = '\n\n'.join(srts)
        return srt

    def bv2av(self, bvid: str) -> int:
        """将BV号转换为AV号"""
        print(f"[log] 转换BV号到AV号: {bvid}")
        assert bvid[:3] == self.PREFIX

        bvid = bvid[3:]
        tmp = 0
        for i in range(self.CODE_LEN):
            idx = self.ALPHABET.index(bvid[self.DECODE_MAP[i]])
            tmp = tmp * self.BASE + idx
        avid = (tmp & self.MASK_CODE) ^ self.XOR_CODE
        print(f"[log] 转换结果: AV{avid}")
        return avid

    def av2bv(self, aid: int) -> str:
        """将AV号转换为BV号"""
        print(f"[log] 转换AV号到BV号: {aid}")
        bvid = [""] * 9
        tmp = (self.MAX_AID | aid) ^ self.XOR_CODE
        for i in range(self.CODE_LEN):
            bvid[self.ENCODE_MAP[i]] = self.ALPHABET[tmp % self.BASE]
            tmp //= self.BASE
        result = self.PREFIX + "".join(bvid)
        print(f"[log] 转换结果: {result}")
        return result

    def get_cid_by_bvid(self, bvid):
        """
        使用新版接口通过BV号获取cid.
        """
        url = 'https://api.bilibili.com/x/player/pagelist'
        params = {
            'bvid': bvid,
            'jsonp': 'jsonp'
        }
        # 输出log便于定位问题
        print(f"[log] 获取cid - 请求URL: {url}, 参数: {params}")
        response = requests.get(url, params=params, headers=self.headers)
        print(f"[log] API响应状态码: {response.status_code}")
        print(f"[log] API响应内容: {response.text[:1000]}")  # 打印响应内容（前1000个字符）
        
        json_data = response.json()
        if json_data.get('code') != 0:
            raise Exception(f"获取cid失败: {json_data.get('message', '未知错误')}")
        
        if not json_data.get('data'):
            raise Exception(f"API返回数据异常: {json_data}")
        
        cid = json_data['data'][0]['cid']
        print(f"[log] 获取cid成功: {cid}")
        return cid

    def get_subtitle_from_url(self, video_url):
        """从视频URL获取字幕内容
        Args:
            video_url (str): Bilibili视频链接 (支持BV号和av号格式)
        Returns:
            list: 字幕内容列表，每个元素为字幕对象
        """

        
        try:
            # 从URL提取视频ID
            if 'BV' in video_url:
                bv_id = video_url.split('BV')[1][:12]
                av_id = self.bv2av(f"BV{bv_id}")  # 使用实例方法
            elif 'av' in video_url.lower():
                av_id = int(video_url.lower().split('av')[1].split('/')[0])
            else:
                raise ValueError("Invalid Bilibili video URL")

            # 获取视频cid
            if 'BV' in video_url:
                bvid = "BV" + video_url.split('BV')[1][:12]
            else:
                bvid = self.av2bv(av_id)
            cid = self.get_cid_by_bvid(bvid)

            # 使用新的API URL格式
            api_url = f"https://api.bilibili.com/x/player/wbi/v2?aid={av_id}&cid={cid}"
            response = requests.get(api_url, headers=self.headers)
            
            print(f"API URL: {api_url}")
            print(f"Response status code: {response.status_code}")
            # print(f"Response headers: {response.headers}")
            
            if response.status_code != 200:
                print(f"Error response content: {response.text}")
                raise Exception(f"Failed to fetch video information. Status code: {response.status_code}")

            data = response.json()
            # print(f"API response: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}...")
            
            if data['code'] != 0:
                raise Exception(f"API Error: {data['message']}")

            # 提取字幕列表
            subtitle_list = []
            if 'subtitle' in data['data']:
                subtitles = data['data']['subtitle']['subtitles']
                for sub in subtitles:
                    subtitle_url = f"https:{sub['subtitle_url']}"
                    sub_response = requests.get(subtitle_url, headers=self.headers)
                    if sub_response.status_code == 200:
                        subtitle_list.append({
                            'language': sub['lan_doc'],
                            'content': sub_response.json()['body']
                        })

            return subtitle_list

        except Exception as e:
            print(f"Detailed error: {str(e)}")
            raise

    def get_ass_from_url(self, video_url):
        """从视频URL获取ass字幕内容
        Args:
            video_url (str): Bilibili视频链接 (支持BV号和av号格式)
        Returns:
            list: List of SRT formatted subtitles
        """
        try:
            print("\nTesting subtitle fetch:")
            subs = self.get_subtitle_from_url(video_url)
            if not subs:
                print("No subtitles found for this video")
                return []
            
            # Convert each subtitle to SRT format
            srt_subtitles = []
            for subtitle in subs:
                srt_content = self.bilisub2srt({'body': subtitle['content']})
                srt_subtitles.append({
                    'language': subtitle['language'],
                    'content': srt_content
                })
            return srt_subtitles
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return []

    def get_plain_subtitle_from_url(self, video_url):
        """从视频URL获取纯文本字幕内容（以换行符分割）
        Args:
            video_url (str): Bilibili视频链接 (支持BV号和av号格式)
        Returns:
            str: 纯文本字幕内容
        """
        try:
            subs = self.get_subtitle_from_url(video_url)
            if not subs:
                print("No subtitles found for this video")
                return []
            plain_subtitles = '\n'.join(item['content'] for sub in subs for item in sub['content'])
            return plain_subtitles
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return []

    def download_video(self, video_url: str, output_path: str = "./") -> bool:
        print(f"[log] 开始下载视频: {video_url}")
        print(f"[log] 输出路径: {output_path}")
        
        try:
            # 获取视频cid和aid
            if 'BV' in video_url:
                bv_id = video_url.split('BV')[1][:12]
                av_id = self.bv2av(f"BV{bv_id}")
            elif 'av' in video_url.lower():
                av_id = int(video_url.lower().split('av')[1].split('/')[0])
            else:
                raise ValueError("Invalid Bilibili video URL")
            
            print(f"解析到av号: {av_id}")
            
            # 获取视频cid
            if 'BV' in video_url:
                bvid = "BV" + video_url.split('BV')[1][:12]
            else:
                bvid = self.av2bv(av_id)
            print(f"[log] 下载视频 - 使用BV号获取cid: {bvid}")
            cid = self.get_cid_by_bvid(bvid)
            print(f"[log] 获取到cid: {cid}")

            # 更新获取视频流URL的API参数
            download_api = "https://api.bilibili.com/x/player/wbi/playurl"  # 使用新版wbi接口
            params = {
                "bvid": bvid,  # 使用bvid而不是avid
                "cid": cid,
                "qn": 80,      # 1080P
                "fnval": 4048, # 启用所有格式支持(DASH+杜比+4K)
                "fnver": 0,
                "fourk": 1,
                "platform": "pc",
                "high_quality": 1,
                "otype": "json"
            }
            
            print(f"[log] 请求视频流信息 - URL: {download_api}, 参数: {params}")
            download_response = requests.get(
                download_api, 
                params=params,
                headers=self.headers
            )
            

            download_data = download_response.json()
            if download_data['code'] != 0:
                raise Exception(f"下载API错误: {download_data['message']}")
            
            # 检查是否存在ffmpeg
            def check_ffmpeg():
                try:
                    subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return True
                except:
                    print("[warning] ffmpeg未安装，将使用备选下载方案")
                    return False

            # 获取最佳质量的URL
            def get_best_quality_url(data):
                if 'durl' in data:
                    return data['durl'][0]['url']
                elif 'dash' in data:
                    return data['dash']['video'][0]['baseUrl']
                return None

            output_file = f"{output_path}/video_{bvid}.mp4"
            
            if 'dash' in download_data['data'] and check_ffmpeg():
                # DASH格式下载 (需要ffmpeg)
                video_url = download_data['data']['dash']['video'][0]['baseUrl']
                audio_url = download_data['data']['dash']['audio'][0]['baseUrl']
                
                try:
                    # 下载视频流
                    print(f"[log] 开始下载视频流")
                    video_temp = f"{output_path}/temp_video.m4s"
                    with requests.get(video_url, headers=self.headers, stream=True) as r:
                        r.raise_for_status()
                        with open(video_temp, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    
                    # 下载音频流
                    print(f"[log] 开始下载音频流")
                    audio_temp = f"{output_path}/temp_audio.m4s"
                    with requests.get(audio_url, headers=self.headers, stream=True) as r:
                        r.raise_for_status()
                        with open(audio_temp, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    
                    # 合并视频和音频
                    cmd = f'ffmpeg -i "{video_temp}" -i "{audio_temp}" -c:v copy -c:a copy "{output_file}"'
                    result = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
                    if result.returncode != 0:
                        raise Exception(f"ffmpeg合并失败: {result.stderr.decode()}")
                    
                    # 清理临时文件
                    os.remove(video_temp)
                    os.remove(audio_temp)
                    
                except Exception as e:
                    print(f"[error] DASH格式处理失败: {str(e)}")
                    print("[log] 尝试使用备选下载方案...")
                    # 如果DASH下载失败，回退到直接下载
                    video_url = get_best_quality_url(download_data['data'])
                    if not video_url:
                        raise Exception("无法获取有效的下载地址")
                        
                    with requests.get(video_url, headers=self.headers, stream=True) as r:
                        r.raise_for_status()
                        with open(output_file, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
            else:
                # 直接下载方案
                video_url = get_best_quality_url(download_data['data'])
                if not video_url:
                    raise Exception("无法获取有效的下载地址")
                    
                print(f"[log] 开始直接下载视频")
                with requests.get(video_url, headers=self.headers, stream=True) as r:
                    r.raise_for_status()
                    with open(output_file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            
            print(f"[log] 下载完成: {output_file}")
            return True
            
        except Exception as e:
            print(f"[error] 下载失败: {str(e)}")
            return False

    def search_video(self, keyword: str) -> list:
        """通过关键字搜索视频，返回第一页结果
        
        Args:
            keyword (str): 搜索关键词
            
        Returns:
            list: 搜索结果列表，每个元素包含视频信息
        """
        print(f"[log] 开始搜索视频 - 关键词: {keyword}")
        
        try:
            # 搜索API地址
            url = "https://api.bilibili.com/x/web-interface/wbi/search/all/v2"
            
            # 请求参数
            params = {
                "keyword": keyword,
                "page": 1,
                "page_size": 20,
                "search_type": "video"  # 指定搜索类型为视频
            }
            
            print(f"[log] 发送搜索请求 - URL: {url}, 参数: {params}")
            
            # 发送请求
            response = requests.get(url, params=params, headers=self.headers)
            

            data = response.json()

            
            # 检查响应状态
            if data['code'] != 0:
                if data['code'] == -412:
                    print("[error] 搜索请求被拦截，请确保cookies中包含buvid3字段")
                raise Exception(f"API错误: {data['message']}")
            
            # 提取视频结果
            results = []
            if 'data' in data and 'result' in data['data']:
                items = data['data']['result']
                print(f"[log] 找到结果数量: {len(items)}")
                
                for item in items:
                    if item['result_type'] == 'video' and item['data']:
                        for video in item['data']:
                            video_info = {
                                'bvid': video.get('bvid', ''),
                                'aid': video.get('aid', 0),
                                'title': video.get('title', ''),
                                'author': video.get('author', ''),
                                'description': video.get('description', ''),
                                'play_count': video.get('play', 0),
                                'duration': video.get('duration', ''),
                                'url': video.get('arcurl', '')
                            }
                            results.append(video_info)
                            print(f"[log] 处理视频: {video_info['title']}")
            else:
                print("[log] 未找到有效的搜索结果")
                print(f"[log] 响应data字段的键: {list(data.get('data', {}).keys())}")
            
            print(f"[log] 搜索成功 - 找到 {len(results)} 个视频结果")
            return results
            
        except Exception as e:
            print(f"[error] 搜索失败: {str(e)}")
            return []

if __name__ == '__main__':
    biliScrapper = biliScrapper()
    # success = downloader.download_video(video_url, output_path="./downloads")
    # if success:
    #     print("视频下载成功！")
    # else:
    #     print("视频下载失败")
    results = biliScrapper.search_video("manus")
    for video in results:
        print(f"标题: {video['title']}")
        print(f"作者: {video['author']}")
        print(f"播放量: {video['play_count']}")
        print(f"链接: {video['url']}")
        # biliScrapper.download_video(video['url'], output_path="./downloads")
        biliScrapper.get_plain_subtitle_from_url(video['url'])
        biliScrapper.get_ass_from_url(video['url'])
        break