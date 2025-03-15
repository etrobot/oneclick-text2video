import os
from silicon_flow import get_llm_config
from utils import llm_gen_json
import webvtt
from yt_dlp import YoutubeDL
import browser_cookie3
import tempfile

class ytbScrapper:
    def __init__(self):
        """初始化YouTube下载器"""
        print("[log] 正在从Chrome获取youtube.com的cookies...")
        try:
            # 从Chrome获取youtube.com的cookies并转换为Netscape格式
            chrome_cookies = browser_cookie3.chrome(domain_name='.youtube.com')
            
            # 创建临时cookie文件
            self.cookie_file = 'cookies/youtube.txt'  # 更新cookie文件路径
            with open(self.cookie_file, 'w') as f:
                f.write("# Netscape HTTP Cookie File\n")
                
                for cookie in chrome_cookies:
                    # 转换为Netscape格式
                    secure = "TRUE" if cookie.secure else "FALSE"
                    http_only = "TRUE" if cookie.has_nonstandard_attr('HttpOnly') else "FALSE"
                    f.write(
                        f".youtube.com\tTRUE\t/\t{secure}\t{int(cookie.expires) if cookie.expires else 0}\t{cookie.name}\t{cookie.value}\n"
                    )
            
            print(f"[log] 成功创建cookies文件: {self.cookie_file}")
            
        except Exception as e:
            print(f"[error] 获取cookies失败: {str(e)}")
            print("[warning] 将使用空cookies继续")
            self.cookie_file = None

    def __del__(self):
        """清理临时cookie文件"""
        # 删除清理方法
        pass

    def search_video(self, keyword: str, max_results: int = 5) -> list:
        """通过关键字搜索视频，返回结果列表
        
        Args:
            keyword (str): 搜索关键词
            max_results (int): 最大返回结果数
            
        Returns:
            list: 搜索结果列表，每个元素包含视频信息
        """
        print(f"[log] 开始搜索视频 - 关键词: {keyword}")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',
            'force_generic_extractor': True
        }
        
        if self.cookie_file:
            ydl_opts['cookiefile'] = self.cookie_file

        videos = []
        with YoutubeDL(ydl_opts) as ydl:
            try:
                search_url = f"ytsearch{max_results*2}:{keyword}"
                search_results = ydl.extract_info(search_url, download=False)
                
                if not search_results.get('entries'):
                    print("[log] 未找到相关视频")
                    return videos

                for entry in search_results['entries']:
                    if entry.get('url'):
                        url = entry.get('url')
                        duration = entry.get('duration', 0)
                        
                        if '/shorts/' in url or duration < 180:  # 跳过短视频
                            continue
                            
                        video_info = {
                            'title': entry.get('title', ''),
                            'description': entry.get('description', ''),
                            'id': entry.get('id', ''),
                            'url': f"https://youtube.com/watch?v={entry.get('id')}",
                            'duration': duration,
                            'view_count': entry.get('view_count', 0),
                            'author': entry.get('uploader', '')
                        }
                        videos.append(video_info)
                        print(f"[log] 找到视频: {video_info['title']}")
                        
                        if len(videos) >= max_results:
                            break
                
                return videos

            except Exception as e:
                print(f"[error] 搜索失败: {str(e)}")
                return videos

    def download_video(self, video_url: str, output_path: str = "./") -> bool:
        """下载指定URL的视频
        
        Args:
            video_url (str): 视频URL
            output_path (str): 输出路径
            
        Returns:
            bool: 下载是否成功
        """
        print(f"[log] 开始下载视频: {video_url}")
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{output_path}/%(title)s.%(ext)s'
        }
        
        if self.cookie_file:
            ydl_opts['cookiefile'] = self.cookie_file

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            print(f"[log] 视频下载成功")
            return True
        except Exception as e:
            print(f"[error] 下载失败: {str(e)}")
            return False

    def get_plain_subtitle_from_url(self, video_url: str) -> str:
        """获取视频的纯文本字幕
        
        Args:
            video_url (str): 视频URL
            
        Returns:
            str: 字幕文本内容
        """
        print(f"[log] 开始获取字幕: {video_url}")
        
        ydl_opts = {
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'outtmpl': 'temp_%(id)s',
        }
        
        if self.cookie_file:
            ydl_opts['cookiefile'] = self.cookie_file

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                video_id = info['id']
                
                # 读取生成的VTT文件
                vtt_file = f"temp_{video_id}.en.vtt"
                if os.path.exists(vtt_file):
                    vtt = webvtt.read(vtt_file)
                    subtitle_text = '\n'.join(caption.text for caption in vtt.captions)
                    
                    # 清理临时文件
                    os.remove(vtt_file)
                    
                    return subtitle_text
                else:
                    print("[warning] 未找到字幕文件")
                    return ""
                    
        except Exception as e:
            print(f"[error] 获取字幕失败: {str(e)}")
            return ""

if __name__ == '__main__':
    scrapper = ytbScrapper()
    # 测试搜索
    results = scrapper.search_video("python tutorial")
    for video in results:
        print(f"标题: {video['title']}")
        print(f"作者: {video['author']}")
        print(f"时长: {video['duration']}秒")
        print(f"链接: {video['url']}")
        
        # 测试下载和字幕
        # scrapper.download_video(video['url'], output_path="./downloads")
        # subtitle = scrapper.get_plain_subtitle_from_url(video['url'])
        # print(f"字幕预览: {subtitle[:200]}...")
        break