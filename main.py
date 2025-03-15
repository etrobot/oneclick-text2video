from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
from pathlib import Path
from makevideo import text2video, merge_videos
from dotenv import load_dotenv,find_dotenv
import datetime
import sqlite3
import pandas as pd
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

load_dotenv(find_dotenv())


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_editor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 在挂载静态文件前创建'static'目录
os.makedirs("static", exist_ok=True)
logger.info("静态文件夹 'static' 已创建或已存在")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 确保必要的目录存在
os.makedirs("output/video", exist_ok=True)
os.makedirs("output/thumbnails", exist_ok=True)

# 创建线程池执行器
executor = ThreadPoolExecutor()

# 添加全局变量和线程锁用于限制单个视频生成任务
video_generation_lock = threading.Lock()
video_generation_in_progress = False

def scan_video_files():
    video_dir = Path("output/video")
    video_files = sorted([str(f) for f in video_dir.glob("*.mp4")])
    return video_files

def get_task_ids() -> List[str]:
    """获取所有任务ID（去重）"""
    logger.info("开始获取所有任务ID")
    try:
        with sqlite3.connect('task_logs.db') as conn:
            df = pd.read_sql_query("SELECT DISTINCT task_id FROM logs", conn)
            task_ids = df['task_id'].tolist()
            logger.info(f"成功获取任务ID列表: {task_ids}")
            return task_ids
    except Exception as e:
        logger.error(f"获取任务ID时出错: {str(e)}")
        return []

def get_task_videos(task_id: str) -> Dict:
    """获取指定任务ID的所有视频文件信息"""
    logger.info(f"开始获取任务ID {task_id} 的视频信息")
    try:
        with sqlite3.connect('task_logs.db') as conn:
            df = pd.read_sql_query(
                "SELECT * FROM logs WHERE task_id = ? AND result_type = 'video'", 
                conn, 
                params=(task_id,)
            )
            videos = []
            for _, row in df.iterrows():
                video_path = row['result']
                if os.path.exists(video_path):
                    thumbnail_path = video_path.replace('.mp4', '.jpg').replace('/video/', '/thumbnails/')
                    videos.append({
                        'video_url': f"/video/{os.path.basename(video_path)}",
                        'thumbnail_url': f"/thumbnail/{os.path.basename(video_path)}",
                        'timestamp': row['timestamp']
                    })
            logger.info(f"成功获取视频信息: {videos}")
            return {'videos': videos}
    except Exception as e:
        logger.error(f"获取任务视频时出错: {str(e)}")
        return {'videos': []}

def delete_task(task_id: str) -> bool:
    """删除指定任务ID的所有记录和相关文件"""
    logger.info(f"开始删除任务ID {task_id} 的所有记录")
    try:
        # 1. 获取所有相关文件路径
        with sqlite3.connect('task_logs.db') as conn:
            df = pd.read_sql_query(
                "SELECT result FROM logs WHERE task_id = ? AND result_type IN ('video', 'audio')", 
                conn, 
                params=(task_id,)
            )
            
        # 2. 删除物理文件
        for file_path in df['result']:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"删除文件: {file_path}")
                # 同时删除对应的缩略图
                if file_path.endswith('.mp4'):
                    thumbnail_path = file_path.replace('.mp4', '.jpg').replace('/video/', '/thumbnails/')
                    if os.path.exists(thumbnail_path):
                        os.remove(thumbnail_path)
                        logger.info(f"删除缩略图: {thumbnail_path}")
            except Exception as e:
                logger.error(f"删除文件时出错: {str(e)}")
                
        # 3. 删除数据库记录
        with sqlite3.connect('task_logs.db') as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM logs WHERE task_id = ?", (task_id,))
            conn.commit()
            logger.info(f"删除数据库记录成功")
            
        return True
    except Exception as e:
        logger.error(f"删除任务时出错: {str(e)}")
        return False

@app.get("/")
async def home(request: Request):
    task_ids = get_task_ids()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "task_ids": task_ids}
    )

def generate_video_sync(script: str, task_id: str):
    """同步执行视频生成的函数"""
    global video_generation_in_progress
    try:
        logger.info(f"[generate_video_sync] 开始生成视频 task_id: {task_id}")
        assets = text2video(script, task_id=task_id)
        video_files = [asset['video_file'] for asset in assets.values()]
        logger.info(f"[generate_video_sync] 视频生成完成 task_id: {task_id}, files: {video_files}")
        return video_files
    except Exception as e:
        logger.error(f"[generate_video_sync] 生成视频出错 task_id: {task_id}, error: {str(e)}")
        raise e
    finally:
        with video_generation_lock:
            video_generation_in_progress = False

@app.post("/generate-video")
async def generate_video(script: str = Form(...)):
    global video_generation_in_progress
    # 检查是否已有视频生成任务在运行
    with video_generation_lock:
        if video_generation_in_progress:
            return JSONResponse(
                status_code=400,
                content={"error": "当前已有视频生成任务正在运行，请稍后再试"}
            )
        video_generation_in_progress = True

    try:
        logger.info(f"[generate_video] 入参 script: {script}")
        if not script.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "请输入文本"}
            )
        
        # 自动生成任务ID
        task_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
        logger.info(f"[generate_video] 自动生成任务ID: {task_id}")
        
        # 使用线程池异步执行视频生成，不阻塞前端日志轮询
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, generate_video_sync, script, task_id)
        
        return JSONResponse(content={
            "message": "视频生成任务已开始",
            "task_id": task_id
        })
    except Exception as e:
        logger.error(f"[generate_video] 创建视频生成任务时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"创建视频生成任务时出错: {str(e)}"}
        )

@app.get("/video/{video_name}")
async def get_video(video_name: str):
    video_path = Path(f"output/video/{video_name}")
    if not video_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "视频不存在"}
        )
    return FileResponse(str(video_path))

@app.get("/thumbnail/{video_name}")
async def get_thumbnail(video_name: str):
    thumbnail_name = video_name.replace('.mp4', '.jpg')
    thumbnail_path = Path(f"output/thumbnails/{thumbnail_name}")
    
    if not thumbnail_path.exists():
        from makevideo import generate_thumbnail
        video_path = Path(f"output/video/{video_name}")
        if video_path.exists():
            thumbnail_path = generate_thumbnail(str(video_path))
    
    if not thumbnail_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "缩略图不存在"}
        )
    
    return FileResponse(str(thumbnail_path))

@app.post("/merge-videos")
async def merge_videos_endpoint():
    try:
        video_files = scan_video_files()
        if not video_files:
            return JSONResponse(
                status_code=400,
                content={"error": "没有可用的视频文件"}
            )
            
        output_path = "output/video/final_video.mp4"
        merge_videos(video_files, output_path)
        
        return JSONResponse(content={
            "message": "视频合并成功",
            "video_path": output_path
        })
    except Exception as e:
        logger.error(f"合并视频时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"合并视频时出错: {str(e)}"}
        )

@app.get("/api/tasks/{task_id}/videos")
async def get_task_videos_endpoint(task_id: str):
    return get_task_videos(task_id)

@app.delete("/api/tasks/{task_id}")
async def delete_task_endpoint(task_id: str):
    success = delete_task(task_id)
    if success:
        return JSONResponse(content={"message": "任务删除成功"})
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "删除任务时出错"}
        )

@app.get("/api/tasks/{task_id}/logs")
async def get_task_logs(task_id: str):
    """获取指定任务ID的所有日志"""
    logger.info(f"开始获取任务ID {task_id} 的日志")
    try:
        with sqlite3.connect('task_logs.db') as conn:
            df = pd.read_sql_query(
                "SELECT timestamp, step, result_type, result FROM logs WHERE task_id = ? ORDER BY timestamp DESC",
                conn,
                params=(task_id,)
            )
            logs = df.to_dict('records')
            return {'logs': logs}
    except Exception as e:
        logger.error(f"获取任务日志时出错: {str(e)}")
        return {'logs': []}

@app.get("/audio/{audio_name}")
async def get_audio(audio_name: str):
    audio_path = Path(f"output/audio/{audio_name}")
    if not audio_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "音频文件不存在"}
        )
    return FileResponse(str(audio_path))

@app.route('/api/tasks/logs')
def get_task_logs():
    task_id = request.args.get('value')  # 从select的value获取task_id
    if not task_id:
        return '<div class="text-gray-500 text-center py-8">请选择一个任务ID查看日志。</div>'
    
    logs = get_logs_for_task(task_id)  # 获取日志的函数
    return render_template('_logs.html', logs=logs)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 