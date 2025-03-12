from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
from pathlib import Path
from makevideo import text2video, merge_videos
import shutil

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

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 确保必要的目录存在
os.makedirs("output/video", exist_ok=True)
os.makedirs("output/thumbnails", exist_ok=True)

def scan_video_files():
    video_dir = Path("output/video")
    video_files = sorted([str(f) for f in video_dir.glob("*.mp4")])
    return video_files

@app.get("/")
async def home(request: Request):
    video_files = scan_video_files()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "video_files": video_files}
    )

@app.post("/generate-video")
async def generate_video(script: str = Form(...)):
    try:
        if not script.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "请输入文本"}
            )
            
        # 生成视频
        assets = text2video(script)
        
        # 获取生成的视频文件列表
        video_files = [asset['video_file'] for asset in assets.values()]
        
        return JSONResponse(content={
            "message": "视频生成成功",
            "video_files": video_files
        })
    except Exception as e:
        logger.error(f"生成视频时出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"生成视频时出错: {str(e)}"}
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 