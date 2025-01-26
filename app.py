import streamlit as st
import os
import time
from makevideo import text2video, merge_videos
from moviepy import VideoFileClip
from PIL import Image
import numpy as np
import logging

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

def scan_video_files():
    video_dir = os.path.join('output', 'video')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                   if f.endswith('.mp4')]
    return sorted(video_files)

# Initialize session state
if 'video_files' not in st.session_state:
    st.session_state.video_files = scan_video_files()
if 'current_video' not in st.session_state:
    st.session_state.current_video = (st.session_state.video_files[0] 
                                    if st.session_state.video_files else None)

def create_thumbnail_grid(video_files, num_cols=6):
    """创建一个固定列数的缩略图网格"""
    total_files = len(video_files)
    num_rows = (total_files + num_cols - 1) // num_cols
    rows = []
    
    for i in range(num_rows):
        start_idx = i * num_cols
        end_idx = min(start_idx + num_cols, total_files)
        rows.append(video_files[start_idx:end_idx])
    
    return rows

@st.cache_resource
def get_video_bytes(video_path):
    if os.path.exists(video_path):
        with open(video_path, 'rb') as f:
            return f.read()
    return None

def main():
    try:
        # Page layout
        st.set_page_config(layout="wide")

        # Create two columns for layout
        left_col, right_col = st.columns([0.4, 0.6])

        # Left column - Text Editor
        with left_col:
            st.markdown("### Script Editor")
            script = st.text_area(label="editor",label_visibility="hidden", height=300)
            if st.button("Generate Video"):
                if script.strip():
                    with st.spinner('Generating video...'):
                        try:
                            # Generate videos
                            assets = text2video(script)
                            
                            # Save assets to session state
                            st.session_state.assets = assets
                            
                            # Update video files list
                            video_files = [asset['video_file'] for asset in assets.values()]
                            st.session_state.video_files = sorted(video_files)
                            if video_files:
                                st.session_state.current_video = video_files[0]
                            
                            st.success('Video generated successfully!')
                        except Exception as e:
                            logger.error(f"Error during video generation: {str(e)}")
                            logger.exception("Full stack trace:")
                            st.error(f"Error generating video: {str(e)}")
                else:
                    st.warning('Please enter a script first.')

        # Right column - Video Preview and Thumbnails
        with right_col:
            # Video preview
            st.markdown("### Video Preview")
            if st.session_state.get('current_video') and os.path.exists(st.session_state.current_video):
                video_bytes = get_video_bytes(st.session_state.current_video)
                if video_bytes:
                    st.video(video_bytes)
            else:
                st.info("没有可用的视频。请先生成视频或将视频文件放入 output/video 文件夹。")

            # Thumbnail list
            st.markdown("### Video Clips")
            video_files = scan_video_files()
            if video_files:
                thumbnail_rows = create_thumbnail_grid(video_files)
                
                for row in thumbnail_rows:
                    cols = st.columns(len(row))
                    for idx, (col, video_file) in enumerate(zip(cols, row)):
                        with col:
                            if os.path.exists(video_file):
                                try:
                                    # Use cached thumbnail if available
                                    thumbnail_path = os.path.join('output/thumbnails', os.path.splitext(os.path.basename(video_file))[0] + '.jpg')
                                    if not os.path.exists(thumbnail_path):
                                        from makevideo import generate_thumbnail
                                        thumbnail_path = generate_thumbnail(video_file)
                                    
                                    if thumbnail_path and os.path.exists(thumbnail_path):
                                        image = Image.open(thumbnail_path)
                                        image = image.resize((160, 90), Image.Resampling.LANCZOS)
                                        # 只显示缩略图，不需要按钮
                                        st.image(image, caption=f"片段 {idx+1}", width=160, use_container_width=False)
                                        # 添加一个点击事件处理器
                                        if st.button("选择", key=f"select_{video_file}"):
                                            st.session_state.current_video = video_file
                                            st.rerun()
                                except Exception as e:
                                    logger.error(f"Error loading thumbnail: {str(e)}")
                                    continue  # 如果出错就跳过当前缩略图
            else:
                st.info("暂无视频片段。请先生成视频或将视频文件放入 output/video 文件夹。")

        # Add button to merge videos
        if 'assets' in st.session_state and st.button("合并所有视频"):
            with st.spinner("正在合并视频..."):
                try:
                    video_files = [asset['video_file'] for asset in st.session_state.assets.values()]
                    output_path = os.path.join('output/video', "final_video.mp4")
                    merge_videos(video_files, output_path)
                    st.success("视频合并完成！")
                    video_bytes = get_video_bytes(output_path)
                    if video_bytes:
                        st.video(video_bytes)
                except Exception as e:
                    logger.error(f"Error merging videos: {str(e)}")
                    st.error(f"合并视频时出错: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        st.error("An unexpected error occurred. Please try refreshing the page.")

# Modify the main entry point
if __name__ == "__main__":
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main()