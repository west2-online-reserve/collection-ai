"""工具函数模块：文件名处理、目录创建等辅助功能"""

import re
import os
import requests
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import unquote

def ensure_dir(dir_path: str | Path) -> None:
    """创建目录（如果不存在）"""
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

def safe_filename(name: str) -> str:
    """清理文件名，移除文件系统非法字符
    
    Args:
        name: 原始文件名
    
    Returns:
        清理后的合法文件名
    """
    # 解码 URL 编码的文件名
    name = unquote(name)
    return re.sub(r'[\\/:*?"<>|]+', '_', name)

def download_file(url: str, save_dir: str | Path, filename: Optional[str] = None,
                 headers: Optional[Dict] = None) -> str:
    """下载文件并保存到指定目录
    
    Args:
        url: 下载链接
        save_dir: 保存目录
        filename: 保存的文件名，如果为 None 则从 URL 获取
        headers: 请求头
    
    Returns:
        保存的文件路径
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
        
    # 确保目录存在
    ensure_dir(save_dir)
    
    # 如果没有提供文件名，从 URL 获取
    if not filename:
        filename = os.path.basename(url)
    
    # 确保文件名合法
    filename = safe_filename(filename)
    
    # 构建保存路径
    save_path = save_dir / filename
    
    # 如果文件已存在，在文件名后添加数字
    base, ext = os.path.splitext(filename)
    counter = 1
    while save_path.exists():
        new_filename = f"{base}_{counter}{ext}"
        save_path = save_dir / new_filename
        counter += 1
    
    # 下载文件
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        return str(save_path)
    except Exception as e:
        print(f"下载文件失败 {url}: {e}")
        return ""