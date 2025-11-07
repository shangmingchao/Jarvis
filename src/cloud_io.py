import numpy as np
from typing import Tuple
import os
import plyfile


def load_ply_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载PLY文件并返回坐标和颜色数据
    
    Args:
        filepath: PLY文件路径
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 坐标数组(Nx3)和颜色数组(Nx3)
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            plydata = plyfile.PlyData.read(f)
        
        vertices = plydata['vertex']
        
        # 提取坐标数据
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        
        # 提取颜色数据（如果存在）
        colors = None
        if 'red' in vertices.data.dtype.names:
            r = vertices['red'].astype(np.float32) / 255.0
            g = vertices['green'].astype(np.float32) / 255.0
            b = vertices['blue'].astype(np.float32) / 255.0
            colors = np.vstack([r, g, b]).T
        else:
            # 如果没有颜色数据，使用默认颜色
            colors = np.ones((len(points), 3)) * 0.5
        
        return points, colors
        
    except Exception as e:
        raise ValueError(f"读取PLY文件失败: {str(e)}")


def save_ply_file(filepath: str, points: np.ndarray, colors: np.ndarray):
    """
    保存点云数据到PLY文件
    
    Args:
        filepath: 输出文件路径
        points: 坐标数组(Nx3)
        colors: 颜色数组(Nx3)，值范围[0,1]
    
    Raises:
        ValueError: 数据格式错误
        IOError: 文件写入错误
    """
    # 验证输入数据
    if len(points) != len(colors):
        raise ValueError("点云数据和颜色数据长度不匹配")
    
    if points.shape[1] != 3:
        raise ValueError("点云数据必须是Nx3格式")
    
    if colors.shape[1] != 3:
        raise ValueError("颜色数据必须是Nx3格式")
    
    # 确保颜色值在有效范围内
    colors = np.clip(colors, 0, 1)
    
    # 转换颜色为整数格式
    colors_int = (colors * 255).astype(np.uint8)
    
    try:
        # 创建顶点数据
        vertex_data = np.empty(len(points), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['red'] = colors_int[:, 0]
        vertex_data['green'] = colors_int[:, 1]
        vertex_data['blue'] = colors_int[:, 2]
        
        # 创建PLY元素和文件
        vertex_element = plyfile.PlyElement.describe(vertex_data, 'vertex')
        plydata = plyfile.PlyData([vertex_element], text=True)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 保存文件
        plydata.write(filepath)
        
    except Exception as e:
        raise IOError(f"保存PLY文件失败: {str(e)}")


if __name__ == "__main__":
    # 测试功能
    print("点云IO模块测试")
    print("功能: 读取和保存PLY格式点云文件")