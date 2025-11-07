import numpy as np
import cv2
import os
from typing import Tuple


def extract_sift_features(points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从点云数据中提取SIFT特征点
    注意：SIFT通常用于2D图像，这里我们将3D点云投影到2D平面后提取特征
    
    Args:
        points: 坐标数组(Nx3)
        colors: 颜色数组(Nx3)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 特征点坐标数组和描述子数组
    """
    # 创建一个虚拟的深度图像用于SIFT特征提取
    # 这里我们使用点云的x-y平面投影，并使用z值作为强度
    
    # 归一化坐标范围到[0, 512]作为图像尺寸
    min_vals = np.min(points[:, :2], axis=0)
    max_vals = np.max(points[:, :2], axis=0)
    range_vals = max_vals - min_vals
    
    # 避免除以零
    if np.any(range_vals < 1e-10):
        range_vals = np.ones_like(range_vals)
    
    # 映射到图像坐标
    img_size = 512
    img_coords = ((points[:, :2] - min_vals) / range_vals * (img_size - 1)).astype(np.int32)
    
    # 确保坐标在有效范围内
    img_coords = np.clip(img_coords, 0, img_size - 1)
    
    # 创建灰度图像（使用z坐标作为强度值）
    z_values = points[:, 2]
    min_z, max_z = np.min(z_values), np.max(z_values)
    if max_z - min_z > 1e-10:
        intensity = ((z_values - min_z) / (max_z - min_z) * 255).astype(np.uint8)
    else:
        intensity = np.ones_like(z_values, dtype=np.uint8) * 128
    
    # 创建图像
    depth_img = np.zeros((img_size, img_size), dtype=np.uint8)
    for i in range(len(points)):
        x, y = img_coords[i]
        depth_img[y, x] = intensity[i]
    
    # 应用高斯模糊减少噪声
    depth_img = cv2.GaussianBlur(depth_img, (5, 5), 0)
    
    # 创建SIFT对象并提取特征点
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(depth_img, None)
    
    if len(keypoints) == 0:
        return np.array([]), np.array([])
    
    # 将2D特征点映射回3D空间
    feature_points = []
    for kp in keypoints:
        # 找到最近的3D点
        img_x, img_y = int(kp.pt[0]), int(kp.pt[1])
        distances = np.sum((img_coords - np.array([img_x, img_y])) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        feature_points.append(points[nearest_idx])
    
    return np.array(feature_points), descriptors


def save_features(filepath: str, keypoints: np.ndarray, descriptors: np.ndarray):
    """
    保存特征点数据到NumPy文件
    
    Args:
        filepath: 输出文件路径
        keypoints: 特征点坐标数组(Nx3)
        descriptors: 描述子数组(Nx128)
    
    Raises:
        IOError: 文件写入错误
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    try:
        # 将特征点和描述子保存为字典
        features_data = {
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        
        np.save(filepath, features_data)
        
    except Exception as e:
        raise IOError(f"保存特征点数据失败: {str(e)}")


def load_features(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从NumPy文件加载特征点数据
    
    Args:
        filepath: 文件路径
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 特征点坐标数组和描述子数组
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        # 加载数据
        features_data = np.load(filepath, allow_pickle=True).item()
        
        # 验证数据格式
        if 'keypoints' not in features_data or 'descriptors' not in features_data:
            raise ValueError("特征点数据格式错误")
        
        return features_data['keypoints'], features_data['descriptors']
        
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise ValueError(f"加载特征点数据失败: {str(e)}")


def filter_features_by_quality(keypoints: np.ndarray, descriptors: np.ndarray, quality_threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据特征点质量进行过滤
    
    Args:
        keypoints: 特征点坐标数组
        descriptors: 描述子数组
        quality_threshold: 质量阈值
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 过滤后的特征点和描述子
    """
    # 计算每个特征点的描述子的平均幅值作为质量指标
    if len(descriptors) == 0:
        return keypoints, descriptors
    
    descriptor_norms = np.linalg.norm(descriptors, axis=1)
    quality_scores = descriptor_norms / (np.max(descriptor_norms) + 1e-10)
    
    # 应用阈值过滤
    mask = quality_scores > quality_threshold
    
    return keypoints[mask], descriptors[mask]


def limit_feature_count(keypoints: np.ndarray, descriptors: np.ndarray, max_count: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    限制特征点数量
    
    Args:
        keypoints: 特征点坐标数组
        descriptors: 描述子数组
        max_count: 最大特征点数量
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 限制数量后的特征点和描述子
    """
    if len(keypoints) <= max_count:
        return keypoints, descriptors
    
    # 计算描述子的幅值并按降序排序
    descriptor_norms = np.linalg.norm(descriptors, axis=1)
    sorted_indices = np.argsort(descriptor_norms)[::-1]
    
    # 选择前max_count个特征点
    selected_indices = sorted_indices[:max_count]
    
    return keypoints[selected_indices], descriptors[selected_indices]


if __name__ == "__main__":
    # 测试功能
    print("特征提取模块测试")
    print("功能: SIFT特征提取、特征点保存与加载、特征点质量过滤")