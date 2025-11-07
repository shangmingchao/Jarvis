import numpy as np
from typing import Tuple


def validate_pointcloud(points: np.ndarray, colors: np.ndarray) -> bool:
    """
    验证点云数据有效性
    
    Args:
        points: 坐标数组(Nx3)
        colors: 颜色数组(Nx3)
    
    Returns:
        bool: 数据是否有效
    """
    # 检查数据类型
    if not isinstance(points, np.ndarray) or not isinstance(colors, np.ndarray):
        return False
    
    # 检查维度
    if points.ndim != 2 or points.shape[1] != 3:
        return False
    
    if colors.ndim != 2 or colors.shape[1] != 3:
        return False
    
    # 检查长度是否匹配
    if len(points) != len(colors):
        return False
    
    # 检查是否包含NaN或Inf
    if np.isnan(points).any() or np.isinf(points).any():
        return False
    
    # 检查颜色值是否在合理范围内
    if np.min(colors) < 0 or np.max(colors) > 1:
        return False
    
    # 检查点数是否合理
    if len(points) == 0 or len(points) > 1000000:  # 限制在100万个点以内
        return False
    
    return True


def normalize_coordinates(points: np.ndarray) -> np.ndarray:
    """
    归一化坐标数据
    
    Args:
        points: 原始坐标数组(Nx3)
    
    Returns:
        np.ndarray: 归一化后的坐标数组
    """
    # 计算边界框
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    # 计算中心和范围
    center = (min_vals + max_vals) / 2
    range_vals = max_vals - min_vals
    
    # 避免除以零
    max_range = np.max(range_vals)
    if max_range < 1e-10:
        return points - center  # 只有当范围非常小时，只进行中心化
    
    # 归一化到[-1, 1]范围
    normalized_points = (points - center) / max_range
    
    return normalized_points


def filter_outliers(points: np.ndarray, colors: np.ndarray, threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用统计方法过滤异常值
    
    Args:
        points: 坐标数组(Nx3)
        colors: 颜色数组(Nx3)
        threshold: 标准差阈值，默认为2.0
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 过滤后的坐标和颜色数组
    """
    # 计算每个点到均值的距离
    mean_point = np.mean(points, axis=0)
    distances = np.linalg.norm(points - mean_point, axis=1)
    
    # 计算距离的均值和标准差
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # 找到在阈值范围内的点
    mask = distances < (mean_distance + threshold * std_distance)
    
    # 应用过滤
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    return filtered_points, filtered_colors


def remove_duplicate_points(points: np.ndarray, colors: np.ndarray, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    移除重复的点
    
    Args:
        points: 坐标数组(Nx3)
        colors: 颜色数组(Nx3)
        tolerance: 点之间的距离容忍度
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 去重后的坐标和颜色数组
    """
    # 使用四舍五入到指定精度来检测重复点
    rounded_points = np.round(points / tolerance) * tolerance
    
    # 找到唯一的点
    unique_points, indices = np.unique(rounded_points, axis=0, return_index=True)
    
    # 返回去重后的点和对应的颜色
    return points[indices], colors[indices]


def downsample_pointcloud(points: np.ndarray, colors: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    对点云进行降采样
    
    Args:
        points: 坐标数组(Nx3)
        colors: 颜色数组(Nx3)
        target_size: 目标点数
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 降采样后的坐标和颜色数组
    """
    if len(points) <= target_size:
        return points, colors
    
    # 随机选择点
    indices = np.random.choice(len(points), target_size, replace=False)
    
    return points[indices], colors[indices]


if __name__ == "__main__":
    # 测试功能
    print("点云预处理模块测试")
    print("功能: 数据验证、坐标归一化、异常值过滤、去重、降采样")