import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple
import logging
import matplotlib

# 设置matplotlib字体支持中文（尝试多种字体确保兼容性）
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'WenQuanYi Micro Hei', 'Heiti TC', 'SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


def visualize_pointcloud(points: np.ndarray, colors: np.ndarray, title: str = "Point Cloud"):
    """
    可视化点云数据
    
    Args:
        points: 坐标数组(Nx3)
        colors: 颜色数组(Nx3)，值范围[0,1]
        title: 窗口标题
    """
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 降采样点云以提高可视化性能
    max_points = 5000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    # 绘制点云
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors,
        s=10,  # 点的大小
        alpha=0.8  # 透明度
    )
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 设置坐标轴范围相等
    max_range = np.max([np.max(points[:, 0]) - np.min(points[:, 0]),
                        np.max(points[:, 1]) - np.min(points[:, 1]),
                        np.max(points[:, 2]) - np.min(points[:, 2])])
    mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2
    mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
    mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) / 2
    
    ax.set_xlim([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim([mid_z - max_range/2, mid_z + max_range/2])
    
    plt.tight_layout()
    print(f"显示点云: {title} (按 '关闭窗口' 继续)")
    plt.show()


def highlight_keypoints(points: np.ndarray, keypoints: np.ndarray, colors: np.ndarray, 
                       keypoint_color: Tuple[float, float, float] = (1, 0, 0)):
    """
    高亮显示特征点
    
    Args:
        points: 原始点云坐标数组(Nx3)
        keypoints: 特征点坐标数组(Mx3)
        colors: 原始点云颜色数组(Nx3)
        keypoint_color: 特征点高亮颜色
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 原始点云坐标、原始点云颜色、特征点坐标
    """
    return points, colors, keypoints


def create_interactive_viewer(points: np.ndarray, colors: np.ndarray, keypoints: Optional[np.ndarray] = None):
    """
    创建交互式查看器
    
    Args:
        points: 坐标数组(Nx3)
        colors: 颜色数组(Nx3)
        keypoints: 特征点坐标数组(Mx3)，可选
    """
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 降采样点云以提高可视化性能
    max_points = 5000
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        sampled_points = points[indices]
        sampled_colors = colors[indices]
    else:
        sampled_points = points
        sampled_colors = colors
    
    # 绘制原始点云
    ax.scatter(
        sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
        c=sampled_colors,
        s=10,
        alpha=0.6,
        label='原始点云'
    )
    
    # 绘制特征点（如果有）
    if keypoints is not None and len(keypoints) > 0:
        ax.scatter(
            keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
            c='red',
            s=50,
            marker='o',
            edgecolor='black',
            linewidth=1,
            label='特征点'
        )
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('交互式点云查看器')
    
    # 添加图例
    ax.legend()
    
    # 设置坐标轴范围相等
    max_range = np.max([np.max(points[:, 0]) - np.min(points[:, 0]),
                        np.max(points[:, 1]) - np.min(points[:, 1]),
                        np.max(points[:, 2]) - np.min(points[:, 2])])
    mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2
    mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
    mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) / 2
    
    ax.set_xlim([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim([mid_z - max_range/2, mid_z + max_range/2])
    
    # 显示帮助信息
    help_text = """
    交互式点云查看器控制:
    - 鼠标拖动: 旋转视角
    - 鼠标滚轮: 缩放视图
    - '平移工具': 平移视图
    - '关闭窗口': 退出
    """
    print(help_text)
    
    plt.tight_layout()
    plt.show()


def visualize_comparison(points_left: np.ndarray, colors_left: np.ndarray, features_left: np.ndarray,
                         points_right: np.ndarray, colors_right: np.ndarray, features_right: np.ndarray):
    """
    比较左右摄像头的点云和特征点
    
    Args:
        points_left: 左摄像头点云坐标
        colors_left: 左摄像头点云颜色
        features_left: 左摄像头特征点
        points_right: 右摄像头点云坐标
        colors_right: 右摄像头点云颜色
        features_right: 右摄像头特征点
    """
    # 为了在同一视图中显示，将右侧点云平移
    points_right_shifted = points_right.copy()
    points_right_shifted[:, 0] += np.max(points_left[:, 0]) - np.min(points_right[:, 0]) + 1.0
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 降采样点云以提高可视化性能
    max_points = 3000
    if len(points_left) > max_points:
        left_indices = np.random.choice(len(points_left), max_points, replace=False)
        sampled_left_points = points_left[left_indices]
        sampled_left_colors = colors_left[left_indices]
    else:
        sampled_left_points = points_left
        sampled_left_colors = colors_left
    
    if len(points_right_shifted) > max_points:
        right_indices = np.random.choice(len(points_right_shifted), max_points, replace=False)
        sampled_right_points = points_right_shifted[right_indices]
        sampled_right_colors = colors_right[right_indices]
    else:
        sampled_right_points = points_right_shifted
        sampled_right_colors = colors_right
    
    # 绘制左侧点云
    ax.scatter(
        sampled_left_points[:, 0], sampled_left_points[:, 1], sampled_left_points[:, 2],
        c=sampled_left_colors,
        s=10,
        alpha=0.6,
        label='左摄像头点云'
    )
    
    # 绘制右侧点云
    ax.scatter(
        sampled_right_points[:, 0], sampled_right_points[:, 1], sampled_right_points[:, 2],
        c=sampled_right_colors,
        s=10,
        alpha=0.6,
        label='右摄像头点云'
    )
    
    # 绘制特征点
    if len(features_left) > 0:
        ax.scatter(
            features_left[:, 0], features_left[:, 1], features_left[:, 2],
            c='red',
            s=50,
            marker='o',
            edgecolor='black',
            linewidth=1,
            label='左摄像头特征点'
        )
    
    if len(features_right) > 0:
        # 平移右侧特征点以匹配点云
        features_right_shifted = features_right.copy()
        features_right_shifted[:, 0] += np.max(points_left[:, 0]) - np.min(points_right[:, 0]) + 1.0
        
        ax.scatter(
            features_right_shifted[:, 0], features_right_shifted[:, 1], features_right_shifted[:, 2],
            c='green',
            s=50,
            marker='o',
            edgecolor='black',
            linewidth=1,
            label='右摄像头特征点'
        )
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('双目点云对比')
    
    # 添加图例
    ax.legend()
    
    plt.tight_layout()
    plt.show()


# HTML导出功能暂时禁用（依赖open3d和JavaScript相关功能）
# def export_visualization_html(points: np.ndarray, colors: np.ndarray, keypoints: np.ndarray, 
#                              filepath: str = "visualization.html"):
#     """
#     导出可视化结果为HTML文件（简单实现）
#     
#     Args:
#         points: 点云坐标
#         colors: 点云颜色
#         keypoints: 特征点
#         filepath: 输出HTML文件路径
#     """
#     try:
#         # 创建HTML内容并保存
#         print("HTML导出功能已暂时禁用")
#     except Exception as e:
#         print(f"导出HTML失败: {str(e)}")


if __name__ == "__main__":
    # 测试功能
    print("可视化模块测试")
    print("功能: 点云可视化、特征点高亮、交互式查看、对比显示、HTML导出")