import numpy as np
import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# 导入点云IO模块
from cloud_io import save_ply_file

def generate_sphere_pointcloud(radius: float = 1.0, num_points: int = 5000) -> tuple:
    """
    生成球体点云
    
    Args:
        radius: 球体半径
        num_points: 点的数量
    
    Returns:
        tuple: (点坐标数组, 颜色数组)
    """
    # 使用球坐标系生成均匀分布的点
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    points = np.vstack([x, y, z]).T
    
    # 根据位置生成颜色（热力图效果）
    colors = np.zeros((num_points, 3))
    # 基于z坐标的颜色映射
    colors[:, 0] = (z + radius) / (2 * radius)  # 红色
    colors[:, 2] = 1 - colors[:, 0]  # 蓝色
    colors[:, 1] = 0.5 * np.ones(num_points)  # 绿色适中
    
    return points, colors

def generate_cube_pointcloud(size: float = 2.0, num_points: int = 5000) -> tuple:
    """
    生成立方体点云
    
    Args:
        size: 立方体边长
        num_points: 点的数量
    
    Returns:
        tuple: (点坐标数组, 颜色数组)
    """
    half_size = size / 2
    
    # 生成立方体表面的随机点
    points = []
    colors = []
    
    # 生成6个面
    faces = [
        # (normal direction, color)
        ((1, 0, 0), (1, 0, 0)),    # 红色前面
        ((-1, 0, 0), (0, 0, 1)),   # 蓝色后面
        ((0, 1, 0), (0, 1, 0)),    # 绿色上面
        ((0, -1, 0), (1, 1, 0)),   # 黄色下面
        ((0, 0, 1), (1, 0, 1)),    # 品红右面
        ((0, 0, -1), (0, 1, 1)),   # 青色左面
    ]
    
    points_per_face = num_points // len(faces)
    
    for (nx, ny, nz), (r, g, b) in faces:
        # 生成面内的随机点
        for _ in range(points_per_face):
            # 根据法线方向确定固定坐标
            fixed_coord = half_size if nx > 0 or ny > 0 or nz > 0 else -half_size
            
            if nx != 0:  # x固定
                x = fixed_coord
                y = np.random.uniform(-half_size, half_size)
                z = np.random.uniform(-half_size, half_size)
            elif ny != 0:  # y固定
                x = np.random.uniform(-half_size, half_size)
                y = fixed_coord
                z = np.random.uniform(-half_size, half_size)
            else:  # z固定
                x = np.random.uniform(-half_size, half_size)
                y = np.random.uniform(-half_size, half_size)
                z = fixed_coord
            
            points.append([x, y, z])
            colors.append([r, g, b])
    
    # 处理剩余的点
    remaining_points = num_points - len(points)
    if remaining_points > 0:
        for _ in range(remaining_points):
            face_idx = np.random.randint(0, len(faces))
            (nx, ny, nz), (r, g, b) = faces[face_idx]
            
            fixed_coord = half_size if nx > 0 or ny > 0 or nz > 0 else -half_size
            
            if nx != 0:
                x = fixed_coord
                y = np.random.uniform(-half_size, half_size)
                z = np.random.uniform(-half_size, half_size)
            elif ny != 0:
                x = np.random.uniform(-half_size, half_size)
                y = fixed_coord
                z = np.random.uniform(-half_size, half_size)
            else:
                x = np.random.uniform(-half_size, half_size)
                y = np.random.uniform(-half_size, half_size)
                z = fixed_coord
            
            points.append([x, y, z])
            colors.append([r, g, b])
    
    return np.array(points), np.array(colors)

def generate_cylinder_pointcloud(radius: float = 1.0, height: float = 2.0, num_points: int = 5000) -> tuple:
    """
    生成圆柱体点云
    
    Args:
        radius: 圆柱体半径
        height: 圆柱体高度
        num_points: 点的数量
    
    Returns:
        tuple: (点坐标数组, 颜色数组)
    """
    points = []
    colors = []
    
    # 分配点数量：侧面60%，顶部20%，底部20%
    side_points = int(num_points * 0.6)
    top_points = int(num_points * 0.2)
    bottom_points = num_points - side_points - top_points
    
    # 生成圆柱体侧面的点
    for _ in range(side_points):
        # 随机角度和高度
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-height/2, height/2)
        
        # 极坐标转换为笛卡尔坐标
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        points.append([x, y, z])
        
        # 基于高度的颜色映射（从底部蓝色到顶部红色）
        t = (z + height/2) / height
        colors.append([t, 0.5 * (1 - abs(2*t - 1)), 1 - t])
    
    # 生成圆柱体顶部的点
    for _ in range(top_points):
        # 顶部圆形区域内的随机点
        r = radius * np.sqrt(np.random.uniform(0, 1))  # 使用平方根使点均匀分布
        theta = np.random.uniform(0, 2 * np.pi)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = height / 2
        
        points.append([x, y, z])
        colors.append([1, 0, 0])  # 顶部为红色
    
    # 生成圆柱体底部的点
    for _ in range(bottom_points):
        # 底部圆形区域内的随机点
        r = radius * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2 * np.pi)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = -height / 2
        
        points.append([x, y, z])
        colors.append([0, 0, 1])  # 底部为蓝色
    
    return np.array(points), np.array(colors)

def generate_sample_pointclouds(output_dir: str):
    """
    生成左右摄像头的示例点云数据
    
    Args:
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成示例点云数据...")
    
    # 生成左摄像头点云（圆柱体）
    left_points, left_colors = generate_cylinder_pointcloud(radius=0.8, height=2.0, num_points=10000)
    # 将左摄像头点云向左平移
    left_points[:, 0] -= 1.0
    
    # 生成右摄像头点云（圆柱体）
    right_points, right_colors = generate_cylinder_pointcloud(radius=0.8, height=2.0, num_points=10000)
    # 将右摄像头点云向右平移
    right_points[:, 0] += 1.0
    
    # 保存为PLY文件
    left_file = os.path.join(output_dir, 'left_camera.ply')
    right_file = os.path.join(output_dir, 'right_camera.ply')
    
    save_ply_file(left_file, left_points, left_colors)
    save_ply_file(right_file, right_points, right_colors)
    
    print(f"\n示例点云数据已生成:")
    print(f"- 左摄像头: {left_file} ({len(left_points)}个点)")
    print(f"- 右摄像头: {right_file} ({len(right_points)}个点)")

def main():
    # 获取当前脚本所在目录的父目录
    current_dir = os.path.dirname(__file__)
    output_dir = current_dir  # 使用当前目录作为输出
    
    generate_sample_pointclouds(output_dir)

if __name__ == "__main__":
    main()