import os
import argparse
import numpy as np
import logging
import sys
import matplotlib

# 设置matplotlib字体支持中文（尝试多种字体确保兼容性）
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'WenQuanYi Micro Hei', 'Heiti TC', 'SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from cloud_io import load_ply_file, save_ply_file
from preprocessing import validate_pointcloud, normalize_coordinates, filter_outliers
from feature_extraction import extract_sift_features, save_features, limit_feature_count
from visualization import create_interactive_viewer, visualize_comparison

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='双目视觉点云处理系统')
    parser.add_argument('--left', type=str, default='../data/left_camera.ply',
                        help='左摄像头PLY文件路径')
    parser.add_argument('--right', type=str, default='../data/right_camera.ply',
                        help='右摄像头PLY文件路径')
    parser.add_argument('--output', type=str, default='../output/',
                        help='输出目录路径')
    parser.add_argument('--visualize', type=bool, default=True,
                        help='是否启用可视化')
    parser.add_argument('--save_features', type=bool, default=True,
                        help='是否保存特征点')
    parser.add_argument('--max_features', type=int, default=1000,
                        help='最大特征点数量')
    # parser.add_argument('--export_html', type=bool, default=False,
    #                     help='是否导出HTML可视化')  # 暂时禁用HTML导出功能
    return parser.parse_args()


def process_pointcloud(filepath: str) -> dict:
    """
    处理单个点云文件
    
    Args:
        filepath: 点云文件路径
    
    Returns:
        dict: 包含原始点云、预处理后点云、特征点等信息的字典
    """
    logger.info(f"开始处理点云文件: {filepath}")
    
    # 1. 加载点云数据
    try:
        points, colors = load_ply_file(filepath)
        logger.info(f"成功加载点云数据，点数: {len(points)}")
    except Exception as e:
        logger.error(f"加载点云失败: {str(e)}")
        raise
    
    # 2. 验证数据
    if not validate_pointcloud(points, colors):
        logger.error("点云数据验证失败")
        raise ValueError("无效的点云数据")
    
    # 3. 预处理
    try:
        # 过滤异常值
        filtered_points, filtered_colors = filter_outliers(points, colors, threshold=2.0)
        logger.info(f"过滤后点数: {len(filtered_points)}")
        
        # 坐标归一化
        normalized_points = normalize_coordinates(filtered_points)
        logger.info("坐标归一化完成")
    except Exception as e:
        logger.error(f"预处理失败: {str(e)}")
        raise
    
    # 4. 特征提取
    try:
        keypoints, descriptors = extract_sift_features(normalized_points, filtered_colors)
        logger.info(f"成功提取特征点: {len(keypoints)}")
        
        # 限制特征点数量
        keypoints, descriptors = limit_feature_count(keypoints, descriptors, max_count=args.max_features)
        logger.info(f"限制后特征点数量: {len(keypoints)}")
    except Exception as e:
        logger.error(f"特征提取失败: {str(e)}")
        # 如果特征提取失败，使用空数组继续
        keypoints, descriptors = np.array([]), np.array([])
    
    return {
        'original_points': points,
        'original_colors': colors,
        'filtered_points': filtered_points,
        'filtered_colors': filtered_colors,
        'normalized_points': normalized_points,
        'keypoints': keypoints,
        'descriptors': descriptors
    }


def main():
    """
    主程序入口
    """
    global args
    args = parse_arguments()
    
    logger.info("开始运行双目视觉点云处理系统")
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # 处理左摄像头点云
        left_data = process_pointcloud(args.left)
        
        # 处理右摄像头点云
        right_data = process_pointcloud(args.right)
        
        # 保存特征点
        if args.save_features:
            try:
                left_feature_file = os.path.join(args.output, 'features_left.npy')
                right_feature_file = os.path.join(args.output, 'features_right.npy')
                
                save_features(left_feature_file, left_data['keypoints'], left_data['descriptors'])
                save_features(right_feature_file, right_data['keypoints'], right_data['descriptors'])
                
                logger.info(f"特征点数据已保存到 {args.output}")
            except Exception as e:
                logger.error(f"保存特征点失败: {str(e)}")
        
        # 可视化
        if args.visualize:
            try:
                logger.info("启动可视化...")
                
                # 分别可视化左右点云
                print("\n=== 左摄像头点云 ===")
                create_interactive_viewer(
                    left_data['filtered_points'],
                    left_data['filtered_colors'],
                    left_data['keypoints']
                )
                
                print("\n=== 右摄像头点云 ===")
                create_interactive_viewer(
                    right_data['filtered_points'],
                    right_data['filtered_colors'],
                    right_data['keypoints']
                )
                
                # 同时显示两个点云进行对比
                print("\n=== 双目点云对比 ===")
                visualize_comparison(
                    left_data['filtered_points'],
                    left_data['filtered_colors'],
                    left_data['keypoints'],
                    right_data['filtered_points'],
                    right_data['filtered_colors'],
                    right_data['keypoints']
                )
                
            except Exception as e:
                logger.error(f"可视化失败: {str(e)}")
        
        # 导出HTML功能已暂时禁用（依赖open3d）
        
        logger.info("处理完成！")
        
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()