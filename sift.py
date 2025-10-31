import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设的双目摄像头参数（实际应用中需要根据相机标定结果设置）
# 相机内参
fx = 520.9  # 焦距，像素单位
fy = 521.0
cx = 325.1
cy = 249.7
# 基线，单位为米
baseline = 0.054

# 使用最新的SIFT实现，避免弃用警告
def get_sift():
    try:
        # 对于OpenCV 4.5+版本，SIFT已经移到主仓库
        return cv2.SIFT_create()
    except:
        # 回退到旧版本的实现
        return cv2.xfeatures2d.SIFT_create()

def process_image(img):
    """处理单个图像，提取SIFT特征点"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = get_sift()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return gray, keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """匹配两个图像的特征点"""
    # 使用FLANN匹配器进行快速特征匹配
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # 使用Lowe比率测试筛选良好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    return good_matches

def calculate_disparity(left_keypoints, right_keypoints, matches):
    """计算视差图"""
    disparities = []
    points_3d = []
    
    for match in matches:
        # 获取左右图像中匹配点的坐标
        left_pt = left_keypoints[match.queryIdx].pt
        right_pt = right_keypoints[match.trainIdx].pt
        
        # 计算视差（假设特征点在同一行）
        disparity = left_pt[0] - right_pt[0]
        
        # 只考虑有效视差
        if disparity > 0:
            # 计算深度
            depth = (fx * baseline) / disparity
            
            # 计算3D点坐标
            x = (left_pt[0] - cx) * depth / fx
            y = (left_pt[1] - cy) * depth / fy
            z = depth
            
            points_3d.append([x, y, z])
            disparities.append(disparity)
    
    return np.array(disparities), np.array(points_3d)

def visualize_point_cloud(points_3d):
    """可视化点云数据"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if len(points_3d) > 0:
        # 从点云中提取x, y, z坐标
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        
        # 绘制3D散点图
        ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Cloud with SIFT Features')
        
        plt.show()

def get_images_from_cameras():
    """从摄像头获取图像
    在实际应用中，这里会分别从左右摄像头获取图像
    现在使用模拟数据，读取同一张图像进行演示
    """
    # 实际应用中应该分别打开两个摄像头
    # cap_left = cv2.VideoCapture(0)  # 左摄像头
    # cap_right = cv2.VideoCapture(1)  # 右摄像头
    
    # 当前使用同一图像作为左右摄像头的输入，模拟双目视觉
    left_img = cv2.imread('./image/doraemon1.jpg')
    right_img = cv2.imread('./image/doraemon1.jpg')
    
    # 为了模拟视差，我们可以对右图像进行轻微的水平平移
    if right_img is not None:
        # 平移右图像，模拟视差
        M = np.float32([[1, 0, -10], [0, 1, 0]])  # 水平左移10个像素
        rows, cols = right_img.shape[:2]
        right_img = cv2.warpAffine(right_img, M, (cols, rows))
    
    return left_img, right_img

def main():
    # 从摄像头获取图像（这里使用模拟数据）
    left_img, right_img = get_images_from_cameras()
    
    if left_img is None or right_img is None:
        print("无法获取图像，请检查图像路径或摄像头连接")
        return
    
    # 处理左右图像，提取SIFT特征
    left_gray, left_keypoints, left_descriptors = process_image(left_img)
    right_gray, right_keypoints, right_descriptors = process_image(right_img)
    
    # 匹配特征点
    matches = match_features(left_descriptors, right_descriptors)
    
    # 计算视差和3D点云
    disparities, points_3d = calculate_disparity(left_keypoints, right_keypoints, matches)
    
    print(f"找到 {len(matches)} 个匹配的特征点")
    print(f"生成 {len(points_3d)} 个3D点")
    
    # 绘制匹配的特征点
    img_matches = cv2.drawMatches(left_img, left_keypoints, right_img, right_keypoints, 
                                 matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)
    
    # 可视化点云数据
    visualize_point_cloud(points_3d)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()