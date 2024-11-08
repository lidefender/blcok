import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 配置 matplotlib 使用 SimHei 字体（黑体）
# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 反相灰度图，将黑白阈值颠倒
def accessPixel(img):
    """
    对灰度图像进行反相处理。

    参数：
        img (numpy.ndarray): 输入的灰度图像。

    返回：
        numpy.ndarray: 反相后的图像。
    """
    img = 255 - img
    return img

# 多种形态学操作
def applyMorphology(img):
    """
    对图像应用多种形态学操作，包括腐蚀、膨胀、开操作和闭操作。

    参数：
        img (numpy.ndarray): 输入的灰度图像。

    返回：
        numpy.ndarray: 经过形态学操作处理后的图像。
    """
    kernel = np.ones((5,5), np.uint8)

    # 腐蚀操作
    # eroded = cv2.erode(img, kernel, iterations=1)

    # 膨胀操作
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # 开操作（先腐蚀再膨胀）
    opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel, iterations=1)

    # 闭操作（先膨胀再腐蚀）
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed

# 提取峰值
def extractPeak(array_vals, min_vals=10, min_rect=20):
    """
    从一维数组中提取连续高于阈值的区间。

    参数：
        array_vals (numpy.ndarray): 一维数组。
        min_vals (int): 峰值的最小阈值。
        min_rect (int): 峰值区间的最小长度。

    返回：
        list: 包含多个 (start, end) 元组的列表。
    """
    extractPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint is None:
            startPoint = i
        elif point <= min_vals and startPoint is not None:
            endPoint = i

        if startPoint is not None and endPoint is not None:
            if endPoint - startPoint >= min_rect:
                extractPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None

    # 处理最后一个区域
    if startPoint is not None and endPoint is None:
        extractPoints.append((startPoint, len(array_vals)))

    return extractPoints

# 寻找边缘，返回边框的左上角和右下角
def findBorderHistogram(img, row_ratio=0.5, col_ratio=0.9, min_vals=10, min_rect=20):
    """
    使用直方图扫描法检测图像中的边界区域。

    参数：
        img (numpy.ndarray): 处理后的灰度图像。
        row_ratio (float): 行方向上计算区域的比例（默认为0.5，即中间50%的行）。
        col_ratio (float): 列方向上计算区域的比例（默认为0.9，即中间90%的列）。
        min_vals (int): 峰值检测的最小阈值。
        min_rect (int): 峰值区间的最小长度。

    返回：
        list: 包含多个边框坐标的列表，每个边框由左上角和右下角坐标组成。
    """
    borders = []
    height, width = img.shape
    # 定义行和列的计算范围
    row_start = int((1 - row_ratio) / 2 * height)  # 中间row_ratio的开始行
    row_end = int((1 + row_ratio) / 2 * height)    # 中间row_ratio的结束行
    col_start = int((1 - col_ratio) / 2 * width)   # 中间col_ratio的开始列
    col_end = int((1 + col_ratio) / 2 * width)     # 中间col_ratio的结束列

    # 行扫描（仅中间row_ratio）
    hori_vals = np.sum(img[row_start:row_end, :], axis=1)
    hori_points = extractPeak(hori_vals, min_vals, min_rect)

    # 根据每一行来扫描列（仅中间col_ratio）
    for hori_point in hori_points:
        # 调整行的索引回到原图中的位置
        actual_row_start = row_start + hori_point[0]
        actual_row_end = row_start + hori_point[1]
        extractImg = img[actual_row_start:actual_row_end, col_start:col_end]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeak(vec_vals, min_vals, min_rect)
        for vect_point in vec_points:
            # 调整列的索引回到原图中的位置
            actual_col_start = col_start + vect_point[0]
            actual_col_end = col_start + vect_point[1]
            border = [(actual_col_start, actual_row_start), (actual_col_end, actual_row_end)]
            borders.append(border)
    return borders

# 可视化直方图和图像
def visualizeHistogramAndImage(original_img, morphed_img, borders, row_ratio=0.5, col_ratio=0.9):
    """
    可视化图像及其在限制区域内的行和列直方图，并对齐显示。

    参数：
        original_img (numpy.ndarray): 原始彩色图像。
        morphed_img (numpy.ndarray): 经过形态学操作处理后的灰度图像。
        borders (list): 边框坐标列表。
        row_ratio (float): 行方向上计算区域的比例。
        col_ratio (float): 列方向上计算区域的比例。
    """
    height, width = morphed_img.shape
    row_start = int((1 - row_ratio) / 2 * height)
    row_end = int((1 + row_ratio) / 2 * height)
    col_start = int((1 - col_ratio) / 2 * width)
    col_end = int((1 + col_ratio) / 2 * width)

    # 行扫描
    hori_vals = np.sum(morphed_img[row_start:row_end, :], axis=1)
    # 列扫描
    vec_vals = np.sum(morphed_img[:, col_start:col_end], axis=0)

    # 创建绘图
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.2], height_ratios=[0.2, 1], 
                          wspace=0.05, hspace=0.05)

    # 行直方图（上方）
    ax_row = fig.add_subplot(gs[0, 0])
    ax_row.plot(hori_vals, color='blue')
    ax_row.set_title('行直方图（中间50%）')
    ax_row.set_xlabel('行号')
    ax_row.set_ylabel('像素和')
    ax_row.tick_params(axis='x', labelbottom=False)  # 隐藏x轴标签

    # 列直方图（右侧）
    ax_col = fig.add_subplot(gs[1, 1])
    ax_col.plot(vec_vals, color='green')
    ax_col.set_title('列直方图（中间90%）')
    ax_col.set_xlabel('列号')
    ax_col.set_ylabel('像素和')
    ax_col.tick_params(axis='y', labelleft=False)  # 隐藏y轴标签

    # 图像显示（中间）
    ax_image = fig.add_subplot(gs[1, 0])
    # 将图像绘制在 Matplotlib 中
    image_display = original_img.copy()
    for border in borders:
        cv2.rectangle(image_display, border[0], border[1], (0, 0, 255), 2)
    ax_image.imshow(cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB))
    ax_image.set_title('检测到的边界')
    ax_image.axis('off')  # 隐藏坐标轴

    # 调整布局
    plt.tight_layout()
    plt.show()

# 显示结果及边框（使用 Matplotlib 进行统一显示）
def showResults(original_img, borders):
    """
    在原始图像上绘制检测到的边框，并显示图像。

    参数：
        original_img (numpy.ndarray): 原始彩色图像。
        borders (list): 边框坐标列表。
    """
    # 将边框绘制在图像上
    image_with_borders = original_img.copy()
    for border in borders:
        cv2.rectangle(image_with_borders, border[0], border[1], (0, 0, 255), 2)
    
    # 显示图像（已在 visualizeHistogramAndImage 中处理）
    # 这里保留函数以便未来扩展
    pass

def main():
    path = '1111_PIL.png'  # 修改后的图像路径
    # 读取原始图像
    original_img = cv2.imread(path)
    if original_img is None:
        print("无法读取图像，请检查路径是否正确。")
        return
    # 转换为灰度图
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 滤波操作：使用高斯滤波减少噪点
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # 反相处理
    inverted_img = accessPixel(blurred_img)
    
    # 应用多种形态学操作
    morphed_img = applyMorphology(inverted_img)
    
    # 边缘检测
    borders = findBorderHistogram(morphed_img, row_ratio=0.5, col_ratio=0.9, min_vals=10, min_rect=20)
    
    # 可视化直方图和图像对齐显示
    visualizeHistogramAndImage(original_img, morphed_img, borders, row_ratio=0.5, col_ratio=0.9)

if __name__ == "__main__":
    main()
