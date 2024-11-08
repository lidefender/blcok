import os
import numpy as np
import cv2

# 反相灰度图，将黑白阈值颠倒


def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img

# 反相二值化图像
def accessBinary(img, threshold=128):
    img = accessPiexl(img)

    cv2.imshow('accessPiexl', img)
    cv2.waitKey(0)

    # 边缘膨胀，不加也可以
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


# 根据长向量找出顶点
def extractPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None

    # 剔除一些噪点
    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints

# 寻找边缘，返回边框的左上角和右下角（利用直方图寻找边缘算法（需行对齐））
def findBorderHistogram(img):
    borders = []

    
    # 行扫描
    hori_vals = np.sum(img, axis=1)
    hori_points = extractPeek(hori_vals)
    # 根据每一行来扫描列
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            borders.append(border)
    return borders
    
# 显示结果及边框
def showResults(img, borders, results=None):

    # 绘制
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        #cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    cv2.imshow('test', img)
    cv2.waitKey(0)



def crop_center(img, n1, n2):
    """
    截取图像的行的中间n1%和列的中间n2%.
    
    参数:
        img (ndarray): 原始图像.
        n1 (float): 行方向截取的百分比(0 < n1 <= 100).
        n2 (float): 列方向截取的百分比(0 < n2 <= 100).
        
    返回:
        ndarray: 截取后的图像.
    """
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 计算截取区域的尺寸
    new_height = int(height * (n1 / 100))
    new_width = int(width * (n2 / 100))
    
    # 计算中间区域的起始和结束坐标
    start_y = (height - new_height) // 2
    start_x = (width - new_width) // 2
    end_y = start_y + new_height
    end_x = start_x + new_width
    
    # 截取图像
    cropped_img = img[start_y:end_y, start_x:end_x]
    return cropped_img

if __name__ == '__main__':

    # 读取图像
    img_path = '1111_PIL.png'
    img = cv2.imread(img_path,0)
    if img is None:
        print("图像读取失败，请检查路径！")
    else:
        # 截取图像的中间部分
        cropped_img = crop_center(img, 50, 50) # 截取中间50%的区域
        
        # 创建保存文件夹
        save_folder = "data/img"
        os.makedirs(save_folder, exist_ok=True)
        
        # 保存截取后的图像
        save_path = os.path.join(save_folder, "cropped_output.jpg")
        cv2.imwrite(save_path, cropped_img)
        print(f"图像已保存为 {save_path}")
    img = cropped_img
    img = accessBinary(img)
    cv2.namedWindow('accessBinary', cv2.WINDOW_NORMAL)
    cv2.imshow('accessBinary', img)
    cv2.waitKey(0) 

    borders = findBorderHistogram(img)
    showResults(img, borders)
