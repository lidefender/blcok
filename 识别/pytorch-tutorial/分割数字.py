import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'C:\work\team\blcok\1111_PIL.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊，去除噪点
blur = cv2.GaussianBlur(gray, (25, 25), 0)

# 自适应阈值二值化
binary = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# 寻找轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤小轮廓，假设字符的最小面积为100
min_area = 100
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# 存储字符位置
char_positions = []

for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    char_positions.append((x, y, w, h))

# 复制原图以进行标记
marked_image = image.copy()

for pos in char_positions:
    x, y, w, h = pos
    # 绘制矩形边界框，颜色为绿色，线条粗细为2
    cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary)
cv2.imshow('Marked Image', marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 可选：保存结果图像
cv2.imwrite('marked_handwritten.jpg', marked_image)

# 打印字符位置信息
for idx, pos in enumerate(char_positions):
    print(f"Character {idx + 1}: x={pos[0]}, y={pos[1]}, w={pos[2]}, h={pos[3]}")
