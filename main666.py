# encoding:utf-8
import open3d as o3d
import numpy as np
import sys

# 归类平面用的指向向量，平面会被分为
top_vector = np.array([0, 0, 1])       # 顶面向量 - 指向上
left_vector = np.array([0, 1, 0])      # 左面向量 - 指向左
right_vector = np.array([1, 0, 0])     # 右面向量 - 指向右
surface_vectors = [top_vector, left_vector, right_vector]  # 存储表面向量列表，方便迭代

# 虚拟平面点集，预备变量
top_face = []
right_face = []
left_face = []
file_path = r'F:\work\python\team\blcok\data\original\aaa.pcd'  # 更新后的点云文件路径

# 用平面分割点云，排除不需要部分
def planar_cut_off(pcd, plane_model, remove_above=True):
    """
    对点云进行平面切割，去除不需要的部分。

    参数:
        pcd (open3d.geometry.PointCloud): 输入点云。
        plane_model (list): 平面模型参数 [a, b, c, d]，表示平面方程 ax + by + cz + d = 0。
        remove_above (bool): 如果为True，移除平面上方的点；否则移除下方的点。

    返回:
        open3d.geometry.PointCloud: 切割后的点云。
    """
    points = np.asarray(pcd.points)
    distances = np.dot(points, plane_model[:3]) + plane_model[3]
    if remove_above:
        indices = np.where(distances < 0)[0]
    else:
        indices = np.where(distances > 0)[0]
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(points[indices])
    return pcd_filtered

# 通过三个指定点计算平面方程
def plane_from_points(p1, p2, p3):
    """
    通过三个点计算平面方程 Ax + By + Cz + D = 0。

    参数:
        p1, p2, p3 (tuple/list): 三个点的坐标。

    返回:
        tuple: 平面方程的系数 (A, B, C, D)。
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    v1 = p2 - p1
    v2 = p3 - p1
    normal_vector = np.cross(v1, v2)
    A, B, C = normal_vector
    D = -np.dot(normal_vector, p1)
    return A, B, C, D

# 计算两平面交线
def plane_intersection(plane1, plane2):
    """
    计算两平面的交线。

    参数:
        plane1, plane2 (tuple): 两个平面方程的系数 (A, B, C, D)。

    返回:
        tuple: 交线上一个点的坐标和方向向量。
    """
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    normal1 = np.array([a1, b1, c1])
    normal2 = np.array([a2, b2, c2])
    direction = np.cross(normal1, normal2)
    if np.all(direction == 0):
        raise ValueError("两平面平行或重合，没有唯一的交线。")
    A = np.array([[a1, b1],
                  [a2, b2]])
    B = np.array([-d1, -d2])
    if np.linalg.det(A) != 0:
        point_xy = np.linalg.solve(A, B)
        point = np.array([point_xy[0], point_xy[1], 0])
    else:
        A = np.array([[b1, c1],
                      [b2, c2]])
        B = np.array([-d1, -d2])
        point_yz = np.linalg.solve(A, B)
        point = np.array([0, point_yz[0], point_yz[1]])
    return point, direction

# 计算点云到直线的距离
def point_cloud_to_line_distances(point_cloud, line_point, line_direction):
    """
    计算点云中每个点到指定直线的距离。

    参数:
        point_cloud (array-like): 点云数据，形状为 (n, 3)。
        line_point (array-like): 直线上的一点。
        line_direction (array-like): 直线的方向向量。

    返回:
        array: 每个点到直线的距离数组。
    """
    P = np.array(line_point)
    d = np.array(line_direction)
    d = d / np.linalg.norm(d)
    point_cloud = np.array(point_cloud)
    P0_P = point_cloud - P
    cross_prods = np.cross(d, P0_P)
    numerators = np.linalg.norm(cross_prods, axis=1)
    denominator = np.linalg.norm(d)
    distances = numerators / denominator
    return distances

# 根据距离筛选点集
def filter_points_by_distance(point_cloud, distances, threshold=2):
    """
    筛选距离直线小于特定阈值的点。

    参数:
        point_cloud (array-like): 点云数据，形状为 (n, 3)。
        distances (array-like): 每个点到直线的距离。
        threshold (float): 距离阈值。

    返回:
        array: 符合条件的点集。
    """
    point_cloud = np.array(point_cloud)
    distances = np.array(distances)
    filtered_points = point_cloud[distances <= threshold]
    return filtered_points

# 分类平面
def classify_plane(plane_model):
    """
    根据平面法向量将平面分类为顶面、左面或右面，并调整法向量方向与预定义的surface_vectors同向。

    参数:
        plane_model (array-like): 平面法向量 [A, B, C]。

    返回:
        tuple: (分类结果的索引（0: 顶面, 1: 左面, 2: 右面）, 调整后的法向量 [A, B, C])。
    """
    # 归一化法向量
    normalized_plane = plane_model / np.linalg.norm(plane_model)

    # 计算与预定义表面向量的点积
    dot_products = [np.dot(normalized_plane, surface_vector) for surface_vector in surface_vectors]

    # 取绝对值最大的点积对应的索引
    max_index = np.argmax(np.abs(dot_products))  # 使用绝对值忽略方向

    # 获取对应的surface_vector
    corresponding_surface_vector = surface_vectors[max_index]

    # 检查法向量是否与surface_vector同向
    if np.dot(normalized_plane, corresponding_surface_vector) < 0:
        # 如果点积为负，则反向法向量
        adjusted_normal = -normalized_plane
    else:
        # 否则保持原方向
        adjusted_normal = normalized_plane

    return max_index, adjusted_normal


# 计算点云在某方向上的跨度
def calculate_span(point_cloud, direction):
    """
    计算点云在指定方向上的跨度。

    参数:
        point_cloud (array-like): 点云数据，形状为 (n, 3)。
        direction (array-like): 指定方向的向量。

    返回:
        tuple: (跨度, 投影值, 最大投影, 最小投影)
    """
    direction = direction / np.linalg.norm(direction)
    projections = np.dot(point_cloud, direction)
    span_max = np.max(projections)
    span_min = np.min(projections)
    span = span_max - span_min
    return span, projections, span_max, span_min

# 将点云按跨度分割为多个区间
def split_point_cloud_into_intervals(point_cloud, direction):
    """
    根据跨度将点云分割成多个区间。

    参数:
        point_cloud (array-like): 点云数据，形状为 (n, 3)。
        direction (array-like): 指定方向的向量。

    返回:
        tuple: (分割后的点集列表, 区间范围列表)
    """
    span, projections, span_max, span_min = calculate_span(point_cloud, direction)
    centers = span_min + [(n * span / 6) for n in range(1, 6)]
    interval_length = 3
    intervals = [(center - interval_length / 2, center + interval_length / 2) for center in centers]
    point_intervals = [[] for _ in range(len(intervals))]
    for i, proj in enumerate(projections):
        for j, (start, end) in enumerate(intervals):
            if start <= proj <= end:
                point_intervals[j].append(point_cloud[i])
                break
    return point_intervals, intervals

# 计算点云到平面的距离
def distances_to_plane(point_cloud, plane_model):
    """
    计算点云中每个点到指定平面的距离。

    参数:
        point_cloud (array-like): 点云数据，形状为 (n, 3)。
        plane_model (list): 平面模型参数 [A, B, C, D]。

    返回:
        array: 每个点到平面的距离数组。
    """
    A, B, C, D = plane_model
    distances = []
    for point in point_cloud:
        x, y, z = point
        distance = abs(A * x + B * y + C * z + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)
        distances.append(distance)
    return np.array(distances)

# 计算五个位置的最大距离
def cal_five_dis(origin, through, desti):
    """
    计算两个平面交线上的点到目标点云的最大距离。

    参数:
        origin (list): 被测表面平面模型 [A, B, C, D]。
        through (list): 测量方向表面平面模型 [A, B, C, D]。
        desti (open3d.geometry.PointCloud): 目标点云。

    返回:
        None
    """
    tmp_plane1 = origin  # 被测表面1
    tmp_plane2 = through  # 测量方向表面
    plane1 = (tmp_plane1[0], tmp_plane1[1], tmp_plane1[2], tmp_plane1[3])
    plane2 = (tmp_plane2[0], tmp_plane2[1], tmp_plane2[2], tmp_plane2[3])
    point, direction = plane_intersection(plane1, plane2)
    point_cloud = desti.points  # 目标点云
    line_point = point
    line_direction = direction
    distances = point_cloud_to_line_distances(point_cloud, line_point, line_direction)
    point_intervals, intervals = split_point_cloud_into_intervals(point_cloud, line_direction)
    for intervalA in point_intervals:
        distances = distances_to_plane(intervalA, tmp_plane1)
        print('最大距离：', np.max(distances))

# 直通滤波函数
def pass_through_filter(pcd, z_min=-7, y_max=180):
    """
    使用直通滤波，仅保留 z >= z_min 且 y <= y_max 的点。

    参数:
        pcd (open3d.geometry.PointCloud): 输入点云。
        z_min (float): z 坐标的最小值。
        y_max (float): y 坐标的最大值。

    返回:
        open3d.geometry.PointCloud: 经过直通滤波后的点云。
    """
    points = np.asarray(pcd.points)
    # 过滤条件：z >= z_min 且 y <= y_max
    condition = (points[:, 2] >= z_min) & (points[:, 1] <= y_max)
    filtered_points = points[condition]
    # 创建新的点云对象
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
    return pcd_filtered

# 默认流程：读取点云
pcd = o3d.io.read_point_cloud(file_path)  # 读取点云文件
points = np.asarray(pcd.points)

# Module4: 迭代拟合平面测试
if True:
    # Step 1：直通滤波，仅保留 z >= -7 且 y <= 180
    if True:
        pcd_filtered = pass_through_filter(pcd, z_min=-7, y_max=180)
        o3d.visualization.draw_geometries([pcd_filtered], window_name='Filtered Body')  # 显示过滤后的点云

    # Step 2：拟合平面（准备检测立方体的角点）
    # Step 2：拟合平面（准备检测立方体的角点）
    if True:
        # 拟合第一个平面
        plane_model1, inliers1 = pcd_filtered.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
        print('Face-1 equation:', plane_model1)
        # 排除拟合到平面1的点
        pcd_rest = pcd_filtered.select_by_index(inliers1, invert=True)
        plane_model1_points = pcd_filtered.select_by_index(inliers1)
        o3d.visualization.draw_geometries([pcd_rest], window_name='Face-1 Fitting')
        # 分类平面
        tmp_plane = plane_model1[:3]  # 仅取法向量部分
        classification_index, adjusted_normal = classify_plane(tmp_plane)
        if classification_index == 0 and not top_face:
            top_face = [*adjusted_normal, plane_model1[3]]  # 保留原来的D值
            top_face_points = plane_model1_points
        elif classification_index == 1 and not left_face:
            left_face = [*adjusted_normal, plane_model1[3]]
            left_face_points = plane_model1_points
        elif classification_index == 2 and not right_face:
            right_face = [*adjusted_normal, plane_model1[3]]
            right_face_points = plane_model1_points
        else:
            print("Classification Error")
            sys.exit()

        # 拟合第二个平面
        plane_model2, inliers2 = pcd_rest.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
        print('Face-2 equation:', plane_model2)
        pcd_rest2 = pcd_rest.select_by_index(inliers2, invert=True)
        plane_model2_points = pcd_rest.select_by_index(inliers2)
        o3d.visualization.draw_geometries([pcd_rest2], window_name='Face-2 Fitting')
        # 分类平面
        tmp_plane = plane_model2[:3]
        classification_index, adjusted_normal = classify_plane(tmp_plane)
        if classification_index == 0 and not top_face:
            top_face = [*adjusted_normal, plane_model2[3]]
            top_face_points = plane_model2_points
        elif classification_index == 1 and not left_face:
            left_face = [*adjusted_normal, plane_model2[3]]
            left_face_points = plane_model2_points
        elif classification_index == 2 and not right_face:
            right_face = [*adjusted_normal, plane_model2[3]]
            right_face_points = plane_model2_points
        else:
            print("Classification Error")
            sys.exit()

        # 拟合第三个平面
        plane_model3, inliers3 = pcd_rest2.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
        print('Face-3 equation:', plane_model3)
        pcd_rest3 = pcd_rest2.select_by_index(inliers3, invert=True)
        plane_model3_points = pcd_rest2.select_by_index(inliers3)
        o3d.visualization.draw_geometries([pcd_rest3], window_name='Face-3 Fitting')
        # 分类平面
        tmp_plane = plane_model3[:3]
        classification_index, adjusted_normal = classify_plane(tmp_plane)
        if classification_index == 0 and not top_face:
            top_face = [*adjusted_normal, plane_model3[3]]
            top_face_points = plane_model3_points
        elif classification_index == 1 and not left_face:
            left_face = [*adjusted_normal, plane_model3[3]]
            left_face_points = plane_model3_points
        elif classification_index == 2 and not right_face:
            right_face = [*adjusted_normal, plane_model3[3]]
            right_face_points = plane_model3_points
        else:
            print("Classification Error")
            sys.exit()

        print('Top Face equation:', top_face)
        print('Left Face equation:', left_face)
        print('Right Face equation:', right_face)
    # Step 2.5：生成用于测量距离的背面平面点集
    if True:
        # 左后三个点及裕量
        A, B, C, D = plane_model1
        plane_model = [A, B, C, D]
        left_face_points = planar_cut_off(pcd_filtered, plane_model, False)
        # 调试用：显示点云
        o3d.visualization.draw_geometries([left_face_points], window_name='left_face_points')

        # 右后
        A, B, C, D = plane_model2
        plane_model = [A, B, C, D]
        right_face_points = planar_cut_off(pcd_filtered, plane_model, True)
        # 调试用：显示点云
        o3d.visualization.draw_geometries([right_face_points], window_name='right_face_points')

        # 底面

        A, B, C, D = plane_model3
        plane_model = [A, B, C, D]
        bottom_face_points = planar_cut_off(pcd_filtered, plane_model, True)
        # 调试用：显示点云
        o3d.visualization.draw_geometries([bottom_face_points], window_name='bottom_face_points')

    # Step 3：计算两平面交线，测量距离
    print("顶面与底面的距离，沿左侧面测量：")
    # o3d.visualization.draw_geometries([top_face,left_face,right_face], window_name='Face-3 Fitting')
    cal_five_dis(top_face, left_face, bottom_face_points)

    print("顶面与底面的距离，沿右侧面测量：")
    cal_five_dis(top_face, right_face, bottom_face_points)

    print("左侧面与右后面的距离，沿顶面测量：")
    cal_five_dis(left_face, top_face, right_face_points)
