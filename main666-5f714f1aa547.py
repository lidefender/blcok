# encoding:utf-8
import open3d as o3d
import numpy as np
import sys
from PIL import Image
# import cv2

# from numba import jit


# 归类平面用的指向向量，平面会被分为
top_vector = np.array([0, 0, 1])       # 顶面向量 - 指向上
left_vector = np.array([0, 1, 0])      # 左面向量 - 指向左
right_vector = np.array([1, 0, 0])     # 右面向量 - 指向右

# 归类平面用的指向向量，平面会被分为
top_vector = np.array([0, 0, 1])       # 顶面向量 - 指向上
left_vector =  np.array([1, 0, 0])    # 左面向量 - 指向左
right_vector =  np.array([0, 1, 0])     # 右面向量 - 指向右
surface_vectors = [top_vector, left_vector, right_vector]  # 存储表面向量列表，方便迭代

# 虚拟平面点集，预备变量
top_face = []
right_face = []
left_face = []
file_path = r'data\original\lmh02.pcd'  # 更新后的点云文件路径
# file_path = r'data\original\TMP001.pcd'


# @jit
def cut_points(pcd_filtered, plane_model,distance=0.05,threshold=30000,remove_above=False,debug=False):
    i= 0
    while True:
        points = planar_cut_off(pcd_filtered, plane_model,remove_above)
        # o3d.visualization.draw_geometries([points], window_name='points')
        if len(points.points) < threshold:
            break


        plane_model = translate_plane(plane_model, distance)

        # debug
        # A, B, C, D = plane_model
        # plane_temp=create_plane(A,B,C,D, size=30)
        # #o3d.visualization.draw_geometries([plane_temp,pcd_filtered], window_name='plane_temp')
        if debug:
            print(len(points.points))
            print(plane_model)
            i+=1
            print(i)
            if i % 5 == 0:
                A, B, C, D = plane_model
                plane_temp=create_plane(A,B,C,D, size=30)
                # o3d.visualization.draw_geometries([plane_temp,pcd_filtered], window_name='plane_temp')
                #o3d.visualization.draw_geometries([plane_temp, points], window_name='plane_temp')
    return points,plane_model


def translate_plane(plane_model, distance):
    """
    沿平面法向量方向平移平面。

    参数:
    a, b, c, d: 原平面方程 ax + by + cz + d = 0 的系数
    distance: 平移距离，正值表示沿法向量方向，负值表示相反方向

    返回:
    新平面方程的系数 (a, b, c, d_new)
    """
    a, b, c, d = plane_model
    normal_vector = np.array([a, b, c])
    norm = np.linalg.norm(normal_vector)  # 可以在函数外部计算并传入或者使用jit优化
    d_new = d - distance * norm
    plane_model = [a, b, c, d_new]
    return plane_model



def create_plane(a, b, c, d, size=10, grid_size=10):
    half_size = size / 2
    # 生成在[-half_size, half_size]范围内的网格，确保在原点附近生成
    xx, yy = np.meshgrid(np.linspace(-half_size, half_size, grid_size),
                         np.linspace(-half_size, half_size, grid_size))

    # 计算 z 值
    zz = (-a * xx - b * yy - d) / c

    # 合并顶点数据
    vertices = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

    # 创建三角形索引
    triangles = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # 创建两个三角形以形成一个矩形
            triangles.append([i * grid_size + j, (i + 1) * grid_size + j, i * grid_size + (j + 1)])
            triangles.append([(i + 1) * grid_size + j, (i + 1) * grid_size + (j + 1), i * grid_size + (j + 1)])
    triangles = np.array(triangles)

    # 创建平面网格
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    plane_mesh.compute_vertex_normals()
    plane_mesh.paint_uniform_color([0.1, 0.9, 0.1])  # 设置平面颜色
    return plane_mesh

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
    # Convert point cloud and distances to numpy arrays (if they aren't already)
    point_cloud = np.array(point_cloud)
    distances = np.array(distances)

    # Filter points where the distance is greater than the threshold
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
    [A, B, C] = plane_model[:3]
    D=plane_model[3]
    plane_model=[A,B,C]
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
        adjusted_normal = np.hstack([-normalized_plane,D])
    else:
        # 否则保持原方向
        adjusted_normal = np.hstack([normalized_plane,D])

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
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # Project each point in the point cloud onto the direction vector
    projections = np.dot(point_cloud, direction)
    span_max = np.max(projections)
    span_min = np.min(projections)

    # Calculate the span (max projection - min projection)
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
    # Step 1: Calculate the span and projections
    span, projections, span_max, span_min = calculate_span(point_cloud, direction)

    # Step 2: Define the centers of the intervals
    centers = span_min + [(n * span / 6) for n in range(1, 6)]

    # Step 3: Define intervals around each center
    interval_length = 3
    intervals = [(center - interval_length / 2, center + interval_length / 2) for center in centers]

    # Step 4: Classify points into intervals
    point_intervals = [[] for _ in range(len(intervals))]  # List to store points in each interval

    for i, proj in enumerate(projections):
        # Check which interval the projected value falls into
        for j, (start, end) in enumerate(intervals):
            if start <= proj <= end:
                point_intervals[j].append(point_cloud[i])
                break  # Once a point is assigned to an interval, stop checking further intervals

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
    max_distances=[]
    tmp_plane1 = origin # 被测表面1，只能是top_face left_face right_face
    tmp_plane2 = through # 测量方向表面
    plane1 = (tmp_plane1[0],tmp_plane1[1],tmp_plane1[2],tmp_plane1[3])
    plane2 = (tmp_plane2[0],tmp_plane2[1],tmp_plane2[2],tmp_plane2[3])
    # 使用“直线经过的点point + 直线的方向向量direction”表示一条直线
    point, direction = plane_intersection(plane1, plane2)
    #print("Point on the intersection line:", point)
    #print("Direction of the intersection line:", direction)

    # Step 4：计算“点集”中的点到直线的距离 - 点集是一个平面，直线是两平面交线。用于提取 离相交轴比较近的点集。
    point_cloud = desti.points # 被测表面2，只能是backleft_face_points backright_face_points bottom_face_points
    #使用“直线经过的点point + 直线的方向向量direction”表示一条直线
    # “直线经过的点point"。 A point on the line (from the previous example)
    line_point = point
    # "直线的方向向量direction”。Direction of the line (from the previous example)
    line_direction = direction
    # Calculate distances for each point in the point cloud
    distances = point_cloud_to_line_distances(point_cloud, line_point, line_direction) # 暂时无用
    # 调试用：显示各点至直线的距离
    #for i, dist in enumerate(distances):
    #    print(f"Distance from point {point_cloud[i]} to the line: {dist:.4f}")
    #points_in_cylinder = filter_points_by_distance(point_cloud, distances, threshold = 20) # 暂时无用
    # 调试用：显示圆柱体内的点云
    #tmp_point_cloud = o3d.geometry.PointCloud()
    #tmp_point_cloud.points = o3d.utility.Vector3dVector(points_in_cylinder)
    ##o3d.visualization.draw_geometries([tmp_point_cloud], window_name='Points in Cylinder')

    #调试用：计算点云在指定方向的span的示例程序。span是总跨度，projections是各点在指定方向上的投影，span_max span_min 是投影最大值和最小值
    #span, projections, span_max, span_min = calculate_span(points_in_cylinder, line_direction)
    #print(f"Span of the point cloud in the given direction: {span}")

    point_intervals, intervals = split_point_cloud_into_intervals(point_cloud, line_direction)
    #调试用： Output the points in each interval
    tmp_point_cloud = o3d.geometry.PointCloud()
    tmp_point_cloud_points = []
    for i, interval_points in enumerate(point_intervals):
        tmp_point_cloud_points.extend(interval_points)
        # tmp_point_cloud = o3d.geometry.PointCloud()
        # tmp_point_cloud.points = o3d.utility.Vector3dVector(interval_points)
        tmp_point_cloud.points = o3d.utility.Vector3dVector(tmp_point_cloud_points)
        o3d.visualization.draw_geometries([tmp_point_cloud,pcd_filtered_withRGB], window_name='Points in Interval')
        print(f"Interval {i+1} (centered around {intervals[i][0] + 3/2:.2f}):")
        print(np.array(interval_points))

    # Calculate distances
    for intervalA in point_intervals:
        distances = distances_to_plane(intervalA, tmp_plane1)
        print('最大距离：', np.max(distances))


# 直通滤波函数
def pass_through_filter(pcd, z_min=-1.7, y_max=150,x_max=200):
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
    condition = (points[:, 2] >= z_min) & (points[:, 1] <= y_max)&((points[:, 0] <= x_max))
    filtered_points = points[condition]
    filtered_colors = colors[condition]

    pcd_filtered_withRGB = o3d.geometry.PointCloud()
    pcd_filtered_withRGB.points = o3d.utility.Vector3dVector(filtered_points)  # Set filtered points (XYZ)
    pcd_filtered_withRGB.colors = o3d.utility.Vector3dVector(filtered_colors)  # Set corresponding filtered colors (RGB)

    # 创建新的点云对象
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)

    return pcd_filtered, pcd_filtered_withRGB

#
# def least_squares_plane_fit(points):
#     """
#     使用最小二乘法拟合平面。
#     点集应为 Nx3 的 numpy 数组。
#     返回平面参数 [a, b, c, d]，使得 ax + by + cz + d = 0
#     """
#     # 平面方程：ax + by + cz + d = 0
#     # 可以重写为 ax + by + cz = -d
#     # 构建设计矩阵 A 和目标向量 b
#     A = points[:, :3]
#     b = -np.ones((points.shape[0], 1))
#
#     # 添加常数项到 A
#     A = np.hstack((A, np.ones((points.shape[0], 1))))
#
#     # 最小二乘解
#     # 解决 Ax = b
#     x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
#     plane = x.flatten()
#
#     # 规范化平面参数，使得法向量为单位向量
#     norm = np.linalg.norm(plane[:3])
#     plane /= norm
#
#     return plane  # [a, b, c, d]

# # 内存不够要下采样
# def least_squares_plane_fit(pcd):
#     """
#     使用最小二乘法拟合平面。
#     点集应为 Nx3 的 numpy 数组。
#     返回平面参数 [a, b, c, d]，使得 ax + by + cz + d = 0
#     """
#     # 平面方程：ax + by + cz + d = 0
#     # 可以重写为 ax + by + cz = -d
#     # -----------------使用SVD分解进行最小二乘拟合平面-----------------------
#     center = pcd.get_center()  # 计算点云质心
#     points = np.asarray(pcd.points)  # 获取点坐标
#     H = points - center  # 去质心
#     U, S, V = np.linalg.svd(H)  # 矩阵奇异值分解
#
#     # 点云最小特征值对应的特征向量即为拟合平面的系数A,B,C
#     A = V[2, 0]
#     B = V[2, 1]
#     C = V[2, 2]
#     D = -(A * center[0] + B * center[1] + C * center[2])
#
#     plane = [A, B, C, D]
#     return plane  # [a, b, c, d]


def least_squares_plane_fit(pcd):
    """
    使用pca最小二乘法拟合平面。
    点集应为 Nx3 的 numpy 数组。
    返回平面参数 [a, b, c, d]，使得 ax + by + cz + d = 0
    """
    # 平面方程：ax + by + cz + d = 0
    # 可以重写为 ax + by + cz = -d
    [center, covariance] = pcd.compute_mean_and_covariance()
    U, S, _ = np.linalg.svd(covariance)
    # 点云最小特征值对应的特征向量即为拟合平面的系数A,B,C
    A = U[0, 2]
    B = U[1, 2]
    C = U[2, 2]
    D = -(A * center[0] + B * center[1] + C * center[2])

    plane = [A, B, C, D]
    return plane  # [a, b, c, d]


# 默认流程：读取点云
pcd = o3d.io.read_point_cloud(file_path)  # 读取点云文件
# o3d.visualization.draw_geometries([pcd])
# if pcd.has_colors():
#     print("Point cloud has colors.")
#     # o3d.visualization.draw_geometries([pcd])
# else:
#     print("Point cloud does not have colors.")

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)  # Extract RGB colors

# Module4: 迭代拟合平面测试
if True:
    # Step 1：直通滤波，仅保留 z >= -7 且 y <= 180
    if True:
        pcd_filtered, pcd_filtered_withRGB = pass_through_filter(pcd, z_min=-1.8, y_max=150)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=100, origin=[0, 0, 0]  # 调整 size 改变坐标轴大小
        )
        o3d.visualization.draw_geometries([pcd_filtered_withRGB,coordinate], window_name='Filtered Body')  # 显示过滤后的点云

    # Step 2：拟合平面（准备检测立方体的角点）
    # Step 2：拟合平面（准备检测立方体的角点）
    if True:
        # 原始点集 pcd_filtered，剩余点集 pcd_rest，分出去的点集 plane_model1_points；
        # 拟合第一个平面
        # Segments a plane in the point cloud using the RANSAC algorithm.
        plane_model1, inliers1 = pcd_filtered.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)

        print('Face-1 equation:', plane_model1)
        print("平面向量的模长为:", np.linalg.norm(plane_model1[:3]),"已单位化")
        # 排除拟合到平面1的点
        pcd_rest = pcd_filtered.select_by_index(inliers1, invert=True)
        plane_model1_points = pcd_filtered.select_by_index(inliers1)
        plane_model1 = least_squares_plane_fit(plane_model1_points)
        print('Face-1 equation2:', plane_model1)
        # #o3d.visualization.draw_geometries([pcd_rest], window_name='Face-1 Fitting')
        # 分类平面
        tmp_plane = plane_model1 # 仅取法向量部分
        tmp, tmp_plane = classify_plane(tmp_plane)

        tmp_plane_points = plane_model1_points

        if tmp == 0 and top_face == []:
            top_face = tmp_plane
            top_face_points = tmp_plane_points
            tmp_plane_points, tmp_plane = cut_points(pcd_filtered, tmp_plane, distance=0.01, threshold=5000, #往上移动并把下面的点删掉
                                                     remove_above=False,debug=True)
        elif tmp == 1 and left_face == []:
            tmp_plane_points, tmp_plane = cut_points(plane_model1_points, tmp_plane, distance=-0.01, threshold=10000,
                                                     remove_above=True)
            left_face = tmp_plane
            backleft_face_points = tmp_plane_points
        elif tmp == 2 and right_face == []:
            tmp_plane_points, tmp_plane = cut_points(plane_model1_points, tmp_plane, distance=-0.01, threshold=10000,
                                                     remove_above=True)
            right_face = tmp_plane
            right_face_points = tmp_plane_points
        else:
            print("Classification Error")
            sys.exit()

        # 拟合第二个平面
        plane_model2, inliers2 = pcd_rest.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
        print('Face-2 equation:', plane_model2)
        pcd_rest2 = pcd_rest.select_by_index(inliers2, invert=True)
        plane_model2_points = pcd_rest.select_by_index(inliers2)
        plane_model2 = least_squares_plane_fit(plane_model2_points)
        # #o3d.visualization.draw_geometries([pcd_rest2], window_name='Face-2 Fitting')
        # 分类平面
        tmp_plane = plane_model2  # 仅取法向量部分
        tmp, tmp_plane = classify_plane(tmp_plane)

        tmp_plane_points = plane_model2_points # TODO 改为函数

        if tmp == 0 and top_face == []:
            top_face = tmp_plane
            top_face_points = tmp_plane_points
            tmp_plane_points, tmp_plane = cut_points(pcd_filtered, tmp_plane, distance=0.01, threshold=5000, #往上移动并把下面的点删掉
                                                     remove_above=False,debug=True)
        elif tmp == 1 and left_face == []:
            tmp_plane_points, tmp_plane = cut_points(plane_model2_points, tmp_plane, distance=-0.01, threshold=10000,
                                                     remove_above=True)
            left_face = tmp_plane
            backleft_face_points = tmp_plane_points
        elif tmp == 2 and right_face == []:
            tmp_plane_points, tmp_plane = cut_points(plane_model2_points, tmp_plane, distance=-0.01, threshold=10000,
                                                     remove_above=True)
            right_face = tmp_plane
            right_face_points = tmp_plane_points
        else:
            print("Classification Error")
            sys.exit()

        # 拟合第三个平面
        plane_model3, inliers3 = pcd_rest2.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
        print('Face-3 equation:', plane_model3)
        pcd_rest3 = pcd_rest2.select_by_index(inliers3, invert=True)
        plane_model3_points = pcd_rest2.select_by_index(inliers3)
        plane_model3 = least_squares_plane_fit(plane_model3_points)
        # #o3d.visualization.draw_geometries([pcd_rest3], window_name='Face-3 Fitting')
        # 分类平面
        tmp_plane = plane_model3  # 仅取法向量部分
        tmp, tmp_plane = classify_plane(tmp_plane)

        tmp_plane_points = plane_model3_points

        if tmp == 0 and top_face == []:
            top_face = tmp_plane
            top_face_points = tmp_plane_points
            tmp_plane_points, tmp_plane = cut_points(pcd_filtered, tmp_plane, distance=0.01, threshold=5000, #往上移动并把下面的点删掉
                                                     remove_above=False,debug=True)
        elif tmp == 1 and left_face == []:
            tmp_plane_points, tmp_plane = cut_points(plane_model3_points, tmp_plane, distance=-0.01, threshold=10000,
                                                     remove_above=True)
            left_face = tmp_plane
            backleft_face_points = tmp_plane_points
        elif tmp == 2 and right_face == []:
            tmp_plane_points, tmp_plane = cut_points(plane_model3_points, tmp_plane, distance=-0.01, threshold=10000,
                                                     remove_above=True)
            right_face = tmp_plane
            right_face_points = tmp_plane_points
        else:
            print("Classification Error")
            sys.exit()
        print('Top Face equation:', top_face)
        print('Left Face equation:', left_face)
        print('Right Face equation:', right_face)
        # 当前总结：
        # 得到3个集总参数的虚拟平面：top_face、left_face、right_face - 平面模型ABCD；及对应点集对象top_face_points、left_face_points...
        # 切除多余点后的 总有效点集：pcd_filtered - 点集对象
        # 暂存的原始输出平面及其点集：（以下这3个平面分别对应了top_face、left_face、right_face）
        #   plane_model1 - 平面1模型ABCD, inliers1 - 索引（无用），plane_model1_points - 点集对象；
        #   plane_model2 - 平面2模型ABCD, inliers2 - 索引（无用），plane_model2_points - 点集对象；
        #   plane_model3 - 平面3模型ABCD, inliers3 - 索引（无用），plane_model3_points - 点集对象；
        # 剩余点集：pcd_rest3 - 点集对象

        # 调试用：可视化拟合出的3个平面所包含的点集
        # #o3d.visualization.draw_geometries([plane_model1_points])
        # #o3d.visualization.draw_geometries([plane_model2_points])
        # #o3d.visualization.draw_geometries([plane_model3_points])

    # Step 2.5：生成用于测量距离的背面平面点集
    if True:
        # 左后
        A, B, C, D = right_face # 由于左后面拍摄不到，所以沿着左侧面的法向量方向，切割出左后面点集
        plane_model = [A, B, C, D-10] # TODO

        plane_model = [A, B, C, D]
        plane_temp1=create_plane(A,B,C,D, size=15)
        plane_model_temp2=translate_plane(plane_model, 10)
        plane_temp2=create_plane(plane_model_temp2[0],plane_model_temp2[1],plane_model_temp2[2],plane_model_temp2[3], size=15)
        #o3d.visualization.draw_geometries([plane_temp1,plane_temp2,pcd_filtered], window_name='bottom_face_points plane_temp01')

        backleft_face_points,plane_model = cut_points(pcd_filtered, plane_model, distance=1, threshold=30000, remove_above=False)
        backleft_face_points.paint_uniform_color([0, 0, 0])  # 设置平面颜色
        
        
        # 调试用：显示点云
        # 选择并过滤点云
        pcd_filtered_points = np.asarray(pcd_filtered.points)
        # 找出不在 left_face_points 中的点
        mask = np.all(np.isin(pcd_filtered_points, backleft_face_points), axis=1)
        filtered_points = pcd_filtered_points[~mask]
        # 创建过滤后的点云
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.paint_uniform_color([1,1,1])
        # 可视化过滤后的点云和左面点
        #o3d.visualization.draw_geometries([filtered_pcd, backleft_face_points], window_name='Left Face Points')
        # #o3d.visualization.draw_geometries([pcd_filtered.select_by_index(np.where(pcd_filtered==left_face_points),invert=True),left_face_points], window_name='left_face_points')



        # 右后
        A, B, C, D = left_face # 由于左后面拍摄不到，所以沿着左侧面的法向量方向，切割出左后面点集

        plane_model = [A, B, C, D]
        plane_temp1=create_plane(A,B,C,D, size=15)
        plane_model_temp2=translate_plane(plane_model, 10)
        plane_temp2=create_plane(plane_model_temp2[0],plane_model_temp2[1],plane_model_temp2[2],plane_model_temp2[3], size=15)
        #o3d.visualization.draw_geometries([plane_temp1,plane_temp2,pcd_filtered], window_name='bottom_face_points plane_temp')

        backright_face_points,plane_model = cut_points(pcd_filtered, plane_model, distance=1,threshold=30000, remove_above=False)

        # 调试用：显示点云
        #o3d.visualization.draw_geometries([backright_face_points], window_name='backright_face_points')

        # right_face_points = planar_cut_off(pcd_filtered, left_face, True)
        # 调试用：显示点云
        # #o3d.visualization.draw_geometries([right_face_points], window_name='right_face_points')

        # 底边
        # 由于底面拍摄不到，所以沿着顶面的法向量方向，切割出底边点集
        # 顶面法向量：top_face，将顶面平面沿着其反向移动直至处在平面下面的点小于阈值N=30000



        A, B, C, D = top_face # 底面直接使用z=5平面
        plane_model = [A, B, C, D]
        plane_temp1=create_plane(A,B,C,D, size=15)
        plane_model_temp2=translate_plane(plane_model, -5)
        plane_temp2=create_plane(plane_model_temp2[0],plane_model_temp2[1],plane_model_temp2[2],plane_model_temp2[3], size=15)
        #o3d.visualization.draw_geometries([plane_temp1,plane_temp2,pcd_filtered,coordinate], window_name='bottom_face_points plane_temp333333333333')
        bottom_face_points,plane_model = cut_points(pcd_filtered, plane_model,distance=-1, threshold=50000, remove_above=True,debug=True)

        # 调试用：显示点云
        #o3d.visualization.draw_geometries([bottom_face_points], window_name='bottom_face_points')

    # Step 3：计算两平面交线，测量距离

    # 函数调用实验
    print("顶面与底面的距离，沿左侧面测量：")
    cal_five_dis(top_face, left_face, bottom_face_points)

    print("顶面与底面的距离，沿右侧面测量：")
    cal_five_dis(top_face, right_face, bottom_face_points)

    print("左侧面与右后面的距离，沿顶面测量：")
    cal_five_dis(left_face, top_face, backright_face_points)


# Module5:OCR识别
if True:
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd_filtered_withRGB)
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()

    imagex = np.asarray(image) * 255
    imagex = imagex.astype(np.uint8)
    pilImage = Image.fromarray(imagex)
    pilImage.save(r'1111_PIL.png')

