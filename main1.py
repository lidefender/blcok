# encoding:utf-8
import open3d as o3d
import numpy as np
import sys

# 归类平面用的指向向量，平面会被分为
top_vector = np.array([0, 0, 1])       # 顶面向量 - 指向上
left_vector = np.array([-1, 0, 0])     # 左面向量 - 指向左
right_vector = np.array([1, 0, 0])     # 右面向量 - 指向右
surface_vectors = [top_vector, left_vector, right_vector]  # 存储表面向量列表，方便迭代

# 存储分类后的平面模型和点云
classified_planes = {
    'top': {'model': None, 'points': None},
    'left': {'model': None, 'points': None},
    'right': {'model': None, 'points': None}
}

file_path = r'F:\work\python\team\blcok\data\original\aaa.pcd'  # 点云文件路径


# 用平面分割点云 排除不需要部分
def planar_cut_off(pcd, plane_model, remove_above=True):
    """
    Performs a planar cut-off on a point cloud.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        plane_model (list): The plane model parameters [A, B, C, D] representing the equation ax + by + cz + d = 0.
        remove_above (bool): If True, removes points above the plane, otherwise removes points below.

    Returns:
        open3d.geometry.PointCloud: The point cloud after the planar cut-off.
    """

    points = np.asarray(pcd.points)

    # Calculate the signed distance from each point to the plane
    distances = np.dot(points, plane_model[:3]) + plane_model[3]

    # Filter points based on the desired side of the plane
    if remove_above:
        indices = np.where(distances < 0)[0]
    else:
        indices = np.where(distances > 0)[0]

    # Create a new point cloud with the filtered points
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(points[indices])

    return pcd_filtered


# 计算两平面交线
def plane_intersection(plane1, plane2):
    """
    计算两平面的交线。

    Args:
        plane1 (list or tuple): 平面1的模型参数 [A, B, C, D]。
        plane2 (list or tuple): 平面2的模型参数 [A, B, C, D]。

    Returns:
        tuple: (point, direction) 交线上的一点和方向向量。
    """
    # Unpack the plane definitions (a1, b1, c1, d1) and (a2, b2, c2, d2)
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2

    # Create normal vectors for each plane
    normal1 = np.array([a1, b1, c1])
    normal2 = np.array([a2, b2, c2])

    # Step 1: Find the direction of the intersection line (cross product of normals)
    direction = np.cross(normal1, normal2)

    # If the cross product is zero, the planes are parallel or identical
    if np.all(direction == 0):
        raise ValueError("两个平面平行或重合，没有唯一的交线。")

    # Step 2: To find a point on the line, solve the system of equations
    # We can use a simple approach by fixing one coordinate (e.g., z = 0)
    A = np.array([[a1, b1],
                  [a2, b2]])
    B = np.array([-d1, -d2])

    if np.linalg.det(A) != 0:
        # Solve for x and y with z = 0
        point_xy = np.linalg.solve(A, B)
        point = np.array([point_xy[0], point_xy[1], 0])
    else:
        # If the determinant is 0, fix x = 0 and solve for y and z
        A = np.array([[b1, c1],
                      [b2, c2]])
        B = np.array([-d1, -d2])
        point_yz = np.linalg.solve(A, B)
        point = np.array([0, point_yz[0], point_yz[1]])

    return point, direction


# 计算点云到一条直线的距离
def point_cloud_to_line_distances(point_cloud, line_point, line_direction):
    """
    Calculate the distances from a point cloud to a line in 3D.

    Parameters:
    point_cloud (array-like): A list or array of points in 3D space, shape (n, 3).
    line_point (array-like): A point on the line P (x1, y1, z1).
    line_direction (array-like): The direction vector of the line (dx, dy, dz).

    Returns:
    array: A 1D array of distances for each point in the point cloud.
    """
    # Convert inputs to numpy arrays
    P = np.array(line_point)
    d = np.array(line_direction)

    # Ensure the direction vector is normalized
    d = d / np.linalg.norm(d)

    # Convert the point cloud to a numpy array (n, 3)
    point_cloud = np.array(point_cloud)

    # Step 1: Compute the vector from the line point to each point in the point cloud
    P0_P = point_cloud - P  # Shape (n, 3)

    # Step 2: Calculate the cross product between the direction vector and P0_P for each point
    cross_prods = np.cross(d, P0_P)  # Shape (n, 3)

    # Step 3: Calculate the magnitudes of the cross products (distances to the line)
    distances = np.linalg.norm(cross_prods, axis=1)  # Already divided by |d|

    return distances


# 根据距离筛选点
def filter_points_by_distance(point_cloud, distances, threshold=2):
    """
    Filter points whose distance from the line is less than or equal to a given threshold.

    Parameters:
    point_cloud (array-like): A list or array of points in 3D space, shape (n, 3).
    distances (array-like): A list or array of distances corresponding to each point.
    threshold (float): The distance threshold for filtering points.

    Returns:
    list: A list of points whose distance is less than or equal to the threshold.
    """
    # Convert point cloud and distances to numpy arrays (if they aren't already)
    point_cloud = np.array(point_cloud)
    distances = np.array(distances)

    # Filter points where the distance is less than or equal to the threshold
    filtered_points = point_cloud[distances <= threshold]

    return filtered_points


# 分类平面
def classify_plane(plane_model):
    """
    根据平面的法向量与预定义的表面向量的夹角，分类平面为top、left、right。

    Args:
        plane_model (list or tuple): 平面模型参数 [A, B, C, D]。

    Returns:
        int: 分类结果，0: top, 1: left, 2: right。
    """
    # 正规化平面法向量
    plane_normal = np.array(plane_model[:3])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # 计算平面法向量与各表面向量的点积
    dot_products = [np.dot(plane_normal, surface_vector / np.linalg.norm(surface_vector))
                   for surface_vector in surface_vectors]

    # 找到最大点积的索引，忽略方向
    max_index = np.argmax(np.abs(dot_products))  # 0: top, 1: left, 2: right

    return max_index


# 计算点云在某个方向上的跨度
def calculate_span(point_cloud, direction):
    """
    计算点云在指定方向上的跨度。

    Args:
        point_cloud (array-like): 点云数据，形状 (n, 3)。
        direction (array-like): 方向向量。

    Returns:
        tuple: (span, projections, span_max, span_min)
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
def split_point_cloud_into_intervals(point_cloud, direction, num_intervals=5):
    """
    将点云在指定方向上的投影分割为多个区间。

    Args:
        point_cloud (array-like): 点云数据，形状 (n, 3)。
        direction (array-like): 方向向量。
        num_intervals (int): 分割的区间数量。

    Returns:
        tuple: (point_intervals, intervals)
            - point_intervals: 分割后的点集列表，每个列表对应一个区间。
            - intervals: 区间的起止值列表。
    """
    span, projections, span_max, span_min = calculate_span(point_cloud, direction)
    interval_length = span / num_intervals
    intervals = [(span_min + i * interval_length, span_min + (i + 1) * interval_length) for i in range(num_intervals)]
    point_intervals = [[] for _ in range(num_intervals)]
    for i, proj in enumerate(projections):
        for j, (start, end) in enumerate(intervals):
            if start <= proj < end:
                point_intervals[j].append(point_cloud[i])
                break
    return point_intervals, intervals


# 计算点云到平面的距离
def distances_to_plane(point_cloud, plane_model):
    """
    计算点云中每个点到指定平面的距离。

    Args:
        point_cloud (array-like): 点云数据，形状 (n, 3)。
        plane_model (list or tuple): 平面模型参数 [A, B, C, D]。

    Returns:
        numpy.ndarray: 距离数组，形状 (n,)。
    """
    A, B, C, D = plane_model
    distances = np.abs(np.dot(point_cloud, plane_model[:3]) + D) / np.linalg.norm(plane_model[:3])
    return distances


# 修改后的 cal_five_dis 函数：选择最远点进行距离计算
def cal_five_dis(origin, through, desti):
    """
    计算两平面交线附近的最远点到origin平面的距离。

    Args:
        origin (list or tuple): 被测表面平面模型 [A, B, C, D]。
        through (list or tuple): 测量方向表面平面模型 [A, B, C, D]。
        desti (open3d.geometry.PointCloud): 被测表面点云对象。

    Returns:
        list: 最远点的坐标列表。
    """
    tmp_plane1 = origin  # 被测表面1，只能是top_face left_face right_face
    tmp_plane2 = through  # 测量方向表面
    plane1 = (tmp_plane1[0], tmp_plane1[1], tmp_plane1[2], tmp_plane1[3])
    plane2 = (tmp_plane2[0], tmp_plane2[1], tmp_plane2[2], tmp_plane2[3])

    try:
        point, direction = plane_intersection(plane1, plane2)
    except ValueError as e:
        print(e)
        return []

    point_cloud = np.asarray(desti.points)
    line_point = point
    line_direction = direction

    # 计算所有点到交线的距离
    distances = point_cloud_to_line_distances(point_cloud, line_point, line_direction)

    # 找到最大距离及对应的点索引
    max_distance_idx = np.argmax(distances)
    max_distance = distances[max_distance_idx]
    farthest_point = point_cloud[max_distance_idx]

    # 计算最远点到origin平面的距离
    distance_to_origin = distances_to_plane([farthest_point], origin)[0]

    print(f"最远点距离为：{distance_to_origin:.4f}")

    return farthest_point.tolist()  # 返回最远点的坐标


# 主流程
def main():
    # 读取点云
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)

    # 数据预处理：滤波（去除噪声）和下采样
    print("进行点云预处理：去除统计离群点和下采样")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_down = pcd.voxel_down_sample(voxel_size=0.005)
    print(f"预处理后点云点数：{len(pcd_down.points)}")

    # 自动检测并分类平面
    pcd_rest = pcd_down
    for i in range(3):  # 仅检测三个平面
        if len(pcd_rest.points) < 100:  # 点云数量过少时停止
            break
        plane_model, inliers = pcd_rest.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        inlier_cloud = pcd_rest.select_by_index(inliers)
        # 分类平面
        classification = classify_plane(plane_model[:3])
        if classification == 0 and classified_planes['top']['model'] is None:
            classified_planes['top']['model'] = plane_model
            classified_planes['top']['points'] = inlier_cloud
            print("检测到顶面")
        elif classification == 1 and classified_planes['left']['model'] is None:
            classified_planes['left']['model'] = plane_model
            classified_planes['left']['points'] = inlier_cloud
            print("检测到左面")
        elif classification == 2 and classified_planes['right']['model'] is None:
            classified_planes['right']['model'] = plane_model
            classified_planes['right']['points'] = inlier_cloud
            print("检测到右面")
        else:
            print("分类错误或重复检测的平面，跳过")
            continue
        # 移除已检测到的平面内的点
        pcd_rest = pcd_rest.select_by_index(inliers, invert=True)

    # 检查是否检测到所有需要的平面
    required_planes = ['top', 'left', 'right']
    for plane in required_planes:
        if classified_planes[plane]['model'] is None:
            print(f"未检测到所需的平面：{plane}")
            sys.exit()

    print('\n分类后的平面模型：')
    for plane in required_planes:
        print(f"{plane} 面方程：{classified_planes[plane]['model']}")

    # Step 2.5：生成用于测量距离的背面平面点集，作为测量背面边缘的终点
    print("\n生成用于测量的背面平面点集")
    # 自动检测背面平面（假设背面平面与左、右、顶面垂直或有特定方向）
    # 这里假设背面平面的法向量与左面或右面平行，可以根据实际情况调整

    # 这里以检测背面平面为例，您可以根据具体情况添加更多平面检测
    # 假设背面平面与左面平行
    # 通过法向量相似度进行平面分类
    def find_back_face(pcd, classified_planes):
        """
        自动检测背面平面。

        Args:
            pcd (open3d.geometry.PointCloud): 点云对象。
            classified_planes (dict): 已分类的平面模型和点云。

        Returns:
            open3d.geometry.PointCloud: 背面平面点云对象。
        """
        # 定义背面平面法向量，假设与左面或右面平行
        # 根据左面和右面的法向量进行选择
        left_normal = np.array(classified_planes['left']['model'][:3])
        left_normal = left_normal / np.linalg.norm(left_normal)
        right_normal = np.array(classified_planes['right']['model'][:3])
        right_normal = right_normal / np.linalg.norm(right_normal)

        # 假设背面平面与左面平面平行
        # 也可以根据实际需求调整
        target_normal = left_normal.copy()

        # 检测与目标法向量夹角小于一定阈值的平面
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        current_normal = np.array(plane_model[:3])
        current_normal = current_normal / np.linalg.norm(current_normal)

        # 计算法向量夹角
        dot_product = np.dot(current_normal, target_normal)
        angle = np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0)) * 180 / np.pi

        if angle < 10:  # 夹角小于10度认为是背面平面
            back_face_cloud = pcd.select_by_index(inliers)
            print("检测到背面平面")
            return back_face_cloud
        else:
            print("未检测到符合条件的背面平面")
            return None

    back_face_points = find_back_face(pcd_rest, classified_planes)
    if back_face_points is not None:
        # 将背面平面加入分类
        classified_planes['back'] = {'model': plane_model, 'points': back_face_points}
    else:
        print("未能检测到背面平面，程序终止。")
        sys.exit()

    # 可视化背面平面（可选）
    # o3d.visualization.draw_geometries([back_face_points], window_name='Back Face')

    # Step 3：计算距离并收集最远点
    # 初始化一个列表用于存储所有最远点
    all_farthest_points = []

    # 定义需要测量的平面组合
    measurement_combinations = [
        ('top', 'left', 'back'),    # top-[left]-back
        ('top', 'right', 'back'),   # top-[right]-back
        ('left', 'right', 'back')   # left-[right]-back
    ]

    for origin_plane, through_plane, desti_plane in measurement_combinations:
        print(f"\n{origin_plane} 面与 {through_plane} 面的最远距离，沿 {through_plane} 面测量：")
        origin = classified_planes[origin_plane]['model']
        through = classified_planes[through_plane]['model']
        desti = classified_planes[desti_plane]['points']
        farthest_point = cal_five_dis(origin, through, desti)
        if farthest_point:
            all_farthest_points.append(farthest_point)

    # 可视化分类后的平面和最远点
    colors = {
        'top': [1, 0, 0],      # 红色
        'left': [0, 1, 0],     # 绿色
        'right': [0, 0, 1],    # 蓝色
        'back': [0.5, 0.5, 0.5],  # 灰色
        'farthest': [1, 1, 0]  # 黄色
    }
    geometries = []

    # 添加分类后的平面
    for plane_name in classified_planes:
        if plane_name in required_planes + ['back'] and classified_planes[plane_name]['points'] is not None:
            plane_cloud = classified_planes[plane_name]['points']
            plane_cloud.paint_uniform_color(colors[plane_name])
            geometries.append(plane_cloud)

    # 创建最远点的点云
    if all_farthest_points:
        farthest_pcd = o3d.geometry.PointCloud()
        farthest_pcd.points = o3d.utility.Vector3dVector(all_farthest_points)
        farthest_pcd.paint_uniform_color(colors['farthest'])  # 最远点颜色为黄色
        geometries.append(farthest_pcd)

    # 创建最远点的几何体（小球）
    if all_farthest_points:
        farthest_geometries = []
        for point in all_farthest_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)  # 小球半径可根据实际情况调整
            sphere.translate(point)
            sphere.paint_uniform_color(colors['farthest'])  # 最远点颜色为黄色
            farthest_geometries.append(sphere)
        geometries.extend(farthest_geometries)

    # 绘制所有几何体
    o3d.visualization.draw_geometries(geometries, window_name='分类后的平面及最远点')


if __name__ == "__main__":
    main()
