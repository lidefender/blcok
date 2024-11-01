# encoding:utf-8
import open3d as o3d
import numpy as np
import sys

# 归类平面用的指向向量，平面会被分为
top_vector = np.array([0, 0, 1])  # 顶面向量 - 指向上
left_vector = np.array([0, 1, 0])  # 左面向量 - 指向左
right_vector = np.array([1, 0, 0])  # 右面向量 - 指向前

# 存储表面向量列表
surface_vectors = [top_vector, left_vector, right_vector]

# 存储分类后的平面模型和点云
classified_planes = {
    'top': {'model': None, 'points': None},
    'left': {'model': None, 'points': None},
    'right': {'model': None, 'points': None}
}

file_path = r'C:\Users\Liminghui\Desktop\Python\Python\Data\aaa.pcd'


# 计算两平面的交线
def plane_intersection(plane1, plane2):
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2
    normal1 = np.array([a1, b1, c1])
    normal2 = np.array([a2, b2, c2])
    direction = np.cross(normal1, normal2)
    if np.all(direction == 0):
        raise ValueError("两个平面平行或重合，没有唯一的交线。")
    # 选择其中一个坐标固定，求解交线上一个点
    A = np.array([[a1, b1],
                  [a2, b2]])
    B = np.array([-d1, -d2])
    if np.linalg.matrix_rank(A) == 2:
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
    P = np.array(line_point)
    d = np.array(line_direction)
    d = d / np.linalg.norm(d)
    point_cloud = np.array(point_cloud)
    P0_P = point_cloud - P
    cross_prods = np.cross(d, P0_P)
    distances = np.linalg.norm(cross_prods, axis=1)  # 已经除以 |d|
    return distances


# 根据距离筛选点
def filter_points_by_distance(point_cloud, distances, threshold=2):
    point_cloud = np.array(point_cloud)
    distances = np.array(distances)
    filtered_points = point_cloud[distances <= threshold]
    return filtered_points


# 分类平面
def classify_plane(plane_model):
    plane_normal = np.array(plane_model[:3])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    dot_products = [np.dot(plane_normal, surface_vector / np.linalg.norm(surface_vector)) for surface_vector in
                    surface_vectors]
    max_index = np.argmax(np.abs(dot_products))  # 使用绝对值忽略方向
    return max_index  # 0: top, 1: left, 2: right


# 计算点云在某个方向上的跨度
def calculate_span(point_cloud, direction):
    direction = direction / np.linalg.norm(direction)
    projections = np.dot(point_cloud, direction)
    span_max = np.max(projections)
    span_min = np.min(projections)
    span = span_max - span_min
    return span, projections, span_max, span_min


# 将点云按跨度分割为多个区间
def split_point_cloud_into_intervals(point_cloud, direction, num_intervals=5):
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
    A, B, C, D = plane_model
    distances = np.abs(np.dot(point_cloud, plane_model[:3]) + D) / np.linalg.norm(plane_model[:3])
    return distances


# 根据预设，计算一个表面到一个对边面 5个位置的最大距离
def cal_five_dis(origin, through, desti, num_measure=5):
    plane1 = origin
    plane2 = through
    try:
        point, direction = plane_intersection(plane1, plane2)
    except ValueError as e:
        print(e)
        return
    point_cloud = np.asarray(desti.points)
    line_point = point
    line_direction = direction
    distances = point_cloud_to_line_distances(point_cloud, line_point, line_direction)
    point_intervals, intervals = split_point_cloud_into_intervals(point_cloud, line_direction,
                                                                  num_intervals=num_measure)

    # 选择每个区间的中间点作为测量点
    measurement_points = []
    for interval_points in point_intervals:
        if len(interval_points) == 0:
            continue
        interval_points = np.array(interval_points)
        span, projections, span_max, span_min = calculate_span(interval_points, line_direction)
        center_proj = (span_max + span_min) / 2
        closest_idx = np.argmin(np.abs(projections - center_proj))
        measurement_points.append(interval_points[closest_idx])
        if len(measurement_points) >= num_measure:
            break

    # 确保选择了足够的测量点
    if len(measurement_points) < num_measure:
        print(f"警告：仅选择了{len(measurement_points)}个测量点，期望{num_measure}个。")

    # 计算这些测量点到origin平面的距离
    distances_to_origin = distances_to_plane(measurement_points, origin)
    average_distance = np.mean(distances_to_origin)

    print(f"测量点距离最大值：{np.max(distances_to_origin):.4f}")
    print(f"测量点距离最小值：{np.min(distances_to_origin):.4f}")
    print(f"平均距离：{average_distance:.4f}")


# 主流程
def main():
    # 读取点云
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)

    # 数据预处理：根据具体情况进行滤波，这里以z >= -7且y <= 180为例
    # 根据您的实际坐标系和长方体位置调整滤波条件
    filtered_points = points[(points[:, 2] >= -7) & (points[:, 1] <= 180)]
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)

    # 自动检测并分类平面
    pcd_rest = pcd_filtered
    for i in range(3):  # 仅检测三个平面
        if len(pcd_rest.points) < 100:  # 点云数量过少时停止
            break
        plane_model, inliers = pcd_rest.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
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

    # 生成用于测量距离的背面平面点集
    # 因为只有三个面，所以无需生成背面和平面的点集
    # 假设测量目标是长方体的对边
    # 例如：顶面到左面的距离，顶面到右面的距离，左面到右面的距离

    # 计算距离
    print("\n顶面与左面的距离：")
    cal_five_dis(classified_planes['top']['model'], classified_planes['left']['model'],
                 classified_planes['left']['points'])

    print("\n顶面与右面的距离：")
    cal_five_dis(classified_planes['top']['model'], classified_planes['right']['model'],
                 classified_planes['right']['points'])

    print("\n左面与右面的距离：")
    cal_five_dis(classified_planes['left']['model'], classified_planes['right']['model'],
                 classified_planes['right']['points'])

    # 可选：可视化分类后的平面
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    geometries = []
    for idx, plane in enumerate(required_planes):
        if classified_planes[plane]['points'] is not None:
            plane_cloud = classified_planes[plane]['points']
            plane_cloud.paint_uniform_color(colors[idx % len(colors)])
            geometries.append(plane_cloud)
    o3d.visualization.draw_geometries(geometries, window_name='分类后的平面')


if __name__ == "__main__":
    main()
