# encoding:utf-8
import open3d as o3d
import numpy as np
import sys

# 归类平面用的指向向量，平面会被分为
top_vector = [-0.0694, 0.824, 0.563] # 顶面向量 - 指向下
left_vector = [0.4085, -0.4951, 0.7669] # 左前面向量 - 指向内
right_vector = [0.9106, 0.27539, -0.308] # 右前面向量 - 指向外
surface_vectors = [top_vector, left_vector, right_vector] # Store surface vectors in a list for easy iteration
# 虚拟平面点集，预备变量
top_face = []
right_face = []
left_face = []
file_path = r'F:\work\python\team\blcok\data\original\A6.pcd'


# 用平面分割点云 排除不需要部分
def planar_cut_off(pcd, plane_model, remove_above=True):
    """
    Performs a planar cut-off on a point cloud.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        plane_model (list): The plane model parameters [a, b, c, d] representing the equation ax + by + cz + d = 0.
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

# 通过3个指定点 计算它们所在平面方程
def plane_from_points(p1, p2, p3):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate two vectors from the points
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute the cross product of the vectors to get the normal vector
    normal_vector = np.cross(v1, v2)

    # The normal vector gives the coefficients A, B, C of the plane equation
    A, B, C = normal_vector

    # Use the normal vector and one of the points to find D
    D = -np.dot(normal_vector, p1)

    return A, B, C, D

# 计算两平面交线
def plane_intersection(plane1, plane2):
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
        raise ValueError("The planes are parallel or identical, no unique line of intersection.")

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
        # If the determinant is 0, the planes are either parallel or we need another approach
        # Fix x = 0 and solve for y and z
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

    # Ensure the direction vector is normalized (optional, but good practice)
    d = d / np.linalg.norm(d)

    # Convert the point cloud to a numpy array (n, 3)
    point_cloud = np.array(point_cloud)

    # Step 1: Compute the vector from the line point to each point in the point cloud
    P0_P = point_cloud - P  # Shape (n, 3)

    # Step 2: Calculate the cross product between the direction vector and P0_P for each point
    cross_prods = np.cross(d, P0_P)  # Shape (n, 3)

    # Step 3: Calculate the magnitudes of the cross products (distances to the line)
    numerators = np.linalg.norm(cross_prods, axis=1)  # Shape (n,)

    # Step 4: The denominator is the magnitude of the direction vector (which is 1 if normalized)
    denominator = np.linalg.norm(d)

    # Step 5: Calculate the distances
    distances = numerators / denominator

    return distances

# 根据函数point_cloud_to_line_distances返回值，筛选距离小于特定值的点集
def filter_points_by_distance(point_cloud, distances, threshold=2):
    """
    Filter points whose distance from the line is greater than a given threshold.

    Parameters:
    point_cloud (array-like): A list or array of points in 3D space, shape (n, 3).
    distances (array-like): A list or array of distances corresponding to each point.
    threshold (float): The distance threshold for filtering points.

    Returns:
    list: A list of points whose distance is greater than the threshold.
    """
    # Convert point cloud and distances to numpy arrays (if they aren't already)
    point_cloud = np.array(point_cloud)
    distances = np.array(distances)

    # Filter points where the distance is greater than the threshold
    filtered_points = point_cloud[distances <= threshold]

    return filtered_points

# 根据平面方向 将平面分为 top left right
def classify_plane(plane_model):
    plane_model = plane_model / np.linalg.norm(plane_model)
    dot_products = [np.dot(plane_model, surface_vector) for surface_vector in surface_vectors]
    # Find the index of the highest dot product (closest alignment)
    max_index = np.argmax(np.abs(dot_products))  # Use absolute value to ignore direction
    return max_index

# 计算点云在某个方向上的跨度。输入 point_cloud - 点云集合（点集对象.points），direction - 平面模型ABCD
def calculate_span(point_cloud, direction):
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # Project each point in the point cloud onto the direction vector
    projections = np.dot(point_cloud, direction)
    span_max = np.max(projections)
    span_min = np.min(projections)

    # Calculate the span (max projection - min projection)
    span = span_max - span_min

    return span, projections, span_max, span_min


# Function to split the point cloud into intervals based on the span
def split_point_cloud_into_intervals(point_cloud, direction):
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


# Function to calculate the distance from points in a point cloud to a plane
def distances_to_plane(point_cloud, plane_model):
    # Calculate the distance for each point in the point cloud
    A = plane_model[0]
    B = plane_model[1]
    C = plane_model[2]
    D = plane_model[3]

    distances = []
    for point in point_cloud:
        x, y, z = point
        # Use the point-plane distance formula
        distance = abs(A * x + B * y + C * z + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)
        distances.append(distance)

    return np.array(distances)

# 根据预设，计算一个表面到一个背面边 5个位置的最大距离
def cal_five_dis(origin, through, desti):
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
    #o3d.visualization.draw_geometries([tmp_point_cloud], window_name='Points in Cylinder')

    #调试用：计算点云在指定方向的span的示例程序。span是总跨度，projections是各点在指定方向上的投影，span_max span_min 是投影最大值和最小值
    #span, projections, span_max, span_min = calculate_span(points_in_cylinder, line_direction)
    #print(f"Span of the point cloud in the given direction: {span}")

    point_intervals, intervals = split_point_cloud_into_intervals(point_cloud, line_direction)
    #调试用： Output the points in each interval
    #for i, interval_points in enumerate(point_intervals):
    #    tmp_point_cloud = o3d.geometry.PointCloud()
    #    tmp_point_cloud.points = o3d.utility.Vector3dVector(interval_points)
    #    o3d.visualization.draw_geometries([tmp_point_cloud], window_name='Points in Interval')
    #    print(f"Interval {i+1} (centered around {intervals[i][0] + 3/2:.2f}):")
    #    print(np.array(interval_points))

    # Calculate distances
    for intervalA in point_intervals:
        distances = distances_to_plane(intervalA, tmp_plane1)
        print('最大距离：', np.max(distances))


# 默认流程：读取点云
pcd = o3d.io.read_point_cloud(file_path) # Open3d读取到的点云通常存储到PointCloud类中,这个类中我们常用的属性就是points和colors
points = np.asarray(pcd.points)


# Module1: 读取点云和点云可视化
if False:
    colors = np.asarray(pcd.colors) * 255 # colors中的RGB数据是归一化的，所以要乘以255
    print(points.shape, colors.shape)
    print(np.concatenate([points, colors], axis=-1))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud])
    # o3d.visualization.draw_geometries([point_cloud],
    #                                  zoom=1,
    #                                  front=[0, 0, 1],
    #                                  lookat=[0, 0, 0],
    #                                  up=[0, 0, 1])


# Module2: 对点云进行滤波（去除离群点）和下采样（去除过密的数据点）
if False:
    # Remove outliers
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Downsample
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)

    # 拟合平面（准备检测立方体的角点）
    # Example in Python using Open3D
    plane_model, inliers = downpcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    print("Finished")


# Module3: 平面切分点云，起到直通滤波效果
if False:
    # 左后三个点及裕量 - 完成
    p1 = (-72.463, -59.748, 384.193 + 100)
    p2 = (-97.307, -20.437, 353.261 + 100)
    p3 = (-99.957, 20.620, 381.476 + 100)
    A, B, C, D = plane_from_points(p1, p2, p3)
    print(f"The equation of the plane is: {A}x + {B}y + {C}z + {D} = 0")
    # Define the plane model parameters
    plane_model = [A, B, C, D]  # Example: A horizontal plane at y = 1
    # Perform the planar cut-off
    pcd_filtered = planar_cut_off(pcd, plane_model, False)

    # 底面三个点及裕量 - 完成
    p1 = (-28.035, 57.862 + 5, 363.491)
    p2 = (-83.535, 39.338 + 5, 383.924)
    p3 = (28.207, 19.614 + 5, 424.500)
    A, B, C, D = plane_from_points(p1, p2, p3)
    print(f"The equation of the plane is: {A}x + {B}y + {C}z + {D} = 0")
    # Define the plane model parameters
    plane_model = [A, B, C, D]  # Example: A horizontal plane at y = 1
    # Perform the planar cut-off
    pcd_filtered = planar_cut_off(pcd_filtered, plane_model, True)

    # 右后三个点及裕量 - 完成
    p1 = (-35.869, -76.708, 413.134 + 50)
    p2 = (11.503, -60.635, 395.618 + 50)
    p3 = (35.021, -13.856, 412.638 + 50)
    A, B, C, D = plane_from_points(p1, p2, p3)
    print(f"The equation of the plane is: {A}x + {B}y + {C}z + {D} = 0")
    # Define the plane model parameters
    plane_model = [A, B, C, D]  # Example: A horizontal plane at y = 1
    # Perform the planar cut-off
    pcd_filtered = planar_cut_off(pcd_filtered, plane_model, True)

    # Visualize the result
    o3d.visualization.draw_geometries([pcd_filtered])


# Module4: 迭代拟合平面测试
if True:
    # Step 1：平面切割法 去除多余点
    if True:
        # 左后三个点及裕量
        p1 = (-72.463, -59.748, 384.193 + 100)
        p2 = (-97.307, -20.437, 353.261 + 100)
        p3 = (-99.957, 20.620, 381.476 + 100)
        A, B, C, D = plane_from_points(p1, p2, p3)
        # print(f"The equation of the plane is: {A}x + {B}y + {C}z + {D} = 0")
        # Define the plane model parameters
        plane_model = [A, B, C, D]  # Example: A horizontal plane at y = 1
        # Perform planar cut-off
        pcd_filtered = planar_cut_off(pcd, plane_model, False)
        o3d.visualization.draw_geometries([pcd_filtered], window_name='Filtered Body00') # 显示过滤后的全部可用点
        # 底面三个点及裕量
        p1 = (-28.035, 57.862 + 5, 363.491)
        p2 = (-83.535, 39.338 + 5, 383.924)
        p3 = (28.207, 19.614 + 5, 424.500)
        A, B, C, D = plane_from_points(p1, p2, p3)
        #print(f"The equation of the plane is: {A}x + {B}y + {C}z + {D} = 0")
        # Define the plane model parameters
        plane_model = [A, B, C, D]  # Example: A horizontal plane at y = 1
        # Perform the planar cut-off
        pcd_filtered = planar_cut_off(pcd_filtered, plane_model, True)
        o3d.visualization.draw_geometries([pcd_filtered], window_name='Filtered Body01') # 显示过滤后的全部可用点
        # 右后三个点及裕量
        p1 = (-35.869, -76.708, 413.134 + 50)
        p2 = (11.503, -60.635, 395.618 + 50)
        p3 = (35.021, -13.856, 412.638 + 50)
        A, B, C, D = plane_from_points(p1, p2, p3)
        #print(f"The equation of the plane is: {A}x + {B}y + {C}z + {D} = 0")
        # Define the plane model parameters
        plane_model = [A, B, C, D]  # Example: A horizontal plane at y = 1
        # Perform the planar cut-off，输出过滤之后的点云对象 pcd_filtered
        pcd_filtered = planar_cut_off(pcd_filtered, plane_model, True)

        o3d.visualization.draw_geometries([pcd_filtered], window_name='Filtered Body02') # 显示过滤后的全部可用点

    # Step 2：拟合平面（准备检测立方体的角点）.plane_model 平面模型；inlier 该平面的内点索引
    if True:
        # 原始点集 pcd_filtered，剩余点集 pcd_rest，分出去的点集 plane_model1_points；
        plane_model1, inliers1 = pcd_filtered.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
        print('Face-1 equation:', plane_model1)
        # 排除已经拟合到某一平面的点
        pcd_rest = pcd_filtered.select_by_index(inliers1, invert=True)
        plane_model1_points = pcd_filtered.select_by_index(inliers1)
        o3d.visualization.draw_geometries([pcd_rest], window_name='Face-1 Fitting')
        # 将拟合出的平面进行分类 - top left right
        tmp_plane = plane_model1
        tmp_plane_points = plane_model1_points
        tmp = classify_plane(tmp_plane[:3])
        if tmp == 0 and top_face == []:
            top_face = tmp_plane
            top_face_points = tmp_plane_points
        elif tmp == 1 and left_face == []:
            left_face = tmp_plane
            left_face_points = tmp_plane_points
        elif tmp == 2 and right_face == []:
            right_face = tmp_plane
            right_face_points = tmp_plane_points
        else:
            print("Classification Error")
            sys.exit()


        # 用剩余点再次拟合
        # 输入点集 pcd_rest，剩余点集 pcd_rest2，分出去的点集 plane_model2_points；
        plane_model2, inliers2 = pcd_rest.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
        print('Face-2 equation:', plane_model2)
        pcd_rest2 = pcd_rest.select_by_index(inliers2, invert=True)
        plane_model2_points = pcd_rest.select_by_index(inliers2)
        o3d.visualization.draw_geometries([pcd_rest2], window_name='Face-2 Fitting')
        # 将拟合出的平面进行分类 - top left right
        tmp_plane = plane_model2
        tmp_plane_points = plane_model2_points
        tmp = classify_plane(tmp_plane[:3])
        if tmp == 0 and top_face == []:
            top_face = tmp_plane
            top_face_points = tmp_plane_points
        elif tmp == 1 and left_face == []:
            left_face = tmp_plane
            left_face_points = tmp_plane_points
        elif tmp == 2 and right_face == []:
            right_face = tmp_plane
            right_face_points = tmp_plane_points
        else:
            print("Classification Error")
            sys.exit()


        # 用剩余点再次拟合
        # 输入点集 pcd_rest2，剩余点集 pcd_rest3，分出去的点集 plane_model3_points；
        plane_model3, inliers3 = pcd_rest2.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=10000)
        print('Face-3 equation:', plane_model3)
        pcd_rest3 = pcd_rest2.select_by_index(inliers3, invert=True)
        plane_model3_points = pcd_rest2.select_by_index(inliers3)
        o3d.visualization.draw_geometries([pcd_rest3], window_name='Face-3 Fitting')
        # 将拟合出的平面进行分类 - top left right
        tmp_plane = plane_model3
        tmp_plane_points = plane_model3_points
        tmp = classify_plane(tmp_plane[:3])
        if tmp == 0 and top_face == []:
            top_face = tmp_plane
            top_face_points = tmp_plane_points
        elif tmp == 1 and left_face == []:
            left_face = tmp_plane
            left_face_points = tmp_plane_points
        elif tmp == 2 and right_face == []:
            right_face = tmp_plane
            right_face_points = tmp_plane_points
        else:
            print("Classification Error")
            sys.exit()

        print('Top Face equation:',top_face)
        print('Left Face equation:',left_face)
        print('Right Face equation:',right_face)

    # 当前总结：
    # 得到3个集总参数的虚拟平面：top_face、left_face、right_face - 平面模型ABCD；及对应点集对象top_face_points、left_face_points...
    # 切除多余点后的 总有效点集：pcd_filtered - 点集对象
    # 暂存的原始输出平面及其点集：（以下这3个平面分别对应了top_face、left_face、right_face）
    #   plane_model1 - 平面1模型ABCD, inliers1 - 索引（无用），plane_model1_points - 点集对象；
    #   plane_model2 - 平面2模型ABCD, inliers2 - 索引（无用），plane_model2_points - 点集对象；
    #   plane_model3 - 平面3模型ABCD, inliers3 - 索引（无用），plane_model3_points - 点集对象；
    # 剩余点集：pcd_rest3 - 点集对象
    #调试用：可视化拟合出的3个平面所包含的点集
    #o3d.visualization.draw_geometries([plane_model1_points])
    #o3d.visualization.draw_geometries([plane_model2_points])
    #o3d.visualization.draw_geometries([plane_model3_points])

    if True:
        # 定义一个函数用于高亮显示选定的三个点
        def highlight_selected_points(p1, p2, p3, original_pcd):
            selected_points = np.array([p1, p2, p3])
            selected_pcd = o3d.geometry.PointCloud()
            selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
            # 设置颜色为红色
            selected_pcd.paint_uniform_color([1, 0, 0])
            # 可以选择放大点的大小，但Open3D默认不支持单独设置点大小
            return selected_pcd


        # 平面切分pcd_filtered
        # 左后三个点及裕量
        p1 = (-72.463, -59.748, 384.193 - 30)
        p2 = (-97.307, -20.437, 353.261 - 30)
        p3 = (-99.957, 20.620, 381.476 - 30)
        A, B, C, D = plane_from_points(p1, p2, p3)
        plane_model = [A, B, C, D]
        backleft_face_points = planar_cut_off(pcd_filtered, plane_model, True)

        # 高亮显示选定的三个点
        backleft_selected = highlight_selected_points(p1, p2, p3, pcd_filtered)

        # 为backleft_face_points染色为绿色
        backleft_face_points.paint_uniform_color([0, 1, 0])  # 绿色

        # 获取剩余点云
        pcd_rest_after_backleft = pcd_filtered.select_by_index(backleft_face_points.points, invert=True)
        pcd_rest_after_backleft.paint_uniform_color([0, 0, 1])  # 蓝色

        # 将染色后的点云与选定的点合并进行可视化
        o3d.visualization.draw_geometries([
            backleft_face_points,
            pcd_rest_after_backleft,
            backleft_selected
        ], window_name='backleft_face_points')

        # 平面切分pcd_filtered
        # 右后三个点及裕量
        p1 = (-35.869, -76.708, 413.134 - 30)
        p2 = (11.503, -60.635, 395.618 - 30)
        p3 = (35.021, -13.856, 412.638 - 30)
        A, B, C, D = plane_from_points(p1, p2, p3)
        plane_model = [A, B, C, D]
        backright_face_points = planar_cut_off(pcd_filtered, plane_model, False)

        # 高亮显示选定的三个点
        backright_selected = highlight_selected_points(p1, p2, p3, pcd_filtered)

        # 为backright_face_points染色为绿色
        backright_face_points.paint_uniform_color([0, 1, 0])  # 绿色

        # 获取剩余点云
        pcd_rest_after_backright = pcd_filtered.select_by_index(backright_face_points.points, invert=True)
        pcd_rest_after_backright.paint_uniform_color([0, 0, 1])  # 蓝色

        # 将染色后的点云与选定的点合并进行可视化
        o3d.visualization.draw_geometries([
            backright_face_points,
            pcd_rest_after_backright,
            backright_selected
        ], window_name='backright_face_points')

        # 平面切分pcd_filtered
        # 底面三个点及裕量
        p1 = (-28.035, 57.862 - 15, 363.491)
        p2 = (-83.535, 39.338 - 15, 383.924)
        p3 = (28.207, 19.614 - 15, 424.500)
        A, B, C, D = plane_from_points(p1, p2, p3)
        plane_model = [A, B, C, D]
        bottom_face_points = planar_cut_off(pcd_filtered, plane_model, False)

        # 高亮显示选定的三个点
        bottom_selected = highlight_selected_points(p1, p2, p3, pcd_filtered)

        # 为bottom_face_points染色为绿色
        bottom_face_points.paint_uniform_color([0, 1, 0])  # 绿色

        # 获取剩余点云
        pcd_rest_after_bottom = pcd_filtered.select_by_index(bottom_face_points.points, invert=True)
        pcd_rest_after_bottom.paint_uniform_color([0, 0, 1])  # 蓝色

        # 将染色后的点云与选定的点合并进行可视化
        o3d.visualization.draw_geometries([
            bottom_face_points,
            pcd_rest_after_bottom,
            bottom_selected
        ], window_name='bottom_face_points')

    # 获得背面用测量点云：backleft_face_points、backright_face_points、bottom_face_points - 点集对象
    # 获得背面用测量点云：backleft_face_points、backright_face_points、bottom_face_points - 点集对象

    # Step 3：找两平面交线、根据交线切分被测量、测量各组被测点距目标点的最大距离
    # 分组测距函数 cal_five_dis的
    # 可选项1 top-[left]-bottom、top-[right]-bottom
    #   函数参数为：cal_five_dis(top_face, left_face, bottom_face_points)、cal_five_dis(top_face, right_face, bottom_face_points)
    # 可选项2 left-[top]-backright、left-[right]-backright
    #   函数参数为：cal_five_dis(left_face, top_face, backright_face_points)、cal_five_dis(left_face, right_face, backright_face_points)
    # 可选项3 right-[top]-backleft、right-[left]-backleft
    #   函数参数为：cal_five_dis(right_face, top_face, backleft_face_points)、cal_five_dis(right_face, left_face, backleft_face_points)

    # 示例：【计算top_face与bottom_face_points的距离，经过left_face；相当于测量top-[left]-bottom】
    if False:
        # top_face与left_face的交线
        #o3d.visualization.draw_geometries([top_face_points], window_name='Top Face')
        print("顶面与底面的距离，沿左侧面测量：")
        tmp_plane1 = top_face # 被测表面1，只能是top_face left_face right_face
        tmp_plane2 = left_face # 测量方向表面，只能是
        plane1 = (tmp_plane1[0],tmp_plane1[1],tmp_plane1[2],tmp_plane1[3])
        plane2 = (tmp_plane2[0],tmp_plane2[1],tmp_plane2[2],tmp_plane2[3])
        # 使用“直线经过的点point + 直线的方向向量direction”表示一条直线
        point, direction = plane_intersection(plane1, plane2)
        #print("Point on the intersection line:", point)
        #print("Direction of the intersection line:", direction)

        # Step 4：计算“点集”中的点到直线的距离 - 点集是一个平面，直线是两平面交线。用于提取 离相交轴比较近的点集。
        point_cloud = bottom_face_points.points # 被测表面2，只能是backleft_face_points backright_face_points bottom_face_points
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
        points_in_cylinder = filter_points_by_distance(point_cloud, distances, threshold = 20) # 暂时无用
        # 调试用：显示圆柱体内的点云
        #tmp_point_cloud = o3d.geometry.PointCloud()
        #tmp_point_cloud.points = o3d.utility.Vector3dVector(points_in_cylinder)
        #o3d.visualization.draw_geometries([tmp_point_cloud], window_name='Points in Cylinder')

        #调试用：计算点云在指定方向的span的示例程序。span是总跨度，projections是各点在指定方向上的投影，span_max span_min 是投影最大值和最小值
        #span, projections, span_max, span_min = calculate_span(points_in_cylinder, line_direction)
        #print(f"Span of the point cloud in the given direction: {span}")

        point_intervals, intervals = split_point_cloud_into_intervals(bottom_face_points.points, line_direction)
        #调试用： Output the points in each interval
        #for i, interval_points in enumerate(point_intervals):
        #    tmp_point_cloud = o3d.geometry.PointCloud()
        #    tmp_point_cloud.points = o3d.utility.Vector3dVector(interval_points)
        #    o3d.visualization.draw_geometries([tmp_point_cloud], window_name='Points in Interval')
        #    print(f"Interval {i+1} (centered around {intervals[i][0] + 3/2:.2f}):")
        #    print(np.array(interval_points))

        # Calculate distances
        for intervalA in point_intervals:
            distances = distances_to_plane(intervalA, top_face)
            print('最大距离：', np.max(distances))

    # 函数调用实验
    print("顶面与底面的距离，沿左侧面测量：")
    cal_five_dis(top_face, left_face, bottom_face_points)

    print("顶面与底面的距离，沿右侧面测量：")
    cal_five_dis(top_face, right_face, bottom_face_points)

    print("左侧面与右后面的距离，沿顶面测量：")
    cal_five_dis(left_face, top_face, backright_face_points)
