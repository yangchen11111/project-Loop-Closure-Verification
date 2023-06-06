import cv2
import argparse
import os
import re
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor
from camera_model import CameraModel
from build_pointcloud import build_pointcloud
from image import load_image
from transform import build_se3_transform
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import glob

def compute_orb_keypoints(filename):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    # load image
    img = cv2.imread(filename)
    
    # create orb object
    orb = cv2.ORB_create()
    
    # set parameters 
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    orb.setWTA_K(3)
    
    # detect keypoints
    kp = orb.detect(img,None)

    # for detected keypoints compute descriptors. 
    kp, des = orb.compute(img, kp)
    
    return img,kp, des

def estimate_rigid_transform(points1, points2):
    assert points1.shape == points2.shape, "Input point clouds must have the same size"
    
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2
    
    H = centered_points1.T @ centered_points2
    U, _, Vt = np.linalg.svd(H)
    rotation_matrix = Vt.T @ U.T
    
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = Vt.T @ U.T
        
    translation_vector = centroid2 - rotation_matrix @ centroid1
    
    return rotation_matrix, translation_vector

def ransac_initial_alignment(points1, points2, min_samples=3, max_trials=100, residual_threshold=1):
    assert points1.shape == points2.shape, "Input point clouds must have the same size"
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points2)
    _, indices = nbrs.kneighbors(points1)
    matched_points2 = points2[indices].reshape(-1, 3)
    
    best_inliers = None
    best_error = np.inf
    best_inliers = np.zeros(points1.shape[0], dtype=bool)

    for _ in range(max_trials):
        # Check if points1.shape[0] is smaller than min_samples
        if points1.shape[0] < min_samples:
            min_samples = points1.shape[0]
        sample_indices = np.random.choice(points1.shape[0], min_samples, replace=False)
        sampled_points1 = points1[sample_indices]
        sampled_points2 = matched_points2[sample_indices]
        
        R, t = estimate_rigid_transform(sampled_points1, sampled_points2)
        
        transformed_points1 = (R @ points1.T).T + t
        errors = np.linalg.norm(transformed_points1 - matched_points2, axis=1)
        
        inliers = errors < residual_threshold
        inlier_count = np.sum(inliers)
        
        if inlier_count > np.sum(best_inliers):
            best_inliers = inliers
            best_error = np.mean(errors[inliers])
            
    R, t = estimate_rigid_transform(points1[best_inliers], matched_points2[best_inliers])
    
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    
    return transformation_matrix

def project_laser_into_camera(image_dir, laser_dir, poses_file, models_dir, extrinsics_dir, image_idx):
    
    model = CameraModel(models_dir, image_dir)

    extrinsics_path = os.path.join(extrinsics_dir, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)
    G_camera_posesource = None

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
    else:
        # VO frame and vehicle frame are the same
        G_camera_posesource = G_camera_vehicle

    timestamps_path = os.path.join(image_dir, os.pardir, model.camera + '.timestamps')

    
    if not os.path.isfile(timestamps_path):
        timestamps_path = os.path.join(image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

    timestamp = 0
    with open(timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            if i == int(image_idx):
                timestamp = int(line.split(' ')[0])

    try:
        pointcloud, reflectance = build_pointcloud(laser_dir, poses_file, extrinsics_dir,timestamp - 1e7, timestamp + 1e7, timestamp)
    except ValueError:
        return None, None

    pointcloud = np.dot(G_camera_posesource, pointcloud)

    image_path = os.path.join(image_dir, str(timestamp) + '.jpg')
    image = load_image(image_path, model)

    uv, depth = model.project(pointcloud, image.shape)

    # 筛选有效的深度值
    valid_depth_indices = np.where(np.isfinite(depth))

    # 获取对应的三维坐标
    valid_points_3d = pointcloud[:, valid_depth_indices].squeeze()

    # 只保留前三行（x, y, z坐标）
    valid_points_3d = valid_points_3d[:3, :]

    # 转置数组，使每一行表示一个点的x，y，z坐标
    valid_points_3d = valid_points_3d.T

    K = np.array([[model.focal_length[0], 0, model.principal_point[0]],
              [0, model.focal_length[1], model.principal_point[1]],
              [0, 0, 1]])

    return valid_points_3d, K

def nameTrans(img_fname):
    prefix, img_name = img_fname.split('/')
    
    if prefix == 'Autumn_mini_val_query':
        base_dir = 'Autumn_val'
    elif prefix == 'Suncloud_mini_val_ref':
        base_dir = 'Suncloud_val'
    elif prefix == 'Night_mini_val_ref':
        base_dir = 'Night_val'
    else:
        raise ValueError(f'Unknown prefix: {prefix}')
    
    image_dir = os.path.join(base_dir, 'stereo', 'centre')
    laser_dir = os.path.join(base_dir, 'ldmrs')
    poses_file = os.path.join(base_dir, 'vo', 'vo.csv')
    
    image_files = sorted(os.listdir(image_dir))
    image_idx = image_files.index(img_name)
    
    return image_dir, laser_dir, poses_file, str(image_idx)

def point_cloud_registration(points1, points2):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points1)
    target.points = o3d.utility.Vector3dVector(points2)

    threshold = 0.02
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return reg_p2p.transformation

def preprocess_point_clouds(points1, points2):

    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    if points1.shape[1] < 3:
        points1 = np.pad(points1, ((0, 0), (0, 3 - points1.shape[1])), mode='constant', constant_values=0)

    if points2.shape[1] < 3:
        points2 = np.pad(points2, ((0, 0), (0, 3 - points2.shape[1])), mode='constant', constant_values=0)
        
    if points1.shape[0] < points2.shape[0]:
        smaller_points, larger_points = points1, points2
    else:
        smaller_points, larger_points = points2, points1
    
    # if points1.shape[1] < points2.shape[1]:
    #     smaller_points, larger_points = points1, points2
    # else:
    #     smaller_points, larger_points = points2, points1


    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(larger_points)
    distances, indices = nbrs.kneighbors(smaller_points)

    # 从较大的点云中选择与较小点云中每个点最近的点
    sampled_points = larger_points[indices.flatten()]

    return smaller_points, sampled_points

def icp(points1, points2, initial_transformation_matrix,max_iterations=25, tolerance=1):
    # 初始化变换矩阵为单位矩阵
    transformation_matrix = initial_transformation_matrix

    # 将点云转换为齐次坐标
    points1_h = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2_h = np.hstack((points2, np.ones((points2.shape[0], 1))))

    for i in range(max_iterations):
        # 寻找最近点
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points2)
        distances, indices = nbrs.kneighbors(points1)

        # 计算变换矩阵
        src_center = np.mean(points1, axis=0)
        tgt_center = np.mean(points2[indices].reshape(-1, 3), axis=0)

        src_centered = points1 - src_center
        tgt_centered = points2[indices].reshape(-1, 3) - tgt_center

        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_center - R @ src_center

        # 更新变换矩阵
        current_transformation = np.eye(4)
        current_transformation[:3, :3] = R
        current_transformation[:3, 3] = t

        transformation_matrix = current_transformation @ transformation_matrix

        # 更新点云
        points1_h = (current_transformation @ points1_h.T).T

        # 检查收敛
        mean_error = np.mean(distances)
        if mean_error < tolerance:
            break

    return transformation_matrix

def loop_verification(img_fname_1, img_fname_2, new_size=None, is_vis=False):
    
    # """
    # Takes in filenames of two input images 
    # Return Fundamental matrix computes 
    # using 8 point algorithm
    # """
    models_dir = 'robotcar-dataset-sdk/models'
    extrinsics_dir = 'robotcar-dataset-sdk/extrinsics'
    image_dir1, laser_dir1, poses_file1, image_idx1 = nameTrans(img_fname_1)
    image_dir2, laser_dir2, poses_file2, image_idx2 = nameTrans(img_fname_2)

    points1, K1 = project_laser_into_camera(image_dir1, laser_dir1, poses_file1, models_dir, extrinsics_dir, image_idx1)
    points2, K2 = project_laser_into_camera(image_dir2, laser_dir2, poses_file2, models_dir, extrinsics_dir, image_idx2)

    if points1 is None or points2 is None:
        return 1
    
    
    # 检查 smaller_points 和 larger_points 是否为空
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        return 1
    processed_points1, processed_points2 = preprocess_point_clouds(points1, points2)
    # 检查 smaller_points 和 larger_points 是否为空
    if processed_points1.shape[0] == 0 or processed_points2.shape[0] == 0:
        return 1
    
    # 使用RANSAC估计初始变换矩阵
    initial_transformation_matrix = ransac_initial_alignment(processed_points1, processed_points2)
    
    transformation_matrix = icp(processed_points1, processed_points2, initial_transformation_matrix)

    # compute ORB keypoints and descriptor for each image
    img1, kp1, des1 = compute_orb_keypoints(image_dir1)
    img2, kp2, des2 = compute_orb_keypoints(image_dir2)

    # 进行特征匹配并进行几何验证
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append([m])

    filtered_matches = []
    threshold=300
    
    for match in good_matches:
        pt1 = np.array([kp1[match[0].queryIdx].pt[0], kp1[match[0].queryIdx].pt[1], 1, 1])  # 将特征点扩展为4维齐次坐标
        pt2 = np.array([kp2[match[0].trainIdx].pt[0], kp2[match[0].trainIdx].pt[1], 1])
        
        # 使用变换矩阵将 pt1 映射到 pt2 的坐标系中
        transformed_pt1 = np.dot(transformation_matrix, pt1)
        transformed_pt1 /= transformed_pt1[2]  # 将齐次坐标转换为非齐次坐标
        
        # 计算映射后的特征点与实际特征点之间的欧氏距离
        distance = np.linalg.norm(transformed_pt1[:2] - pt2[:2])
        
        # 如果距离小于阈值，则认为这对匹配是有效的
        if distance < threshold:
            filtered_matches.append(match)

    
    return len(filtered_matches) 

def evaluate(gt_txt):
    match_points = []
    fp = open(gt_txt, "r")
    for line in tqdm(fp):
        line_str = line.strip().split(", ")
        query, reference = line_str[0], line_str[1]
        match_points.append(loop_verification(query, reference, new_size=None, is_vis=False))
    return np.array(match_points)

if __name__ == '__main__':
    # visualization
    # loop_verification("Kudamm_mini_query/image0197.jpg", "Kudamm_mini_ref/image0174.jpg", new_size=None, is_vis=True)

    print("-------- Processing robotcar_qAutumn_dbNight_val_final ----------")
    match_points = evaluate("robotcar_qAutumn_dbNight_val_final.txt")
    scaled_scores = match_points / max(match_points)
    threshold = 0.4
    # 根据阈值将 match_points 转换为 results
    results = np.where(scaled_scores > threshold, 1, 0)
    # 将 results 保存到文件中
    with open("robotcar_qAutumn_dbNight_val_result.txt", "w") as f:
        for result in results:
            f.write(str(result) + "\n")
    print("Results have been saved to robotcar_qAutumn_dbNight_val_result.txt")


    print("-------- Processing robotcar_qAutumn_dbSunCloud_val_final ----------")
    match_points = evaluate("robotcar_qAutumn_dbSunCloud_val_final.txt")
    scaled_scores = match_points / max(match_points)
    threshold = 0.4
    # 根据阈值将 match_points 转换为 results
    results = np.where(scaled_scores > threshold, 1, 0)
    # 将 results 保存到文件中
    with open("robotcar_qAutumn_dbSunCloud_val_result.txt", "w") as f:
        for result in results:
            f.write(str(result) + "\n")
    print("Results have been saved to robotcar_qAutumn_dbSunCloud_val_result.txt")
    

    
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.legend()
    # plt.title("Precision-Recall Curves for ORB baseline")
    # plt.savefig("pr_curve_ORB.png")








    
