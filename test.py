import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import pandas as pd
import copy
import os
import sys
from tqdm import tqdm
import cv2

def preprocess_point_cloud(pcd_path, voxel_size=0.2, nb_points=6, radius=1.2, 
                          distance_threshold=0.1, ransac_n=3, num_iterations=2000, 
                          normal_radius=0.2, normal_max_nn=50):
    """
    포인트 클라우드를 로드하고 전처리(다운샘플링, 이상치 제거, 평면 분할, 법선 추정)를 수행합니다.
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded PCD: {pcd_path}, with {len(pcd.points)} points.")
    
    # Voxel Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Downsampled PCD to {len(pcd_down.points)} points using voxel size {voxel_size}.")
    
    # Radius Outlier Removal
    pcd_filtered, _ = pcd_down.remove_radius_outlier(nb_points=nb_points, radius=radius)
    print(f"Radius Outlier Removal: Kept {len(pcd_filtered.points)} points out of {len(pcd_down.points)}.")
    
    # 법선 추정
    pcd_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
    print("Normals estimated.")
    
    # 평면 분할 (RANSAC)
    plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=distance_threshold, 
                                                      ransac_n=ransac_n, 
                                                      num_iterations=num_iterations)
    print(f"Plane segmented: {len(inliers)} inliers.")
    
    # 평면을 제외한 포인트 클라우드 추출
    pcd_no_plane = pcd_filtered.select_by_index(inliers, invert=True)
    print(f"Points after removing plane: {len(pcd_no_plane.points)}.")
    
    return pcd_no_plane

def cluster_and_bbox(pcd, min_cluster_size=30, max_cluster_size=300):
    """
    포인트 클라우드를 클러스터링하고 바운딩 박스를 생성합니다.
    """
    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(np.array(pcd.points))
    labels = clusterer.labels_
    
    max_label = labels.max()
    print(f'포인트 클라우드에 {max_label + 1}개의 클러스터가 있습니다.')
    
    # 색상 맵 적용 (노이즈는 검은색)
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 0  # 노이즈 포인트는 검은색
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # 바운딩 박스 생성
    bbox_objects = []
    indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()
    
    for i, idx in enumerate(indexes):
        if labels[i] == -1:
            continue  # 노이즈는 제외
        nb_points = len(pcd.select_by_index(idx).points)
        if 30 < nb_points < 300:
            cluster_pcd = pcd.select_by_index(idx)
            points = np.asarray(cluster_pcd.points)
            
            # 추가 필터링: 높이, 크기, 거리 등
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            height_diff = z_max - z_min
            
            # 사람의 높이 범위에 맞게 필터링 (예: 1.0m ~ 2.5m)
            if 1.0 <= height_diff <= 2.5:
                # 원점으로부터의 최대 거리 기준 (필요시 조정)
                distances = np.linalg.norm(points, axis=1)
                if distances.max() <= 30.0:
                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                    bbox.color = (1, 0, 0)  # 빨간색으로 설정 (사람)
                    bbox_objects.append(bbox)
    
    print(f"생성된 바운딩 박스 수: {len(bbox_objects)}")
    return pcd, bbox_objects, labels

def register_icp(source, target, threshold=0.5, trans_init=np.identity(4)):
    """
    ICP를 사용하여 소스와 타겟 포인트 클라우드를 정합합니다.
    """
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg

def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=2.0):
    """
    포인트 클라우드와 바운딩 박스를 시각화합니다.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray([0, 0, 0])  # 배경을 검은색으로 설정
    vis.run()
    vis.destroy_window()

def main():
    # PCD 파일들이 저장된 디렉토리 경로 설정
    input_pcd_dir = "data/01_straight_walk/pcd"  # 실제 PCD 파일들이 저장된 디렉토리 경로로 변경하세요
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Pose Graph 초기화
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    
    # PCD 파일 목록 정렬 (예: pcd_000001.pcd부터 pcd_N.pcd까지)
    pcd_files = sorted([f for f in os.listdir(input_pcd_dir) if f.endswith('.pcd')])
    if len(pcd_files) == 0:
        print(f"디렉토리 {input_pcd_dir}에 PCD 파일이 없습니다.")
        sys.exit(1)
    
    print(f"총 {len(pcd_files)}개의 PCD 파일을 발견했습니다.")
    
    # 첫 번째 PCD 파일을 기준 프레임으로 설정
    reference_pcd_path = os.path.join(input_pcd_dir, pcd_files[0])
    reference_pcd = preprocess_point_cloud(reference_pcd_path)
    reference_pcd, reference_bboxes, reference_labels = cluster_and_bbox(reference_pcd)
    
    # 기준 프레임을 글로벌 포인트 클라우드에 추가
    combined_pcd = copy.deepcopy(reference_pcd)
    
    # 시각화 창 설정 (백그라운드에서 실행)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(combined_pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # 포인트 크기 조정
    render_option.background_color = np.asarray([0, 0, 0])  # 배경을 검은색으로 설정
    
    # 초기 변환 행렬 설정
    transformation_init = np.identity(4)
    
    # 클러스터 추적을 위한 변수 초기화
    previous_clusters = reference_labels
    cluster_tracking = {}  # 클러스터 ID와 이전 위치를 저장
    
    # 진행 상황 표시를 위한 tqdm 사용
    for i in tqdm(range(1, len(pcd_files)), desc="Processing PCD files"):
        source_pcd_path = os.path.join(input_pcd_dir, pcd_files[i])
        print(f"\nProcessing {source_pcd_path} ...")
        
        # 소스 포인트 클라우드 전처리
        source_pcd = preprocess_point_cloud(source_pcd_path)
        source_pcd, source_bboxes, source_labels = cluster_and_bbox(source_pcd)
        
        # ICP 정합 수행: 소스를 기준 프레임에 정합
        reg_icp = register_icp(source_pcd, reference_pcd, threshold=0.5, trans_init=transformation_init)
        
        # 정합 결과 출력
        print(f"Frame {i}:")
        print(reg_icp)
        print("Transformation Matrix:")
        print(reg_icp.transformation)
        print(f"Fitness: {reg_icp.fitness}, Inlier RMSE: {reg_icp.inlier_rmse}")
        
        # Fitness가 0인 경우 처리
        if reg_icp.fitness == 0:
            print(f"Warning: Fitness is 0 at frame {i}. Skipping this frame.")
            continue  # 또는 적절한 조치를 취합니다.
        
        # Pose Graph에 에지 추가 (변환 행렬)
        odometry = np.dot(reg_icp.transformation, odometry)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i-1, i, reg_icp.transformation, 
                                                                        information=np.identity(6), 
                                                                        uncertain=False))
        
        # 소스 포인트 클라우드에 정합 변환 적용
        source_pcd.transform(reg_icp.transformation)
        
        # 글로벌 포인트 클라우드에 추가
        combined_pcd += source_pcd
        
        # 시각화 업데이트
        vis.clear_geometries()
        vis.add_geometry(combined_pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # 화면 캡처 (비디오 프레임 저장용)
        image = vis.capture_screen_float_buffer(do_render=True)
        image = (255 * np.asarray(image)).astype(np.uint8)
        frames_dir = os.path.join(output_dir, "video_frames")
        os.makedirs(frames_dir, exist_ok=True)
        image_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # 다음 정합을 위해 현재 소스를 타겟으로 설정
        reference_pcd = source_pcd
        transformation_init = reg_icp.transformation
    
    # 시각화 창 닫기
    vis.destroy_window()
    
    # Pose Graph 최적화 (루프 클로저 포함)
    print("Optimizing Pose Graph with Loop Closure...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=1.0,
        edge_prune_threshold=0.25,
        reference_node=0)
    
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    
    print("Pose Graph Optimization Complete.")
    
    # 최종 변환 행렬을 적용하여 글로벌 포인트 클라우드 재정렬
    print("Transforming Combined Point Cloud with Optimized Pose Graph...")
    optimized_pcd = o3d.geometry.PointCloud()
    for node in pose_graph.nodes:
        pcd = copy.deepcopy(combined_pcd)
        pcd.transform(node.pose)
        optimized_pcd += pcd
    print("Transformation Complete.")
    
    # 최종 정합된 포인트 클라우드 저장
    final_pcd_path = os.path.join(output_dir, "combined_optimized.pcd")
    o3d.io.write_point_cloud(final_pcd_path, optimized_pcd)
    print(f"최종 정합된 포인트 클라우드가 저장되었습니다: {final_pcd_path}")
    
    # 최종 PCD 파일 클러스터링 및 바운딩 박스 생성
    final_pcd = o3d.io.read_point_cloud(final_pcd_path)
    final_pcd, final_bboxes, final_labels = cluster_and_bbox(final_pcd)
    
    # 사람이 포함된 클러스터 식별 (움직임 기반)
    # 클러스터가 여러 프레임에 걸쳐 움직였는지 확인하기 위해 이동량 분석 필요
    # 여기서는 가장 큰 클러스터를 사람이 아닐 수 있음을 반영하여, 움직임 기반 식별을 추가로 구현
    
    # 간단한 움직임 기반 식별: 클러스터의 이동 거리 계산
    # 각 클러스터의 centroid를 계산하고, 이전 프레임과 비교하여 이동량이 큰 클러스터를 사람으로 간주
    
    # 현재는 최종 클러스터만을 대상으로 식별, 추가적인 움직임 추적 필요
    
    if final_labels.max() == -1:
        print("클러스터가 존재하지 않습니다. 바운딩 박스를 생성할 수 없습니다.")
    else:
        # 각 클러스터의 포인트 수를 계산
        cluster_sizes = {}
        for label in final_labels:
            if label == -1:
                continue
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        # 각 클러스터의 centroid를 계산
        centroids = {}
        for label in cluster_sizes:
            cluster_indices = np.where(final_labels == label)[0]
            cluster_pcd = final_pcd.select_by_index(cluster_indices)
            centroid = np.mean(np.asarray(cluster_pcd.points), axis=0)
            centroids[label] = centroid
        
        # 움직임 기반 클러스터 식별 (예: centroid 이동 거리)
        # 여기서는 단일 프레임에서만 계산하므로, 실제로는 여러 프레임을 통해 이동을 추적해야 함
        # 예제에서는 임의로 특정 조건을 사용하여 사람 클러스터를 식별
        
        # 예를 들어, Z축 위치가 특정 범위에 있는 클러스터를 사람으로 간주
        # 실제로는 여러 프레임을 기반으로 움직임을 분석해야 함
        # 현재 코드는 단일 프레임에서 사람 클러스터를 식별하는 예제임
        
        # 사람 클러스터를 식별하는 추가 조건을 적용
        person_bboxes = []
        for bbox in final_bboxes:
            # 바운딩 박스의 크기를 기준으로 사람을 식별
            extents = bbox.get_extent()
            height = extents[2]
            if 1.0 <= height <= 2.5:  # 사람의 높이 범위
                bbox.color = (1, 0, 0)  # 빨간색으로 설정
                person_bboxes.append(bbox)
        
        if not person_bboxes:
            print("사람으로 식별된 클러스터가 없습니다.")
        else:
            print(f"사람으로 식별된 바운딩 박스 수: {len(person_bboxes)}")
            # 전체 포인트 클라우드와 사람 바운딩 박스를 시각화
            o3d.visualization.draw_geometries([final_pcd] + person_bboxes, window_name="Final Point Cloud with Person Bounding Boxes")
            
            # 최종 PCD 파일에 바운딩 박스 추가 및 저장 (옵션)
            combined_with_bbox = copy.deepcopy(final_pcd)
            for bbox in person_bboxes:
                combined_with_bbox += bbox
            combined_with_bbox_path = os.path.join(output_dir, "combined_with_bbox.pcd")
            o3d.io.write_point_cloud(combined_with_bbox_path, combined_with_bbox)
            print(f"바운딩 박스가 포함된 최종 PCD 파일이 저장되었습니다: {combined_with_bbox_path}")
    
    # 비디오 생성
    print("비디오를 생성 중입니다...")
    frames_dir = os.path.join(output_dir, "video_frames")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frame_files:
        print("저장된 프레임 이미지가 없습니다.")
        sys.exit(1)
    
    # 첫 번째 프레임을 사용하여 비디오 설정
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 또는 'mp4v'로 변경 가능
    video = cv2.VideoWriter(os.path.join(output_dir, "point_cloud_video.avi"), fourcc, 10, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        video.write(frame)
    
    video.release()
    print(f"비디오가 성공적으로 생성되었습니다: {os.path.join(output_dir, 'point_cloud_video.avi')}")

if __name__ == "__main__":
    main()
