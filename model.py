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

def preprocess_point_cloud(pcd_path, voxel_size=0.1, nb_points=20, radius=0.3, 
                          distance_threshold=0.1, ransac_n=3, num_iterations=100, normal_radius=0.1, normal_max_nn=30):
    """
    포인트 클라우드를 로드하고 전처리(다운샘플링, 이상치 제거, 평면 분할, 법선 추정)를 수행합니다.
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_filtered, _ = pcd_down.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
    plane_model, road_inliers = pcd_filtered.segment_plane(distance_threshold=distance_threshold, 
                                                           ransac_n=ransac_n, 
                                                           num_iterations=num_iterations)
    pcd_no_plane = pcd_filtered.select_by_index(road_inliers, invert=True)
    return pcd_no_plane

def cluster_and_bbox(pcd, min_cluster_size=50, max_cluster_size=500):
    """
    포인트 클라우드를 클러스터링하고 바운딩 박스를 생성합니다.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(np.array(pcd.points))
    labels = clusterer.labels_
    
    max_label = labels.max()
    print(f'포인트 클라우드에 {max_label + 1}개의 클러스터가 있습니다.')
    
    # 색상 맵 적용
    if max_label > 0:
        colors = plt.get_cmap("tab20")(labels / max_label)
    else:
        colors = plt.get_cmap("tab20")(labels)
    colors[labels < 0] = 0  # 노이즈 포인트는 검은색
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # 바운딩 박스 생성
    bbox_objects = []
    indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()
    
    for i, idx in enumerate(indexes):
        if labels[i] == -1:
            continue  # 노이즈는 제외
        nb_points = len(pcd.select_by_index(idx).points)
        if 50 < nb_points < max_cluster_size:
            sub_cloud = pcd.select_by_index(idx)
            bbox = sub_cloud.get_axis_aligned_bounding_box()
            bbox.color = (0, 0, 1)  # 파란색
            bbox_objects.append(bbox)
    
    print(f"생성된 바운딩 박스 수: {len(bbox_objects)}")
    return pcd, bbox_objects

def register_generalized_icp(source, target, threshold=0.3, trans_init=np.identity(4)):
    """
    Generalized ICP를 사용하여 소스와 타겟 포인트 클라우드를 정합합니다.
    """
    reg = o3d.pipelines.registration.registration_generalized_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())
    return reg

def visualize_registration(source, target, transformation=np.identity(4)):
    """
    소스와 타겟 포인트 클라우드를 정합된 상태로 시각화합니다.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])  # 빨간색
    target_temp.paint_uniform_color([0, 0, 1])  # 파란색
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name="Registration Check")

def main():
    # 입력 PCD 파일들이 저장된 디렉토리 경로 설정
    input_pcd_dir = "data/02_straight_duck_walk/pcd"  # 실제 PCD 파일들이 저장된 경로로 변경하세요
    
    # 출력 디렉토리 설정
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 비디오 프레임 저장 디렉토리 생성
    frames_dir = os.path.join(output_dir, "video_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # 비디오 파일 경로 설정
    video_path = os.path.join(output_dir, "point_cloud_video.avi")
    
    # 최종 정합된 포인트 클라우드 파일 경로 설정
    final_pcd_path = os.path.join(output_dir, "combined_point_cloud.pcd")
    
    # PCD 파일 목록 정렬 (예: pcd_000001.pcd부터 pcd_00288.pcd까지)
    pcd_files = sorted([f for f in os.listdir(input_pcd_dir) if f.endswith('.pcd')])
    if len(pcd_files) == 0:
        print(f"디렉토리 {input_pcd_dir}에 PCD 파일이 없습니다.")
        sys.exit(1)
    
    print(f"총 {len(pcd_files)}개의 PCD 파일을 발견했습니다.")
    
    # 첫 번째 PCD 파일을 기준 프레임으로 설정
    reference_pcd_path = os.path.join(input_pcd_dir, pcd_files[0])
    reference_pcd = preprocess_point_cloud(reference_pcd_path)
    reference_pcd, reference_bboxes = cluster_and_bbox(reference_pcd)
    
    # 기준 프레임을 글로벌 포인트 클라우드에 추가
    combined_pcd = copy.deepcopy(reference_pcd)
    
    # 시각화 창 설정 (백그라운드에서 실행)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(combined_pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # 포인트 크기 조정
    render_option.background_color = np.asarray([0, 0, 0])  # 배경을 검은색으로 설정
    
    # 프레임 1부터 시작 (이미 기준 프레임)
    for i in tqdm(range(1, len(pcd_files)), desc="Processing PCD files"):
        source_pcd_path = os.path.join(input_pcd_dir, pcd_files[i])
        print(f"\nProcessing {source_pcd_path} ...")
        
        # 소스 포인트 클라우드 전처리
        source_pcd = preprocess_point_cloud(source_pcd_path)
        source_pcd, source_bboxes = cluster_and_bbox(source_pcd)
        
        # Generalized ICP 정합 수행: 소스를 기준 프레임에 정합
        reg_icp = register_generalized_icp(source_pcd, reference_pcd, threshold=0.3, trans_init=np.identity(4))
        
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
        
        # 소스 포인트 클라우드에 정합 변환 적용
        source_pcd.transform(reg_icp.transformation)
        
        # 글로벌 포인트 클라우드에 추가
        combined_pcd += source_pcd
        
        # 시각화 업데이트
        vis.clear_geometries()
        vis.add_geometry(combined_pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # 화면 캡처
        image = vis.capture_screen_float_buffer(do_render=True)
        image = (255 * np.asarray(image)).astype(np.uint8)
        image_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # 시각화 창 닫기
    vis.destroy_window()
    
    # 최종 정합된 포인트 클라우드 저장
    o3d.io.write_point_cloud(final_pcd_path, combined_pcd)
    print(f"최종 정합된 포인트 클라우드가 저장되었습니다: {final_pcd_path}")
    
    # 비디오 생성
    print("비디오를 생성 중입니다...")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frame_files:
        print("저장된 프레임 이미지가 없습니다.")
        sys.exit(1)
    
    # 첫 번째 프레임을 사용하여 비디오 설정
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 또는 'mp4v'로 변경 가능
    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        video.write(frame)
    
    video.release()
    print(f"비디오가 성공적으로 생성되었습니다: {video_path}")
    
    # 최종 PCD 파일 시각화 (옵션)
    visualize_registration(combined_pcd, reference_pcd, np.identity(4))

if __name__ == "__main__":
    main()
