import os
import numpy as np
import open3d as o3d

def pcd_to_numpy(pcd_file, output_file):
    # PCD 파일 읽기
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    # 포인트 좌표 추출
    points = np.asarray(pcd.points)
    
    # 강도 정보 설정 (강도 정보가 없는 경우 0으로 설정)
    num_points = points.shape[0]
    intensity = np.zeros((num_points, 1))  # 강도 정보가 없는 경우 0으로 설정
    
    # 실제 강도 정보가 있는 경우, 해당 정보를 추출하여 사용하세요.
    # 예시: 색상에서 강도 추출 (RGB의 밝기를 강도로 사용)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        intensity = np.linalg.norm(colors, axis=1, keepdims=True)
    
    # 강도 정규화 (0~1)
    if np.max(intensity) > 0:
        intensity = intensity / np.max(intensity)
    
    # x, y, z, intensity로 구성된 배열 생성
    points = np.hstack((points, intensity))
    
    # NumPy 파일로 저장
    np.save(output_file, points)

def convert_all_pcd_to_numpy(input_base_dir, output_base_dir):
    """
    모든 시나리오의 PCD 파일을 NumPy 파일로 변환합니다.
    Args:
        input_base_dir (str): 입력 PCD 파일들이 있는 기본 디렉토리.
        output_base_dir (str): 출력 NumPy 파일들을 저장할 기본 디렉토리.
    """
    # 입력 디렉토리 내 모든 시나리오 디렉토리 순회
    for scenario in os.listdir(input_base_dir):
        scenario_input_dir = os.path.join(input_base_dir, scenario, 'pcd')
        scenario_output_dir = os.path.join(output_base_dir, scenario)
        
        # 출력 디렉토리가 없으면 생성
        os.makedirs(scenario_output_dir, exist_ok=True)
        
        # 각 시나리오의 모든 .pcd 파일 순회
        for file_name in os.listdir(scenario_input_dir):
            if file_name.endswith('.pcd'):
                pcd_path = os.path.join(scenario_input_dir, file_name)
                npy_file_name = file_name.replace('.pcd', '.npy')
                npy_path = os.path.join(scenario_output_dir, npy_file_name)
                
                # 변환 수행
                pcd_to_numpy(pcd_path, npy_path)
                print(f'변환 완료: {pcd_path} -> {npy_path}')

if __name__ == "__main__":
    # 입력 및 출력 기본 디렉토리 설정
    input_base_dir = './data'          # 입력 PCD 파일들이 있는 기본 디렉토리
    output_base_dir = './numpy_data'   # 출력 NumPy 파일들을 저장할 기본 디렉토리
    
    convert_all_pcd_to_numpy(input_base_dir, output_base_dir)
    print('모든 PCD 파일의 변환이 완료되었습니다.')