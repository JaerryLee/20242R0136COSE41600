import open3d as o3d

def visualize_final_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd], window_name="Final Combined Point Cloud")

if __name__ == "__main__":
    final_pcd_path = "output_results/combined_point_cloud.pcd"  # 최종 PCD 파일 경로로 변경
    visualize_final_pcd(final_pcd_path)