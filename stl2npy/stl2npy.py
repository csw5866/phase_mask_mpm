from turtle import pos

import numpy as np
import warp as wp
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
refer to indentation.yaml,

width (x-dir) = res[0] * pixel_size = 512 * 2.6e-6 = 1.3312 mm
length (z-dir) = res[1] * pixel_size = 512 * 2.6e-6 = 1.3312 mm
h_tot (y-dir elastomer height) = profile_hmax + thickness = 5 um + 50 um = 55 um = 0.055 mm
dx = pixel_size * sampling = 2.6e-6 * 3 = 7.8e-6 m = 0.0078 mm
self.ny = int(h_tot * 2.0 * self.inv_dx) + 2 = int(0.055 * 2.0 / 0.0078) + 2 = 14 + 2 = 16
valid max_height = (self.ny - 1) * dx = 15 * 0.0078 = 0.117 mm 
single_step delta_y = indenter_velocity * dt = 0.2 m/s * 5e-8 s = 1e-8 m = 0.00001 mm
hundred_steps delta_y = 1e-7 m = 0.001 mm          (if step indenter is used, about 0.006 mm difference would be good for visualization)
total 2000 steps delta_y = 1e-6 m = 0.02 mm

id_spacing = pixel_size * 0.8 = 2.6e-6 * 0.8 = 2.08e-6 m = 0.00208 mm   (used as resolution for stl sampling)

resolution in this script is the spacing between sampled points, which should be smaller than the MPM grid spacing (dx) to ensure we capture the geometry accurately.
Don't get confused with the res in indentation.yaml, which is the grid resolution of the input heightmap (512x512) and is used to determine the physical size of the simulation domain.

*** If you design the stl model, make sure that your minimum feature height (y-value) is larger than the h_tot (55 um) to ensure it is located above the elastomer
*** Also, stle model should be located near the (x=0, z=0) for better simulation results
*** it is also recommended to design unwanted parts extending out of the valid limits to reduce the number of particles
'''

# 2. 표면 샘플링 커널
@wp.kernel
def sample_surface_kernel(
    mesh: wp.uint64,
    grid_origin: wp.vec3,
    spacing: float,
    dim_x: int, dim_y: int, dim_z: int,
    max_y: float,                       # valid_max_height
    threshold: float,
    output_pos: wp.array(dtype=wp.vec3),
    output_mask: wp.array(dtype=int)
):
    # 격자 인덱스 계산
    tid = wp.tid()
    iz = tid // (dim_x * dim_y)
    iy = (tid // dim_x) % dim_y
    ix = tid % dim_x

    # 실제 공간 좌표 계산
    pos = grid_origin + wp.vec3(float(ix) * spacing, float(iy) * spacing, float(iz) * spacing)

    # 1. 사용자 지정 높이 제한 (Upper Bound)
    # x_min, z_min can be also applied using similar logic if needed (receive parameters to check if pos.x and pos.z are within valid range)
    if pos.y <= max_y:
        # --- [Warp 1.12.0 시그니처 대응] ---
        max_dist = wp.float32(threshold) # 검색 반경
        inside = wp.float32(0.0)         # inside 결과가 담길 변수
        face_id = wp.int32(-1)           # 가장 가까운 삼각형 인덱스
        u = wp.float32(0.0)              # 무게중심 좌표 u
        v = wp.float32(0.0)              # 무게중심 좌표 v
        
        # 7개의 인자를 정확한 타입으로 전달
        if wp.mesh_query_point(mesh, pos, max_dist, inside, face_id, u, v):
            # 쿼리가 true를 반환했다면 threshold 내에 표면이 존재한다는 뜻
            output_pos[tid] = pos
            output_mask[tid] = 1



def sample_stl_surface(stl_path, resolution, max_y, device="cuda:0"):
    wp.init()

    mesh = trimesh.load(stl_path)
    mesh.apply_scale(0.001)         # mm to m (assuming STL is designed in mm, adjust if needed)
    # 삼각형 데이터 준비
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Warp Mesh 객체 생성 (GPU 메모리에 업로드)
    wp_mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(faces.flatten(), dtype=int, device=device)
    )
    # 격자 영역 (Bounds) 설정
    bounds = mesh.bounds
    print(f"Mesh Bounds: {mesh.bounds}")
    padding = resolution * 1.0
    origin = wp.vec3(bounds[0][0] - padding, bounds[0][1] - padding, bounds[0][2] - padding)
    end = wp.vec3(bounds[1][0] + padding, bounds[1][1] + padding, bounds[1][2] + padding)
    
    dims = (
        int((end[0] - origin[0]) / resolution) + 1,
        int((end[1] - origin[1]) / resolution) + 1,
        int((end[2] - origin[2]) / resolution) + 1
    )
    total_cells = dims[0] * dims[1] * dims[2]
    print(f"Grid Dimensions: {dims}, Total points to check: {total_cells}")

    # 메모리 할당
    out_pos = wp.zeros(total_cells, dtype=wp.vec3, device=device)
    out_mask = wp.zeros(total_cells, dtype=int, device=device)

    # GPU 커널 실행
    wp.launch(
        kernel=sample_surface_kernel,
        dim=total_cells,
        inputs=[wp_mesh.id, origin, resolution, dims[0], dims[1], dims[2], max_y, resolution * 0.6, out_pos, out_mask],
        device=device
    )

    # 결과 가져오기
    mask_np = out_mask.numpy()
    pos_np = out_pos.numpy()
    surface_particles = pos_np[mask_np == 1]
    
    print(f"Sampling Completed. Particles generated: {len(surface_particles)}")
    return surface_particles
    

def visualize_particles(particles, title="Sampled Surface Particles"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 입자 수가 너무 많으면 일부만 샘플링하여 시각화 속도 향상
    if len(particles) > 20000:
        idx = np.random.choice(len(particles), 20000, replace=False)
        p_plot = particles[idx]
        print("Note: Plotting 20,000 points for performance.")
    else:
        p_plot = particles

    ax.scatter(p_plot[:, 0], p_plot[:, 1], p_plot[:, 2], s=1, c=p_plot[:, 2], cmap='viridis', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 축 비율을 실제 데이터 비율에 맞춤
    max_range = np.array([p_plot[:,0].max()-p_plot[:,0].min(), 
                          p_plot[:,1].max()-p_plot[:,1].min(), 
                          p_plot[:,2].max()-p_plot[:,2].min()]).max() / 2.0
    mid_x = (p_plot[:,0].max()+p_plot[:,0].min()) * 0.5
    mid_y = (p_plot[:,1].max()+p_plot[:,1].min()) * 0.5
    mid_z = (p_plot[:,2].max()+p_plot[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()



stl_file = "pyramid.stl" 

res = 2.6e-6 * 0.8  
max_y = 1.17e-4 
sampled_pts = sample_stl_surface(stl_file, resolution=res, max_y=max_y)
        
# 시각화 실행
visualize_particles(sampled_pts)
