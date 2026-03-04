import numpy as np
import warp as wp
import trimesh
import os
from .data_types import Particle

# Warp가 초기화되지 않았을 때만 초기화 수행 (main.py에서 이미 했다면 무시됨)
if not wp.is_initialized():
    wp.init()

# --- [GPU 커널 정의] ---
@wp.kernel
def check_inside_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    inside_mask: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    p = points[tid]
    
    # wp.mesh_query_point는 GPU 기반의 매우 빠른 공간 쿼리입니다.
    # 수천 개의 삼각형 중 가장 가까운 면을 찾고 부호 있는 거리(SDF)를 계산합니다.
    # 세 번째 인자(max_dist)는 충분히 크게 설정합니다.
    query = wp.mesh_query_point(mesh_id, p, 1e6)
    
    if query.result:
        # Warp의 mesh query에서 dist가 음수이면 메쉬 내부(Inside)를 의미합니다.
        if query.dist <= 0.0:
            inside_mask[tid] = 1



# --- [함수 구현] ---
def create_block(stl_path, spacing, density, ind_cfg):
    """
    기존 시그니처를 유지하면서 NVIDIA Warp GPU 가속을 사용하여 
    입자를 생성하는 함수입니다.
    """
    # 1. STL 로드 (trimesh를 사용하여 데이터를 먼저 읽습니다)
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL 파일을 찾을 수 없습니다: {stl_path}")
    
    mesh_raw = trimesh.load(stl_path)
    
    # 2. 입자가 생성될 후보 그리드 생성 (기존 로직 유지)
    min_bound, max_bound = mesh_raw.bounds
    x_range = np.arange(min_bound[0], max_bound[0], spacing)
    y_range = np.arange(min_bound[1], max_bound[1], spacing)
    z_range = np.arange(min_bound[2], max_bound[2], spacing)
    
    grid = np.stack(np.meshgrid(x_range, y_range, z_range, indexing='ij'), axis=-1).reshape(-1, 3)
    
    # 3. [GPU 가속] 공간 가속 구조 구축 및 내부 판정
    # Warp Mesh 객체 생성 (GPU 메모리로 전송)
    wp_mesh = wp.Mesh(
        points=wp.array(mesh_raw.vertices, dtype=wp.vec3),
        indices=wp.array(mesh_raw.faces.flatten(), dtype=wp.int32)
    )
    
    # 데이터를 Warp 배열로 변환
    points_gpu = wp.array(grid, dtype=wp.vec3)
    inside_mask_gpu = wp.zeros(len(grid), dtype=wp.int32)
    
    # 커널 실행 (병렬 처리)
    wp.launch(
        kernel=check_inside_kernel,
        dim=len(grid),
        inputs=[wp_mesh.id, points_gpu, inside_mask_gpu]
    )
    
    # 결과를 CPU로 다시 가져오기
    inside = inside_mask_gpu.numpy().astype(bool)
    final_points = grid[inside]
    
    # 4. 일관된 질량/부피 할당 및 Particle 객체 생성 (기존 로직 유지)
    p_volume = spacing ** 3
    p_mass = density * p_volume
    
    particles = []
    
    # 파이썬 리스트로 변환하는 과정은 CPU에서 이루어집니다.
    # (여기서 mat_id 등 구조체 초기화)
    for pos in final_points:
        p = Particle() # Particle 클래스가 정의되어 있다고 가정합니다.
        p.x = wp.vec3(*pos)
        p.v = wp.vec3(0.0, 0.0, 0.0)
        p.F = wp.mat33(1.0, 0.0, 0.0, 
                       0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0)                
        p.C = wp.mat33(0.0)
        p.mass = p_mass
        p.volume = p_volume
        p.mat_id = 0
        
        particles.append(p)

# --- 2. Indenter Particles (mat_id: 1) ---
    ind_center = np.array(ind_cfg['center'])
    ind_radius = float(ind_cfg['radius'])
    indenter_v = np.array(ind_cfg['velocity'])
    
    # 인덴터 표면과 내부를 채울 입자 간격 (엘라스토머보다 조밀하게)
    ind_sp = spacing * 0.8 
    
    # 인덴터를 감싸는 가상의 박스(Bounding Box) 범위 계산
    ind_min = ind_center - ind_radius
    ind_nx = int((2.0 * ind_radius) / ind_sp) + 1
    ind_ny = int((2.0 * ind_radius) / ind_sp) + 1
    ind_nz = int((2.0 * ind_radius) / ind_sp) + 1

    for i in range(ind_nx):
        for j in range(ind_ny):
            for k in range(ind_nz):
                # 가상 박스 내의 입자 후보 좌표 계산
                pos = ind_min + ind_sp * np.array([i, j, k])
                
                # 구의 중심으로부터의 거리가 반지름 이내인 경우만 입자로 생성
                if np.linalg.norm(pos - ind_center) <= ind_radius:
                    p = Particle()
                    p.x = wp.vec3(*pos)
                    # 인덴터의 초기 속도는 YAML 설정에 따라 Solver에서 업데이트될 수 있습니다.
                    p.v = wp.vec3(*indenter_v) 
                    p.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                    p.C = wp.mat33(0.0)
                    # 인덴터는 강체성을 위해 보통 엘라스토머보다 큰 질량을 부여합니다.
                    p.mass = p_mass * 10.0 
                    p.volume = p_volume
                    p.mat_id = 1  # Rigid Indenter ID
                    particles.append(p)
        
    return particles