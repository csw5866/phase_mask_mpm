import trimesh
import numpy as np
import warp as wp
from .data_types import Particle
'''
def create_block(block_min, block_max, spacing, density, ind_cfg):
    particles = []

    cell_volume = spacing ** 3
    p_volume = cell_volume
    p_mass = density * p_volume

    nx = int((block_max[0] - block_min[0]) / spacing) + 1
    ny = int((block_max[1] - block_min[1]) / spacing) + 1
    nz = int((block_max[2] - block_min[2]) / spacing) + 1

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = block_min + spacing * np.array([i, j, k])

                p = Particle()
                p.x = wp.vec3(*x)
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
''' 
# 상단 코드는 flat_surface용!!

## rtree 갑ㄴ일 시 매우 오래 걸림(특히 mla구조의 경우 반구형태에 수만개의 trimesh가 존재하므로 더 오래걸림. -->> gpu이용 시 warp의 wp.Mesh객체의 wp.mesh_query_point 사용 시 빨리 끝낼 수 있음)


def create_block(stl_path, spacing, density,ind_cfg):
    # 1. STL 로드
    mesh = trimesh.load(stl_path)
    
    # 2. 입자가 생성될 후보 그리드 생성 (Bounding Box 기준)
    min_bound, max_bound = mesh.bounds
    x = np.arange(min_bound[0], max_bound[0], spacing)
    y = np.arange(min_bound[1], max_bound[1], spacing)
    z = np.arange(min_bound[2], max_bound[2], spacing)
    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
    
    # 3. [핵심] 메시 내부에 포함된 점들만 필터링
    # trimesh의 contains 기능을 쓰면 균일 밀도가 보장됩니다.
    inside = mesh.contains(grid)
    final_points = grid[inside]
    
    # 4. 일관된 질량/부피 할당
    p_volume = spacing ** 3
    p_mass = density * p_volume
    
    particles = []
    for pos in final_points:
        p = Particle() #
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