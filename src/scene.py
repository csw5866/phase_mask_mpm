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

def create_block(block_min, block_max, spacing, density, profile_cfg, ind_cfg):
    
    p_volume = spacing ** 3
    p_mass = density * p_volume
    
    # 2. MLA 중심 좌표 미리 계산
    mla_centers = []
    if profile_cfg['enabled']:
        N = profile_cfg['grid_count']
        R = profile_cfg['radius']
        x_start = block_min[0]
        x_end = block_max[0]
        z_start = block_min[2]
        z_end = block_max[2]
        
        # 렌즈 간 간격이 딱 붙도록 배치
        for i in range(N):
            for j in range(N):
                cx = x_start + R + (i * 2 * R)
                cz = z_start + R + (j * 2 * R)
                # 반구의 바닥면 중심은 블록의 상단면(b_max[1])
                mla_centers.append(np.array([cx, block_max[1], cz]))

    # 3. SDF 샘플링을 위한 전체 바운딩 박스 설정
    scan_min = block_min
    scan_max = block_max.copy()
    if profile_cfg['enabled']:
        scan_max[1] += profile_cfg['radius'] # 블록 높이 + 렌즈 반지름까지 스캔

    nx = int((scan_max[0] - scan_min[0]) / spacing) + 1
    ny = int((scan_max[1] - scan_min[1]) / spacing) + 1
    nz = int((scan_max[2] - scan_min[2]) / spacing) + 1

    particles = []
    
    # 정규 격자 스캔 (Uniform Density 보장)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pos = scan_min + spacing * np.array([i, j, k])
                
                is_inside = False
                
                # [SDF 1] 베이스 블록 내부 판정
                if (block_min[0] <= pos[0] <= block_max[0] and 
                    block_min[1] <= pos[1] <= block_max[1] and 
                    block_min[2] <= pos[2] <= block_max[2]):
                    is_inside = True
                
                # [SDF 2] MLA 반구 내부 판정
                elif profile_cfg['enabled'] and pos[1] > block_max[1]:
                    R = profile_cfg['radius']
                    for center in mla_centers:
                        # 구의 방정식 기반 SDF: ||p - c|| <= R
                        dist = np.linalg.norm(pos - center)
                        if dist <= R:
                            is_inside = True
                            break
                
                if is_inside:
                    p = Particle()
                    p.x = wp.vec3(*pos)
                    p.v = wp.vec3(0.0, 0.0, 0.0)
                    p.F = wp.mat33(1.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0,
                                   0.0,0.0,1.0)# 항등 행렬 초기화 필수
                    p.C = wp.mat33(0.0)
                    p.mass = p_mass
                    p.volume = p_volume
                    p.mat_id = 0 # Elastomer
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