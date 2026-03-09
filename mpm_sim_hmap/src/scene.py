import numpy as np
import warp as wp
from .data_types import Particle
@wp.kernel
def sample_layered_block_kernel(
    spacing: float,
    upscale: int,
    Nx: int, Ny: int, Nz: int,
    profile: wp.array(dtype=float, ndim=2),
    out_mask: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3),
    out_mat_id: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    idx_x = tid // (Ny * Nz)
    idx_y = (tid // Nz) % Ny
    idx_z = tid % Nz
    
    pos = wp.vec3(float(idx_x) * spacing, float(idx_y) * spacing, float(idx_z) * spacing)
    
    # [Step 1] 전체 샘플링 범위 내에 있는지 확인
    if (idx_x <= Nx and idx_y <= Ny and idx_z <= Nz):
        # Bilinear sampling을 위해 원본 hmap 인덱스(0~767)로 변환
        # float(idx_x) / float(upscale)을 통해 768 해상도 좌표로 매핑
        u = float(idx_x) / float(upscale)
        v = float(idx_z) / float(upscale)
        
        boundary_y = wp.sample_bilinear(profile, u, v)
        
        # [Step 2] 계산된 높이보다 아래에 있는 경우에만 입자 생성
        if pos[1] <= boundary_y:
            out_mask[tid] = 1
            out_pos[tid] = pos
            out_mat_id[tid] = 1 # Elastomer ID
        else:
            out_mask[tid] = 0
    else:
        out_mask[tid] = 0


# --- [GPU 커널] 인덴터(구형) 입자 샘플링 ---
@wp.kernel
def sample_indenter_kernel(
    ind_min: wp.vec3,
    spacing: float,
    nx: int, ny: int, nz: int,
    center: wp.vec3,
    radius: float,
    out_mask: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    idx_x = tid // (ny * nz)
    idx_y = (tid // nz) % ny
    idx_z = tid % nz
    
    pos = ind_min + wp.vec3(float(idx_x) * spacing, float(idx_y) * spacing, float(idx_z) * spacing)
    dist = wp.length(pos - center)
    
    if dist <= radius:
        out_mask[tid] = 1
        out_pos[tid] = pos
    else:
        out_mask[tid] = 0

# --- [Main Function] ---
def create_block(nx, nz, tot_h, upscale, spacing, thickness, density, mask_path, ind_cfg, device):
    """
    NVIDIA Warp GPU 가속을 활용한 다층 구조 입자 생성 함수
    density: np.array([dens_mat1, dens_mat2])
    device: "cuda:0" 또는 "cpu"
    """
    hmap_np = np.load(mask_path)
    hmap_np = hmap_np + thickness
    Ny = int(tot_h * 1.1 / spacing)
    #checking
    print(Ny)

    Nx = upscale * (nx - 1) + 1
    Nz = upscale * (nz - 1) + 1

    with wp.ScopedDevice(device):
        hmap_wp = wp.array(hmap_np, dtype=wp.float32)
        # 2. 블록 입자 생성 (GPU 커널 실행)
        total_particle_pts = Nx * Ny * Nz
        
        b_mask = wp.zeros(total_particle_pts, dtype=wp.int32)
        b_pos = wp.zeros(total_particle_pts, dtype=wp.vec3)
        b_mat = wp.zeros(total_particle_pts, dtype=wp.int32)

        wp.launch(kernel=sample_layered_block_kernel, dim=total_particle_pts, inputs=[
            spacing, upscale,
            Nx,Ny,Nz,
            hmap_wp,
            b_mask, b_pos, b_mat
        ])

        # 3. 인덴터 입자 생성 (GPU 커널 실행)
        ind_center = np.array(ind_cfg['center'])
        ind_radius = float(ind_cfg['radius'])
        ind_sp = spacing * 0.8 # 인덴터는 약간 더 조밀하게
        ind_min = ind_center - ind_radius
        i_nx = int((2.0 * ind_radius) / ind_sp) + 1
        i_ny = int((2.0 * ind_radius) / ind_sp) + 1
        i_nz = int((2.0 * ind_radius) / ind_sp) + 1
        total_ind_pts = i_nx * i_ny * i_nz
        
        i_mask = wp.zeros(total_ind_pts, dtype=wp.int32)
        i_pos = wp.zeros(total_ind_pts, dtype=wp.vec3)

        wp.launch(kernel=sample_indenter_kernel, dim=total_ind_pts, inputs=[
            wp.vec3(*ind_min), ind_sp, i_nx, i_ny, i_nz,
            wp.vec3(*ind_center), ind_radius,
            i_mask, i_pos
        ])

        # 4. CPU 데이터 회수 및 Particle 객체화
        # 초기화 단계에서 한 번만 수행되므로 numpy 전환은 효율적입니다.
        particles = []
        p_volume = spacing ** 3
        
        # 블록 데이터 처리
        m_host, p_host, id_host = b_mask.numpy(), b_pos.numpy(), b_mat.numpy()
        for idx in range(total_particle_pts):
            if m_host[idx] == 1:
                mid = id_host[idx]
                p = Particle()
                p.x = wp.vec3(*p_host[idx])
                p.v = wp.vec3(0.0)
                p.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                p.C = wp.mat33(0.0)
                p.mass = density * p_volume
                p.volume = p_volume
                p.mat_id = mid
                particles.append(p)

        # 인덴터 데이터 처리 (mat_id=0)
        im_host, ip_host = i_mask.numpy(), i_pos.numpy()
        ind_vel = wp.vec3(*ind_cfg['velocity'])
        for idx in range(total_ind_pts):
            if im_host[idx] == 1:
                p = Particle()
                p.x = wp.vec3(*ip_host[idx])
                p.v = ind_vel
                p.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                p.C = wp.mat33(0.0)
                # 인덴터는 보통 첫 번째 재질 밀도의 10배를 적용하여 강체성 부여
                p.mass = density * p_volume * 10.0
                p.volume = p_volume
                p.mat_id = 0 # Rigid Indenter
                particles.append(p)

    return particles
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
'''
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
    '''

