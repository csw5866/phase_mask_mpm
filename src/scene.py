import numpy as np
import warp as wp
import trimesh
from .data_types import Particle

@wp.kernel
def init_elastomer_kernel(
    spacing: float, 
    Nx: int, Ny: int, Nz: int, 
    profile: wp.array(dtype=float, ndim=2),
    density: float, 
    p_vol_std: float, 
    particles: wp.array(dtype=Particle)
):
    tid = wp.tid()
    
    # 3D 인덱스 계산
    ix = tid // (Ny * Nz)
    iy = (tid // Nz) % Ny
    iz = tid % Nz

    if ix >= Nx or iz >= Nz:
        return

    target_h = profile[ix, iz]
    pos_y = float(iy) * spacing
    
    # 입자생성 및 최상단 입자 부피 보정
    if pos_y <= target_h:
        is_top_particle = (pos_y + spacing) > target_h
        
        p = Particle()
        
        if is_top_particle:
            p.x = wp.vec3(float(ix) * spacing, target_h, float(iz) * spacing)
            
            # 2. 부피 보정: 이 입자가 담당하는 실제 높이를 계산
            lower_bound = (float(iy) - 0.5) * spacing
            if lower_bound < 0.0: lower_bound = 0.0 # 바닥 보정
            
            actual_h = target_h - lower_bound
            p.volume = spacing * spacing * actual_h
            p.mass = density * p.volume
        else:
            p.x = wp.vec3(float(ix) * spacing, pos_y, float(iz) * spacing)
            p.volume = p_vol_std
            p.mass = density * p_vol_std
            
        p.v = wp.vec3(0.0)
        p.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        p.C = wp.mat33(0.0)
        p.mat_id = 1
        p.active = 1
        particles[tid] = p


@wp.kernel
def init_indenter_from_stl_kernel(
    mesh: wp.uint64,
    grid_origin: wp.vec3,
    spacing: float,
    dim_x: int, dim_y: int, dim_z: int,
    min_b: wp.vec3, max_b: wp.vec3, # 사용자 지정 XYZ 제한
    threshold: float,
    density: float,
    p_vol: float,
    particles: wp.array(dtype=Particle), # data_types.py의 Particle 구조체
    offset: int # 엘라스토머 이후 시작 위치
):
    tid = wp.tid()
    
    # 격자 인덱스 및 좌표 계산
    iz = tid // (dim_x * dim_y)
    iy = (tid // dim_x) % dim_y
    ix = tid % dim_x
    pos = grid_origin + wp.vec3(float(ix) * spacing, float(iy) * spacing, float(iz) * spacing)

    # 1. XYZ 경계 제한 확인
    if (pos.x >= min_b.x and pos.x <= max_b.x and
        pos.y >= min_b.y and pos.y <= max_b.y and
        pos.z >= min_b.z and pos.z <= max_b.z):
        
        # 2. STL 표면 쿼리 변수 준비 (Warp 1.12.0 시그니처)
        max_dist = wp.float32(threshold)
        inside = wp.float32(0.0)
        face_id = wp.int32(-1)
        u = wp.float32(0.0)
        v = wp.float32(0.0)
        
        # 3. 표면 판단 및 Particle 데이터 기입
        if wp.mesh_query_point(mesh, pos, max_dist, inside, face_id, u, v):
            p = Particle()
            p.x = pos
            p.v = wp.vec3(0.0)
            p.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            p.C = wp.mat33(0.0)
            p.mass = density * p_vol * 10.0 # 강체 특성
            p.volume = p_vol
            p.mat_id = 0 # 인덴터 ID
            p.active = 1
            particles[offset + tid] = p
            return

    # 조건을 만족하지 못하면 해당 슬롯은 active=0 (비활성) 상태로 남음
    # (wp.zeros로 초기화되었으므로 별도 기입 불필요)



def create_block_gpu(Nx, Nz, tot_h, spacing, thickness, density, mask_path, ind_path, min_b, max_b, device):
    
    # [CPU] 메타데이터 및 STL 로드
    hmap_np = np.load(mask_path).astype(np.float32) + thickness
    Ny = int(tot_h / spacing) + 2
    total_el_pts = Nx * Ny * Nz
    
    # STL 로드 및 기초 변환
    mesh = trimesh.load(ind_path)
    mesh.apply_scale(0.001)
    
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    with wp.ScopedDevice(device):
        # 1. GPU 메모리 업로드 (Mesh)
        wp_mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3),
            indices=wp.array(faces.flatten(), dtype=int)
        )

        # 2. 인덴터 샘플링 영역 계산
        b = mesh.bounds
        ind_sp = spacing * 0.8 # 인덴터는 약간 더 촘촘하게
        origin = wp.vec3(b[0][0] - ind_sp, b[0][1] - ind_sp, b[0][2] - ind_sp)
        end = wp.vec3(b[1][0] + ind_sp, b[1][1] + ind_sp, b[1][2] + ind_sp)
        
        dims = (
            int((end[0]-origin[0])/ind_sp) + 1,
            int((end[1]-origin[1])/ind_sp) + 1,
            int((end[2]-origin[2])/ind_sp) + 1
        )
        total_ind_grid_pts = dims[0] * dims[1] * dims[2]

        # 3. 전체 입자 배열 통합 할당 (Elastomer + Indenter Bounding Box)
        # numpy를 타지 않기 위해 인덴터는 Bounding Box 내의 모든 격자점을 공간으로 확보합니다.
        particles = wp.zeros(total_el_pts + total_ind_grid_pts, dtype=Particle)
        hmap_wp = wp.array(hmap_np, dtype=wp.float32)

        # 4. 엘라스토머 배치 (기존 커널)
        wp.launch(
            kernel=init_elastomer_kernel, 
            dim=total_el_pts, 
            inputs=[spacing, Nx, Ny, Nz, hmap_wp, density, spacing**3, particles]
        )
        
        wp.launch(
            kernel=init_indenter_from_stl_kernel,
            dim=total_ind_grid_pts,
            inputs=[
                wp_mesh.id, origin, ind_sp, dims[0], dims[1], dims[2],
                min_b, max_b, ind_sp * 0.6, density, spacing**3, 
                particles, total_el_pts
            ]
        )

        print(f"   - Elastomer Slots: {total_el_pts}")
        print(f"   - Indenter Slots: {total_ind_grid_pts}")
        
        return particles