import numpy as np
import warp as wp
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
def init_indenter_kernel(ind_min: wp.vec3, spacing: float, nx: int, ny: int, nz: int, center: wp.vec3,
                         radius: float, tot_h: float, density: float, p_vol: float,
                         particles: wp.array(dtype=Particle), offset: int):
    tid = wp.tid()
    ix, iy, iz = tid // (ny * nz), (tid // nz) % ny, tid % nz
    pos = ind_min + wp.vec3(float(ix) * spacing, float(iy) * spacing, float(iz) * spacing)
    if wp.length(pos - center) <= radius and pos[1] < tot_h * 2.0:
        p = Particle()
        p.x = pos
        p.v = wp.vec3(0.0) 
        p.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        p.C = wp.mat33(0.0)
        p.mass = density * p_vol* 10.0 # 인덴터는 질량이 더 높아서 움직임이 덜하도록 (velocity 직접 제어)
        p.volume = p_vol
        p.mat_id = 0
        p.active = 1
        particles[offset + tid] = p

def create_block_gpu(Nx, Nz, tot_h, spacing, thickness, density, mask_path, ind_cfg, device):
    hmap_np = np.load(mask_path) + thickness
    Ny = int(tot_h / spacing) + 2
    total_el_pts = Nx * Ny * Nz
    
    ind_radius = float(ind_cfg['radius'])
    ind_sp = spacing * 0.8
    i_n = int((2.0 * ind_radius) / ind_sp) + 1
    total_ind_pts = i_n**3

    with wp.ScopedDevice(device):
        hmap_wp = wp.array(hmap_np, dtype=wp.float32)
        particles = wp.zeros(total_el_pts + total_ind_pts, dtype=Particle)
        
        wp.launch(init_elastomer_kernel, dim=total_el_pts, inputs=[spacing, Nx, Ny, Nz, hmap_wp, density, spacing**3, particles])
        
        ind_center = wp.vec3(*ind_cfg['center'])
        ind_min = wp.vec3(*(np.array(ind_cfg['center']) - ind_radius))
        wp.launch(init_indenter_kernel, dim=total_ind_pts, inputs=[ind_min, ind_sp, i_n, i_n, i_n, ind_center, ind_radius, tot_h, density, spacing**3, particles, total_el_pts])
        
    return particles