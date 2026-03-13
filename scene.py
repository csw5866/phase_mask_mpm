import numpy as np
import warp as wp
from .data_types import Particle

@wp.kernel
def init_elastomer_kernel(spacing: float, Nx: int, Ny: int, Nz: int, profile: wp.array(dtype=float, ndim=2),
                          density: float, p_vol: float, particles: wp.array(dtype=Particle)):
    tid = wp.tid()
    ix, iy, iz = tid // (Ny * Nz), (tid // Nz) % Ny, tid % Nz
    pos = wp.vec3(float(ix) * spacing, float(iy) * spacing, float(iz) * spacing)
    if ix < Nx and iz < Nz and pos[1] <= profile[ix, iz]:
        p = Particle()
        p.x = pos
        p.v = wp.vec3(0.0)
        p.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        p.C = wp.mat33(0.0)
        p.mass = density * p_vol
        p.volume = p_vol
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
        p.v = wp.vec3(0.0) # Solver에서 나중에 업데이트하더라도 초기화는 개별로
        p.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        p.C = wp.mat33(0.0)
        p.mass = density * p_vol* 10.0 # 인덴터는 질량이 더 높아서 움직임이 덜하도록 (Solver에서 velocity 직접 제어할 예정)
        p.volume = p_vol
        p.mat_id = 0
        p.active = 1
        particles[offset + tid] = p

def create_block_gpu(Nx, Nz, tot_h, spacing, thickness, density, mask_path, ind_cfg, device):
    hmap_np = np.load(mask_path) + thickness
    Ny = int(tot_h * 1.1 / spacing)
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