import warp as wp
import numpy as np
import cv2
from .kernels import *
from .scene import create_block_gpu
from .data_types import Particle
from .utils import *

class MPMSolver:
    def __init__(self, cfg):
        self.device = cfg["device"]
        self.dt = float(cfg["dt"])
        self.pixel_size = float(cfg["pixel_size"])
        self.sampling = int(cfg["sampling"])
        self.dx = self.pixel_size * self.sampling
        self.inv_dx = 1.0 / self.dx
        self.Nx, self.Nz = int(cfg["res"][0]), int(cfg["res"][1])
        self.nx, self.nz = self.Nx // self.sampling + 2, self.Nz // self.sampling + 2
        self.thickness = float(cfg["mask_thickness"])
        
        self.profile_hmax = float(cfg["profile_hmax"])
        h_tot = self.profile_hmax + self.thickness
        self.ny = int(h_tot * 2.0 * self.inv_dx) + 2
        self.grid_size = self.nx * self.ny * self.nz
        
        E, nu = float(cfg["E"]), float(cfg["nu"])
        mu_val = E / (2.0 * (1.0 + nu))
        lam_val = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        self.ind_v = wp.vec3(*cfg["indenter"]["velocity"])
        self.gravity = wp.vec3(*cfg["gravity"])
        self.step_count = 0

        self.particles = create_block_gpu(self.Nx, self.Nz, h_tot, self.pixel_size, self.thickness, float(cfg["density"]), cfg["mask_path"], cfg["indenter"], self.device)
        self.np = len(self.particles)

        with wp.ScopedDevice(self.device):
            self.grid_v = wp.zeros(self.grid_size, dtype=wp.vec3)
            self.grid_m = wp.zeros(self.grid_size, dtype=float)
            self.mu = wp.array([mu_val], dtype=wp.float32)
            self.lam = wp.array([lam_val], dtype=wp.float32)

    def step(self):
        with wp.ScopedDevice(self.device):
            wp.launch(clear_grid, dim=self.grid_size, inputs=[self.grid_v, self.grid_m])
            wp.launch(p2g, dim=self.np, inputs=[self.particles, self.grid_v, self.grid_m, self.dx, self.inv_dx, self.dt, self.mu, self.lam, self.nx, self.ny, self.nz])
            wp.launch(grid_update, dim=self.grid_size, inputs=[self.grid_v, self.grid_m, self.gravity, self.dt, self.dx, self.nx, self.ny, self.nz])
            wp.launch(g2p, dim=self.np, inputs=[self.particles, self.grid_v, self.grid_m, self.ind_v, self.dx, self.inv_dx, self.dt, self.nx, self.ny, self.nz])
        self.step_count += 1
        #########################################################################
        n_glass = 1.5
        wavelength = 633e-9     # 633nm (빨간색 레이저)
        k = 2.0 * np.pi / wavelength
        lens_pitch = 100e-6     # 약 100um
        target_sag = 5e-6       # 5um (높이)

        # 곡률 반경 및 초점 거리 계산
        lens_radius = lens_pitch / 2
        Radius_of_curvature = (lens_radius**2) / (2 * target_sag)
        focal_length = Radius_of_curvature / (n_glass - 1)
        #####################################################################

        if self.step_count % 20 == 0 or self.step_count == 1: # 20스텝마다 시각화
                    # 1. GPU 버퍼 초기화 (엘라스토머는 0, 인덴터는 아주 높은 값으로 초기화)
                    hmap_el_gpu = wp.zeros(shape=(self.nx, self.nz), dtype=float, device=self.device)
                    hmap_in_gpu = wp.full(shape=(self.nx, self.nz), dtype=float, value=1e3, device=self.device)
                    
                    # 2. 커널 실행 (nx, nz 해상도에 맞게 추출)
                    wp.launch(compute_dual_heightmaps_kernel, dim=self.np, inputs=[
                        self.particles, self.nx, self.nz, self.dx, self.inv_dx, 
                        hmap_el_gpu, hmap_in_gpu
                    ])
                    
                    # 3. CPU로 복사 (이미지 한 장 크기이므로 매우 빠름)
                    el_map = hmap_el_gpu.numpy()
                    in_map = hmap_in_gpu.numpy()
                    # 인덴터가 없는 곳(1e3)은 NaN 처리
                    in_map[in_map > 10.0] = np.nan
                    print(el_map.min(), el_map.max(), in_map[np.isfinite(in_map)].min(), in_map[np.isfinite(in_map)].max())

                    refined_el = cv2.resize(el_map, (self.Nx, self.Nz), interpolation=cv2.INTER_CUBIC)
                    refined_ind = cv2.resize(in_map, (self.Nx, self.Nz), interpolation=cv2.INTER_CUBIC)
                    print(el_map.min(), el_map.max(), in_map[np.isfinite(in_map)].min(), in_map[np.isfinite(in_map)].max())

                    refined_el_smooth = cv2.GaussianBlur(refined_el, (3, 3), 0.5)
                    el_map_profile = refined_el_smooth - self.thickness
                    if el_map_profile.max() > self.profile_hmax and el_map_profile.min() < 0:
                        print(f"Warning: Unexpected height values detected (max: {el_map_profile.max():.2e}, min: {el_map_profile.min():.2e})")
                    U_in = np.exp(1j * k * (n_glass - 1) * el_map_profile) # 높이 차이에 따른 위상 마스크 계산
                    Intensity = propagate_asm(U_in, focal_length, wavelength, self.pixel_size, self.step_count)
                    
                    save_visuals(refined_el, refined_ind, self.step_count, self.pixel_size)

                    print(f"Step {self.step_count} completed, visuals saved.")
