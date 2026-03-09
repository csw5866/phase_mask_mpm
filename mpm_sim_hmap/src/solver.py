import warp as wp
from .kernels import *
from .scene import create_block
from .utils import *

class MPMSolver:
    def __init__(self, cfg):

        self.dt = float(cfg["dt"])
        self.step_count = 0
        self.device = cfg["device"]
        self.gravity = np.array(cfg["gravity"])
        self.pixel_size = float(cfg["pixel_size"])
        self.upscale = int(cfg["spacing"])
        self.spacing = self.pixel_size / self.upscale
        # grid는 dx = pixel_size 간격으로 배치(grid_res = mask_hmap res), particle = self.pixel_size/self.spacing 간격으로 배치
        # 따라서 particle의 res인 particle_res = upscale * (res - 1) + 1 
        #nx, ny, nz는 grid resolution이고 Nx, Ny,
        self.dx = self.pixel_size
        self.inv_dx = 1.0 / self.dx
        self.nx = int(cfg["res"][0])
        self.nz = int(cfg["res"][1])


        #elastomer and grid
        # 단일두께의 직육면체와 윗면은 .npy 기반 profile만큼 높이 추가
        profile_h = float(cfg["profile_hmax"])
        self.thickness = float(cfg["mask_thickness"])
        self.h_tot = profile_h + self.thickness
        self.ny = int(self.h_tot * 2.0 * self.inv_dx) + 2
        self.grid_size = self.nx * self.ny * self.nz
        self.mask_path = cfg["mask_path"]
        self.density = np.array(cfg["density"])

        E = np.float(cfg["E"])
        nu = np.float(cfg["nu"]) 
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))

        #indenter
        self.indenter = cfg["indenter"]
        
        parts = create_block(
            self.nx, self.nz, self.h_tot,
            self.upscale, 
            self.spacing,
            self.thickness,
            self.density,
            self.mask_path,
            self.indenter,
            self.device,
        )


        with wp.ScopedDevice(self.device):
            self.particles = wp.array(parts, dtype=Particle)
            self.np = len(parts)
            self.grid_v = wp.zeros(self.grid_size, dtype=wp.vec3)
            self.grid_m = wp.zeros(self.grid_size, dtype=float)
            self.mu = wp.array(mu, dtype=wp.float32, device=self.device)
            self.lam = wp.array(lam, dtype=wp.float32, device=self.device)
            self.gravity = wp.vec3(*cfg["gravity"])

    def step(self):
        if self.step_count == 0:
            hmap = compute_height_map(self.particles,self.nx,self.nz)
            save_height_map(hmap, self.step_count)
            save_height_map_3d(hmap, self.step_count)

        

        with wp.ScopedDevice(self.device):

            wp.launch(clear_grid, self.grid_size, [self.grid_v, self.grid_m])

            wp.launch(p2g, self.np, [
                self.particles, self.grid_v, self.grid_m,
                self.dx, self.inv_dx, self.dt,
                self.mu, self.lam,                      #need to fix
                self.nx, self.ny, self.nz
            ])

            wp.launch(grid_update, self.grid_size, [
                self.grid_v, self.grid_m,
                self.gravity, self.dt,self.dx,
                self.nx, self.ny, self.nz
            ])

            wp.launch(g2p, self.np, [
                self.particles, self.grid_v, self.grid_m, wp.vec3(*np.array(self.indenter["velocity"])),
                self.dx, self.inv_dx, self.dt,
                self.nx, self.ny, self.nz
            ])

        self.step_count += 1

        if self.step_count % 10 == 0 or self.step_count < 10 :
            x = self.particles.numpy()["x"]
            if not np.all(np.isfinite(x)):
                print("NaN detected at step", self.step_count)
                return
            print(f"[MPM] Step {self.step_count}")
            save_particle_frame(self.particles, self.step_count)
            hmap_1 = compute_height_map(self.particles, 1, self.nx, self.nz)
            save_height_map(hmap_1, self.step_count,"height_1")
            save_height_map_3d(hmap_1, self.step_count,"height_1_3d")
            hmap_2 = compute_height_map(self.particles,2,self.nx, self.nz)
            save_height_map(hmap_2, self.step_count,"height_2")
            save_height_map_3d(hmap_2, self.step_count,"height_2_3d")
            save_height_map_3d_with_indenter(self.particles,hmap_1,hmap_2, self.step_count)
            #save_height_map_3d_surface_indenter(self.particles,hmap, self.step_count)
