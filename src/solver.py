import warp as wp
from .kernels import *
from .scene import create_block
from .utils import *

class MPMSolver:
    def __init__(self, cfg):
        #self.spacing = float(cfg["spacing"])
        self.dt = float(cfg["dt"])
        self.step_count = 0

        self.nx = int(cfg["grid_res"][0])
        self.ny = int(cfg["grid_res"][1])
        self.nz = int(cfg["grid_res"][2])
        self.grid_size = self.nx * self.ny * self.nz
        self.block_min = np.array(cfg["block_min"])
        self.block_max = np.array(cfg["block_max"])
        self.dx = 1.0 / (self.nx - 1)     #self.nx = slef.ny = self.nz이고 domain은 정육면체 형태 ([0, 1.0]^3)
        self.spacing = self.dx / 3          #indentation.yaml 대신 grid_res 기반 충분한 sampling이 가능하도록 조절
        self.inv_dx = 1.0 / self.dx
        self.indenter = cfg["indenter"]
        self.profile = cfg["mla"]
        self.density = float(cfg["density"])

        E = float(cfg["E"])
        nu = float(cfg["nu"])
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))

        parts = create_block(
            self.block_min,
            self.block_max,
            self.spacing,
            self.density,
            self.profile,
            self.indenter,
        )

        self.particles = wp.array(parts, dtype=Particle)
        self.np = len(parts)
        self.grid_v = wp.zeros(self.grid_size, dtype=wp.vec3)
        self.grid_m = wp.zeros(self.grid_size, dtype=float)

        self.gravity = wp.vec3(*cfg["gravity"])

    def step(self):
        if self.step_count == 0:
            hmap = compute_height_map(self.particles,self.nx,self.nz)
            save_height_map(hmap, self.step_count)
            save_height_map_3d(hmap, self.step_count)


        wp.launch(clear_grid, self.grid_size, [self.grid_v, self.grid_m])

        wp.launch(p2g, self.np, [
            self.particles, self.grid_v, self.grid_m,
            self.dx, self.inv_dx, self.dt,
            self.mu, self.lam,
            self.nx, self.ny, self.nz
        ])

        wp.launch(grid_update, self.grid_size, [
            self.grid_v, self.grid_m,
            self.gravity, self.dt,self.dx,
            wp.vec3(self.block_min[0],self.block_min[1],self.block_min[2]),
            wp.vec3(self.block_max[0],self.block_max[1],self.block_max[2]),
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
            hmap = compute_height_map(self.particles, self.nx, self.nz)
            save_height_map(hmap, self.step_count)
            save_height_map_3d(hmap, self.step_count)
            save_height_map_3d_with_indenter(self.particles,hmap, self.step_count)
            save_height_map_3d_surface_indenter(self.particles,hmap, self.step_count)
