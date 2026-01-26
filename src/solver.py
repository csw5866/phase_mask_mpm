import warp as wp
from .kernels import p2g_kernel, grid_update_kernel, g2p_kernel
from .data_types import ParticleState

class MPMSolver:
    def __init__(self, config, e_pos, i_pos, mu, lam):
        self.dt, self.dx = float(config['dt']), 1.0 / float(config['res'][0])
        self.mu, self.lam = float(mu), float(lam)
        self.gravity = wp.vec3(0.0, 0.0, 0.0)
        
        # [수정] Attribute 이름 일치 확인
        self.num_elastomer = len(e_pos)
        self.num_particles = self.num_elastomer + len(i_pos)
        self.v_rigid = wp.vec3(*config['indenter']['velocity'])
        
        self.p = wp.array(dtype=ParticleState, shape=self.num_particles)
        self.grid_m = wp.zeros(shape=config['res'], dtype=wp.float32)
        self.grid_v = wp.zeros(shape=config['res'], dtype=wp.vec3)
        
        vol = float(config['spacing']**3)
        mass = vol * float(config['density'])
        p_structs = []
        
        for pos in e_pos:
            p_obj = ParticleState()
            p_obj.x, p_obj.v = wp.vec3(*pos), wp.vec3(0.0)
            p_obj.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            p_obj.C, p_obj.m, p_obj.vol, p_obj.mat_id = wp.mat33(0.0), mass, vol, 0
            p_structs.append(p_obj)
            
        for pos in i_pos:
            p_obj = ParticleState()
            p_obj.x, p_obj.v = wp.vec3(*pos), self.v_rigid
            p_obj.F, p_obj.C = wp.mat33(0.0), wp.mat33(0.0)
            p_obj.m, p_obj.vol, p_obj.mat_id = mass * 200.0, vol, 1
            p_structs.append(p_obj)
            
        self.p.assign(p_structs)

    def step(self):
        self.grid_m.zero_()
        self.grid_v.zero_()
        # [수정] 커널 인자 개수 동기화
        wp.launch(p2g_kernel, dim=self.num_particles, inputs=[self.p, self.grid_m, self.grid_v, self.dx, self.mu, self.lam, self.num_elastomer])
        wp.launch(grid_update_kernel, dim=self.grid_m.shape, inputs=[self.grid_m, self.grid_v, self.dt, self.gravity])
        wp.launch(g2p_kernel, dim=self.num_particles, inputs=[self.p, self.grid_v, self.dt, self.dx, self.num_elastomer, self.v_rigid])