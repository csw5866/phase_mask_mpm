import numpy as np
import warp as wp

def setup_scene(config):
    # 1. 엘라스토머 입자 생성 (Material ID: 0)
    b_min, b_max, sp = np.array(config['block_min']), np.array(config['block_max']), config['spacing']
    x, y, z = np.arange(b_min[0], b_max[0], sp), np.arange(b_min[1], b_max[1], sp), np.arange(b_min[2], b_max[2], sp)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    elastomer_pos = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1).astype(np.float32)
    
    # 2. 인덴터 입자 생성 (Material ID: 1 - Rigid)
    ind_cfg = config['indenter']
    ind_center = np.array(ind_cfg['center'])
    ind_radius = float(ind_cfg['radius'])
    
    # 인덴터 표면과 내부를 채울 입자 생성
    ind_sp = sp * 0.8 # 인덴터는 조금 더 밀도 있게 생성
    ix = np.arange(ind_center[0]-ind_radius, ind_center[0]+ind_radius, ind_sp)
    iy = np.arange(ind_center[1]-ind_radius, ind_center[1]+ind_radius, ind_sp)
    iz = np.arange(ind_center[2]-ind_radius, ind_center[2]+ind_radius, ind_sp)
    IX, IY, IZ = np.meshgrid(ix, iy, iz, indexing='ij')
    potential_ind_pos = np.stack([IX.flatten(), IY.flatten(), IZ.flatten()], axis=-1).astype(np.float32)
    
    # 구 형태 안에 있는 입자만 필터링
    dists = np.linalg.norm(potential_ind_pos - ind_center, axis=1)
    indenter_pos = potential_ind_pos[dists <= ind_radius]
    
    return elastomer_pos, indenter_pos

def get_material_params(config):
    E, nu = float(config['youngs_modulus']), float(config['poisson_ratio'])
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam