import warp as wp
from .data_types import ParticleState

@wp.kernel
def p2g_kernel(
    p: wp.array(dtype=ParticleState),
    grid_m: wp.array(dtype=wp.float32, ndim=3),
    grid_v: wp.array(dtype=wp.vec3, ndim=3),
    dx: wp.float32,
    mu: wp.float32,
    lam: wp.float32,
    num_e: wp.int32
):
    tid = wp.tid()
    inv_dx = 1.0 / dx
    pos = p[tid].x
    
    stress_term = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    if tid < num_e:
        F = p[tid].F
        J = wp.determinant(F)
        
        # [블랙홀 방지 핵심] J가 0.6 이하로 떨어지지 않게 하여 발산 차단
        # PDMS의 비압축성을 수치적으로 안정시키기 위한 장치입니다.
        J_safe = wp.clamp(J, 0.6, 1.5) 
        
        inv_F_T = wp.transpose(wp.inverse(F))
        
        # Neo-Hookean Stress 안정화 버전
        # ln(J) 항이 과도한 인력을 만들지 않도록 조절됨
        P = mu * (F - inv_F_T) + lam * wp.log(J_safe) * inv_F_T
        stress = (1.0 / J_safe) * (P * wp.transpose(F))
        
        # 그리드에 전달할 힘의 가중치 계산
        volume = p[tid].vol * J_safe
        stress_term = -1.0 * volume * stress * 4.0 * inv_dx * inv_dx

    # P2G Transfer 로직 (이하 동일하지만 안정성 강화)
    rx, ry, rz = grid_m.shape[0], grid_m.shape[1], grid_m.shape[2]
    base_i = wp.clamp(wp.int32(wp.floor(pos[0] * inv_dx - 0.5)), 0, rx - 3)
    base_j = wp.clamp(wp.int32(wp.floor(pos[1] * inv_dx - 0.5)), 0, ry - 3)
    base_k = wp.clamp(wp.int32(wp.floor(pos[2] * inv_dx - 0.5)), 0, rz - 3)
    
    fx = pos * inv_dx - wp.vec3(wp.float32(base_i), wp.float32(base_j), wp.float32(base_k))
    v0, v1, v2 = wp.vec3(1.5)-fx, fx-wp.vec3(1.0), fx-wp.vec3(0.5)
    
    w_x = wp.vec3(0.5*v0[0]*v0[0], 0.75-v1[0]*v1[0], 0.5*v2[0]*v2[0])
    w_y = wp.vec3(0.5*v0[1]*v0[1], 0.75-v1[1]*v1[1], 0.5*v2[1]*v2[1])
    w_z = wp.vec3(0.5*v0[2]*v0[2], 0.75-v1[2]*v1[2], 0.5*v2[2]*v2[2])

    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w_x[i] * w_y[j] * w_z[k]
                dpos = (wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)) - fx) * dx
                mass_w = weight * p[tid].m
                mom = mass_w * (p[tid].v + (p[tid].C * dpos))
                
                wp.atomic_add(grid_m, base_i + i, base_j + j, base_k + k, mass_w)
                wp.atomic_add(grid_v, base_i + i, base_j + j, base_k + k, mom + (stress_term * weight) * dpos)

@wp.kernel
def grid_update_kernel(grid_m: wp.array(dtype=wp.float32, ndim=3), grid_v: wp.array(dtype=wp.vec3, ndim=3), dt: wp.float32, gravity: wp.vec3):
    i, j, k = wp.tid()
    m = grid_m[i, j, k]
    if m > 1e-6:
        v = grid_v[i, j, k] / m + gravity * dt
        rx, ry, rz = grid_m.shape[0], grid_m.shape[1], grid_m.shape[2]
        # 5면 고정 경계 조건 (단단히 고정)
        if i < 3 or i > rx - 4 or j < 3 or k < 3 or k > rz - 4:
            v = wp.vec3(0.0)
        grid_v[i, j, k] = v

@wp.kernel
def g2p_kernel(p: wp.array(dtype=ParticleState), grid_v: wp.array(dtype=wp.vec3, ndim=3), dt: wp.float32, dx: wp.float32, num_e: wp.int32, v_rigid: wp.vec3):
    tid = wp.tid()
    if tid >= num_e: # 인덴터: 완벽한 강체 평행이동
        p[tid].x = p[tid].x + v_rigid * dt
        p[tid].v = v_rigid
        p[tid].C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return

    inv_dx = 1.0 / dx
    pos = p[tid].x
    rx, ry, rz = grid_v.shape[0], grid_v.shape[1], grid_v.shape[2]
    base_i = wp.clamp(wp.int32(wp.floor(pos[0] * inv_dx - 0.5)), 0, rx - 3)
    base_j = wp.clamp(wp.int32(wp.floor(pos[1] * inv_dx - 0.5)), 0, ry - 3)
    base_k = wp.clamp(wp.int32(wp.floor(pos[2] * inv_dx - 0.5)), 0, rz - 3)
    fx = pos * inv_dx - wp.vec3(wp.float32(base_i), wp.float32(base_j), wp.float32(base_k))
    v0, v1, v2 = wp.vec3(1.5)-fx, fx-wp.vec3(1.0), fx-wp.vec3(0.5)
    w_x = wp.vec3(0.5*v0[0]*v0[0], 0.75-v1[0]*v1[0], 0.5*v2[0]*v2[0])
    w_y = wp.vec3(0.5*v0[1]*v0[1], 0.75-v1[1]*v1[1], 0.5*v2[1]*v2[1])
    w_z = wp.vec3(0.5*v0[2]*v0[2], 0.75-v1[2]*v1[2], 0.5*v2[2]*v2[2])

    nv = wp.vec3(0.0); nC = wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w_x[i] * w_y[j] * w_z[k]
                gv = grid_v[base_i + i, base_j + j, base_k + k]
                dpos = (wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)) - fx) * dx
                nv += weight * gv
                term = wp.mat33(gv[0]*dpos[0], gv[0]*dpos[1], gv[0]*dpos[2], gv[1]*dpos[0], gv[1]*dpos[1], gv[1]*dpos[2], gv[2]*dpos[0], gv[2]*dpos[1], gv[2]*dpos[2])
                nC += term * (weight * 4.0 * inv_dx * inv_dx)

    p[tid].v = nv
    p[tid].C = nC
    # CFL 가드: 한 프레임에 그리드 한 칸 이상 이동 절대 금지
    limit = wp.vec3(dx * 0.5)
    p[tid].x = p[tid].x + wp.min(wp.max(nv * dt, -limit), limit)
    p[tid].F = (wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) + nC * dt) * p[tid].F