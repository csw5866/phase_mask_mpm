import warp as wp
import numpy as np
from .data_types import Particle

@wp.kernel
def clear_grid(grid_v: wp.array(dtype=wp.vec3),
               grid_m: wp.array(dtype=float)):
    i = wp.tid()
    grid_v[i] = wp.vec3(0.0, 0.0, 0.0)
    grid_m[i] = 0.0


@wp.kernel
def p2g(
    particles: wp.array(dtype=Particle),
    grid_v: wp.array(dtype=wp.vec3),
    grid_m: wp.array(dtype=float),
    dx: float,
    inv_dx: float,
    dt: float,
    mu: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
):
    pid = wp.tid()
    p = particles[pid]

    # particle position in grid coordinates
    xp = p.x * inv_dx

    base = wp.vec3(
        wp.floor(xp[0] - 0.5),
        wp.floor(xp[1] - 0.5),
        wp.floor(xp[2] - 0.5),
    )

    fx = xp - base

    # quadratic B-spline weights
    w = wp.mat33(0.0)
    for d in range(3):          #tacchi 2.0 style과 다르게 [a,b] indexing에서 a가 dimensino을, b가 이웃한 grid의 index를 의미
        x = fx[d]
        w[d, 0] = 0.5 * (1.5 - x) * (1.5 - x)
        w[d, 1] = 0.75 - (x - 1.0) * (x - 1.0)
        w[d, 2] = 0.5 * (x - 0.5) * (x - 0.5)

    ###############################################################################(확인 필요)
    # Neo-Hookean stress (corotated model 이용)
    F = p.F
    U = wp.mat33()
    S = wp.vec3()  # Sigma: 3개의 부동소수점 값을 담는 벡터
    V = wp.mat33()
    wp.svd3(F,U,S,V)
    J = S[0]*S[1]*S[2]
    '''
    if p.mat_id == 0:   # for indenter
        mu0 = mu[0]
        lam0 = lam[0]
    else:
        mu0 = mu[p.mat_id - 1]
        lam0 = lam[p.mat_id - 1]
    
    stress = 2.0 * mu * (F - U @ wp.transpose(V)) @ wp.transpose(F) + lam * J * (J-1.0) * wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
     #warp 문서상으로도 kirchoff_stress_FCR로 정의
    stress = (-dt * p.volume * 4.0 * inv_dx * inv_dx) * stress

    affine = stress + p.mass * p.C
    '''

    if p.mat_id > 0:
        # Neo-Hookean Stress 계산 (Kirchhoff Stress FCR)
        # transpose(F)가 아니라 p.F의 행렬곱 순서 확인 필요 (보통 F_trial @ F_old^T)
        stress = 2.0 * mu * (F - U @ wp.transpose(V)) @ wp.transpose(F) + \
                lam * J * (J - 1.0) * wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        
        stress = (-dt * p.volume * 4.0 * inv_dx * inv_dx) * stress
        affine = stress + p.mass * p.C
    else:
        # 인덴터는 강체이므로 탄성 스트레스가 없음
        affine = p.mass * p.C



    ###############################################################################
    # scatter to grid
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[0, i] * w[1, j] * w[2, k]

                node = wp.vec3(
                    base[0] + float(i),
                    base[1] + float(j),
                    base[2] + float(k),
                )

                ix = int(node[0])
                iy = int(node[1])
                iz = int(node[2])

                if ix < 0 or iy < 0 or iz < 0:
                    continue
                if ix >= nx or iy >= ny or iz >= nz:
                    continue

                idx = ix * ny * nz + iy * nz + iz       #grid의 index는 z -> y -> x순으로 차례대로 할당

                dpos = (node - xp) * dx     #normalize 안 된 실제 물리적인 값
                
                wp.atomic_add(
                    grid_v,
                    idx,
                    weight * (p.mass * p.v + affine @ dpos),
                )
                wp.atomic_add(
                    grid_m,
                    idx,
                    weight * p.mass,
                )



@wp.kernel
def grid_update(grid_v: wp.array(dtype=wp.vec3),
                grid_m: wp.array(dtype=float),
                gravity: wp.vec3,
                dt: float,
                dx: float,
                nx: int, ny: int, nz: int):

    i = wp.tid()        # i = 0 ~ (dim - 1)
    m = grid_m[i]
    iz = i % nz
    iy = (i // nz) % ny
    ix = i // (nz*ny)           #ix, iy, iz는 normalize된 grid 위치(0~64 사이값)이고, i_xpos,i_ypos,i_zpos는 물리적 거리를 고려한 위치 (0~1사이값)

    i_xpos = float(ix) * dx
    i_ypos = float(iy) * dx
    i_zpos = float(iz) * dx

    if m > 1e-14:
        v = grid_v[i] / m
        v += dt * gravity
        
        if ix <= 1 or ix >= nx - 2 or \
                iz <= 1 or iz >= nz - 2 or \
                iy <= 1: 
            v = wp.vec3(0.0, 0.0, 0.0)
            '''
            eps = 1e-6
            if i_xpos < eps or i_xpos > 1.0 - eps or \
            i_ypos < eps or i_ypos > 1.0 - eps or \
            i_zpos < eps or i_zpos > 1.0 - eps:
                v = wp.vec3(0.0)
            '''
    else: 
        v = wp.vec3(0.0)

    grid_v[i] = v


@wp.kernel
def g2p(
    particles: wp.array(dtype=Particle),
    grid_v: wp.array(dtype=wp.vec3),
    grid_m: wp.array(dtype=float),
    indenter_v: wp.vec3,
    dx: float,
    inv_dx: float,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
):
    pid = wp.tid()
    p = particles[pid]

    xp = p.x * inv_dx

    base = wp.vec3(
        wp.floor(xp[0] - 0.5),
        wp.floor(xp[1] - 0.5),
        wp.floor(xp[2] - 0.5),
    )

    fx = xp - base

    # quadratic B-spline weights
    w = wp.mat33(0.0)
    for d in range(3):
        x = fx[d]
        w[d, 0] = 0.5 * (1.5 - x) * (1.5 - x)
        w[d, 1] = 0.75 - (x - 1.0) * (x - 1.0)
        w[d, 2] = 0.5 * (x - 0.5) * (x - 0.5)

    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0)


    if p.mat_id != 0:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[0, i] * w[1, j] * w[2, k]

                    node = wp.vec3(
                        base[0] + float(i),
                        base[1] + float(j),
                        base[2] + float(k),
                    )

                    ix = int(node[0])
                    iy = int(node[1])
                    iz = int(node[2])

                    if ix < 0 or iy < 0 or iz < 0:
                        continue
                    if ix >= nx or iy >= ny or iz >= nz:
                        continue

                    idx = ix * ny * nz + iy * nz + iz

                    if grid_m[idx] > 0.0:
                        gv = grid_v[idx] 
                        dpos = wp.vec3(float(i), float(j), float(k)) - fx

                        new_v += weight * gv
                        new_C += 4.0 * inv_dx * weight * wp.outer(gv, dpos)
    elif p.mat_id == 0:
        new_v = indenter_v


    p.v = new_v
    p.x += dt * new_v
    p.C = new_C
    # deformation gradient update
    if p.mat_id == 0:
        p.F = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    else:
        p.F = (wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) + dt * new_C) @ p.F


    particles[pid] = p
