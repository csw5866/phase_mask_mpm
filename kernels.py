import warp as wp
import numpy as np
from .data_types import Particle

@wp.kernel
def clear_grid(grid_v: wp.array(dtype=wp.vec3), grid_m: wp.array(dtype=float)):
    i = wp.tid()
    grid_v[i] = wp.vec3(0.0)
    grid_m[i] = 0.0

@wp.kernel
def p2g(particles: wp.array(dtype=Particle), grid_v: wp.array(dtype=wp.vec3), grid_m: wp.array(dtype=float),
        dx: float, inv_dx: float, dt: float, mu: wp.array(dtype=float), lam: wp.array(dtype=float),
        nx: int, ny: int, nz: int):
    pid = wp.tid()
    p = particles[pid]
    if p.active == 0: return

    mid = p.mat_id
    xp = p.x * inv_dx
    base = wp.vec3(wp.floor(xp[0] - 0.5), wp.floor(xp[1] - 0.5), wp.floor(xp[2] - 0.5))
    fx = xp - base

    w = wp.mat33(0.0)
    for d in range(3):
        x = fx[d]
        w[d, 0] = 0.5 * (1.5 - x) * (1.5 - x)
        w[d, 1] = 0.75 - (x - 1.0) * (x - 1.0)
        w[d, 2] = 0.5 * (x - 0.5) * (x - 0.5)

    if mid > 0:
        mu_p = mu[0]
        lam_p = lam[0]
        F = p.F
        U = wp.mat33()
        S = wp.vec3()
        V = wp.mat33()
        wp.svd3(F, U, S, V)
        J = wp.max(S[0] * S[1] * S[2], 0.01)
        stress = 2.0 * mu_p * (F - U @ wp.transpose(V)) @ wp.transpose(F) + \
                 lam_p * J * (J - 1.0) * wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        stress = (-dt * p.volume * 4.0 * inv_dx * inv_dx) * stress
        affine = stress + p.mass * p.C
    else:
        affine = p.mass * p.C

    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[0, i] * w[1, j] * w[2, k]
                node_idx = (int(base[0]) + i) * ny * nz + (int(base[1]) + j) * nz + (int(base[2]) + k)
                dpos = (wp.vec3(base[0] + float(i), base[1] + float(j), base[2] + float(k)) - xp) * dx
                wp.atomic_add(grid_v, node_idx, weight * (p.mass * p.v + affine @ dpos))
                wp.atomic_add(grid_m, node_idx, weight * p.mass)

@wp.kernel
def grid_update(grid_v: wp.array(dtype=wp.vec3), grid_m: wp.array(dtype=float), gravity: wp.vec3,
                dt: float, dx: float, nx: int, ny: int, nz: int):
    i = wp.tid()
    m = grid_m[i]
    if m > 1e-14:
        v = grid_v[i] / m + dt * gravity
        iz = i % nz
        iy = (i // nz) % ny
        ix = i // (nz * ny)
        if ix <= 1 or ix >= nx - 2 or iz <= 1 or iz >= nz - 2 or iy <= 1:
            v = wp.vec3(0.0)
        grid_v[i] = v
    else:
        grid_v[i] = wp.vec3(0.0)

@wp.kernel
def g2p(particles: wp.array(dtype=Particle), grid_v: wp.array(dtype=wp.vec3), grid_m: wp.array(dtype=float),
        ind_v: wp.vec3, dx: float, inv_dx: float, dt: float, nx: int, ny: int, nz: int):
    pid = wp.tid()
    p = particles[pid]
    if p.active == 0: return

    if p.mat_id == 0:
        p.v = ind_v
        p.x += dt * ind_v
        particles[pid] = p
        return

    xp = p.x * inv_dx
    base = wp.vec3(wp.floor(xp[0] - 0.5), wp.floor(xp[1] - 0.5), wp.floor(xp[2] - 0.5))
    fx = xp - base
    w = wp.mat33(0.0)
    for d in range(3):
        x = fx[d]
        w[d, 0] = 0.5 * (1.5 - x) * (1.5 - x)
        w[d, 1] = 0.75 - (x - 1.0) * (x - 1.0)
        w[d, 2] = 0.5 * (x - 0.5) * (x - 0.5)

    new_v = wp.vec3(0.0)
    new_C = wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[0, i] * w[1, j] * w[2, k]
                idx = (int(base[0]) + i) * ny * nz + (int(base[1]) + j) * nz + (int(base[2]) + k)
                if grid_m[idx] > 0.0:
                    gv = grid_v[idx]
                    dpos = wp.vec3(float(i), float(j), float(k)) - fx
                    new_v += weight * gv
                    new_C += 4.0 * inv_dx * weight * wp.outer(gv, dpos)

    p.v = new_v
    p.x += dt * new_v
    p.C = new_C
    p.F = (wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) + dt * new_C) @ p.F
    particles[pid] = p



@wp.kernel
def compute_dual_heightmaps_kernel(
    particles: wp.array(dtype=Particle),
    nx: int, nz: int,
    dx: float,
    inv_dx: float,
    hmap_el: wp.array(dtype=float, ndim=2),
    hmap_in: wp.array(dtype=float, ndim=2)
):
    pid = wp.tid()
    p = particles[pid]
    if p.active == 0: return

    # pixel_size 기반 인덱싱
    ix = int(p.x[0] * inv_dx)
    iz = int(p.x[2] * inv_dx)

    # 단순 점이 아니라 주변 1x1 범위라도 채워주기 (Aliasing 방지)
    for ox in range(0, 2):
        for oz in range(0, 2):
            curr_x = ix + ox
            curr_z = iz + oz
            if curr_x < nx and curr_z < nz:
                if p.mat_id == 1:
                    wp.atomic_max(hmap_el, curr_x, curr_z, p.x[1])
                elif p.mat_id == 0:
                    wp.atomic_min(hmap_in, curr_x, curr_z, p.x[1])
