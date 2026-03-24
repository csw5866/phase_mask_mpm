import warp as wp

@wp.struct
class Particle:
    x: wp.vec3
    v: wp.vec3
    F: wp.mat33
    C: wp.mat33
    mass: float
    volume: float
    mat_id: int
    active: int