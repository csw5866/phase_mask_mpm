import warp as wp

@wp.struct
class Particle:
    x: wp.vec3        # position
    v: wp.vec3        # velocity
    F: wp.mat33       # deformation gradient
    C: wp.mat33       # APIC affine matrix
    mass: float
    volume: float
    mat_id: wp.int32 #0 : elastomer, 1: rigid indenter