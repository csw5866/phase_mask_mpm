import warp as wp

@wp.struct
class ParticleState:
    x: wp.vec3    
    v: wp.vec3    
    F: wp.mat33   
    C: wp.mat33   
    m: wp.float32 
    vol: wp.float32 
    mat_id: wp.int32 # 0: Elastomer, 1: Rigid Indenter