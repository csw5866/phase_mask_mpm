import yaml
import warp as wp
from src.solver import MPMSolver
from src.utils import *

wp.init()
print(wp.vec3(0.0))
print(wp.mat33(0.0))
print(wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

with open('input/indentation.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

solver = MPMSolver(cfg)

for step in range(cfg["steps"]):
    solver.step()
'''
make_gif()
make_gif("height_3d","mpm_3d.gif")
make_gif("height_3d_w_indenter","mpm_3d_w_indenter.gif")
make_gif("height_3d_w_surf_indenter","mpm_3d_w_surf_indenter.gif")
'''
