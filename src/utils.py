import pyvista as pv
import os
import numpy as np
import imageio
from PIL import Image

class Visualizer:
    def __init__(self, config):
        self.output_dir = config.get('output_dir', 'output')
        self.hm_3d_dir = os.path.join(self.output_dir, 'heightmaps_3d')
        os.makedirs(self.hm_3d_dir, exist_ok=True)
            
        self.window_size = [1024, 768]
        # 메인 시각화 (입자)
        self.plotter = pv.Plotter(off_screen=not config.get('show_window', True), window_size=self.window_size)
        self.plotter.set_background("white")
        
        # 연속체 하이트맵 전용 (Delaunay)
        self.hm_plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
        self.hm_plotter.set_background("white")
        
        self.e_actor, self.i_actor = None, None
        self.frames_list = []
        self.frame_idx = 0
        self.b_min, self.b_max = config['block_min'], config['block_max']

        if config.get('show_window', True):
            self.plotter.show(interactive_update=True)

    def update(self, p_pos, num_e):
        # [방어 로직] 입자가 사라졌는지(NaN) 체크
        if np.isnan(p_pos).any() or len(p_pos) == 0:
            print(f"Warning: Frame {self.frame_idx} contains invalid particle data. Skipping render.")
            return

        # 1. 3D 입자 시각화
        if self.e_actor: self.plotter.remove_actor(self.e_actor)
        self.e_actor = self.plotter.add_mesh(pv.PolyData(p_pos[:num_e]), color="deepskyblue", point_size=3)
        if self.i_actor: self.plotter.remove_actor(self.i_actor)
        self.i_actor = self.plotter.add_mesh(pv.PolyData(p_pos[num_e:]), color="red", point_size=5)
        
        self.plotter.update()
        img = self.plotter.screenshot()
        if img is not None:
            self.frames_list.append(np.array(Image.fromarray(img).resize(self.window_size)))

        # 2. 연속체 메쉬 생성 (Delaunay 2D)
        self.save_smooth_heightmap(p_pos[:num_e])
        self.frame_idx += 1

    def save_smooth_heightmap(self, elastomer_pos):
        """흩어진 점들을 연결하여 매끄러운 3D 곡면으로 변환"""
        if len(elastomer_pos) < 10: return # 입자가 너무 적으면 메쉬 생성 불가

        try:
            cloud = pv.PolyData(elastomer_pos)
            # Delaunay 2D로 점들을 연결하여 삼각형 면(Face)을 만듭니다.
            surface = cloud.delaunay_2d(alpha=0.05) 
            
            self.hm_plotter.clear()
            # 고무 느낌이 나도록 부드러운 쉐이딩 적용
            self.hm_plotter.add_mesh(surface, cmap="viridis", lighting=True, smooth_shading=True)
            self.hm_plotter.camera_position = [(0.5, 0.8, 1.2), (0.5, 0.2, 0.5), (0, 1, 0)]
            
            path = os.path.join(self.hm_3d_dir, f"smooth_hm_{self.frame_idx:04d}.png")
            self.hm_plotter.screenshot(path)
        except Exception as e:
            print(f"Mesh generation error: {e}")

    def finalize(self):
        if self.frames_list:
            imageio.mimsave(os.path.join(self.output_dir, "simulation.gif"), self.frames_list, fps=30)
        self.plotter.close()
        self.hm_plotter.close()