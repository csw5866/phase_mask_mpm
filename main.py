import warp as wp
import yaml
import os
import numpy as np
# [수정] scene_maker로 변경 확인 필요. 만약 파일명이 scene.py라면 그대로 두세요.
try:
    from src.scene_maker import setup_scene
except ImportError:
     from src.scene import setup_scene

from src.scene import get_material_params
from src.solver import MPMSolver
from src.utils import Visualizer

def main():
    input_path = "input/indentation.yaml"
    with open(input_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    wp.init()
    wp.set_device(config.get('device', 'cpu'))

    # 해상도 증가로 인해 입자 수가 많아지므로 초기화 시간 소요
    print("장면 초기화 중...")
    e_pos, i_pos = setup_scene(config)
    mu, lam = get_material_params(config)
    
    solver = MPMSolver(config, e_pos, i_pos, mu, lam)
    viz = Visualizer(config)
    
    print(f"시뮬레이션 시작 (Grid: {config['res']}, Elastomer: {len(e_pos)}, Indenter: {len(i_pos)})...")

    try:
        for f in range(config['frame_count']):
            solver.step()
            
            if f % 2 == 0:
                current_pos = solver.p.numpy()['x']
                if np.isnan(current_pos).any():
                    print(f"\n[오류] NaN 감지됨. 중단.")
                    break
                viz.update(current_pos, solver.num_elastomer)
                
            if f % 10 == 0: # 진행 상황 더 자주 출력
                print(f"Frame {f}/{config['frame_count']}...")
                
    except Exception as e:
        print(f"\n[실패] 오류 발생: {e}")
    finally:
        if config.get('save_gif', True):
            viz.finalize()
        print("종료.")

if __name__ == "__main__":
    main()