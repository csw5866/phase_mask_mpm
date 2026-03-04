import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def visualize_cad_temp(stl_path):
    # 1. 파일 존재 여부 확인
    if not os.path.exists(stl_path):
        print(f"에러: 파일을 찾을 수 없습니다 -> {stl_path}")
        return

    # 2. STL 로드
    mesh = trimesh.load(stl_path)
    #mesh.apply_scale(0.001)  #--> CAD 그대로 불러오면 10mm -> 10.0 이 되므로 scaling을 통해 10mm = 0.01m -> 0.01 이 되도록 설정
    #mesh.apply_translation(dx,dy,dz)   # --> CAD와 실험 도메인 중심 안 맞을 때 + (orientation도 확인 필요)
    print(f"모델 로드 완료: {stl_path}")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"Bounding Box: {mesh.bounds}")

    # 3. Matplotlib 3D 설정
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 4. 메쉬 데이터를 Poly3DCollection으로 변환
    # mesh.vertices[mesh.faces]를 통해 각 면의 정점 좌표들을 추출합니다.
    poly3d = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.5)
    poly3d.set_edgecolor('black')   # 와이어프레임 색상
    poly3d.set_facecolor('dodgerblue') # 면 색상
    ax.add_collection3d(poly3d)

    # 5. 축 범위 설정 (Bounding Box 기준)
    # Matplotlib은 자동으로 범위를 잡지 못하는 경우가 많아 수동 설정이 안전합니다.
    scale = mesh.vertices.flatten()
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])

    # 6. 라벨링
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"CAD Preview: {os.path.basename(stl_path)}")

    # 7. 출력
    plt.show()

# --- 실행부 ---
stl_path = "input/mla_16.stl"  # 실제 파일 경로로 수정하세요
visualize_cad_temp(stl_path)