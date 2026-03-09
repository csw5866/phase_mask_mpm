import numpy as np
import matplotlib.pyplot as plt
# 3D 플롯을 위해 필요
from mpl_toolkits.mplot3d import Axes3D
import os
# [핵심 추가] 입자들을 감싸는 표면 계산을 위한 라이브러리
from scipy.spatial import ConvexHull
import imageio

def save_particle_frame(particles, step, out_dir="frames"):
    """
    Matplotlib 시각화 수정: 깊이 문제 해결을 위해 단일 scatter 호출 사용
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # GPU 데이터를 CPU로 복사
    data = particles.numpy()
    x = data["x"]
    
    # mat_id 필드 확인 및 추출
    if "mat_id" in data.dtype.names:
        mat_id = data["mat_id"]
    else:
        # mat_id가 없으면 모두 엘라스토머(0)로 가정
        mat_id = np.zeros(len(x), dtype=np.int32)

    # 시각화 설정
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- [핵심 수정 부분] ---
    # 입자를 따로 나누지 않고, mat_id를 기반으로 색상 배열을 미리 만듭니다.
    # 0번 인덱스: 'dodgerblue' (엘라스토머), 1번 인덱스: 'red' (인덴터)
    colormap = np.array(['dodgerblue', 'red'])
    
    # mat_id 값이 0 또는 1을 벗어나는 경우를 대비해 안전하게 클램핑 (선택사항)
    safe_mat_id = np.clip(mat_id, 0, 1)
    colors = colormap[safe_mat_id]

    # 만약 인덴터와 엘라스토머의 점 크기를 다르게 하고 싶다면 사이즈 배열도 만들어야 합니다.
    # (현재 요청하신 코드에서는 둘 다 s=0.5이므로 단일 값 사용)
    # sizes = np.where(mat_id == 1, 1.0, 0.5) # 예시: 인덴터를 2배 크게

    # 모든 입자를 한 번에 그립니다.
    # Matplotlib이 내부적으로 깊이 정렬을 시도합니다.
    # (MPM 좌표계 Y가 높이이므로, Matplotlib Z축에 Y값을 넣습니다.)
    # utils.py 내의 시각화 부분 수정
    ax.scatter(x[:, 0], x[:, 2], x[:, 1], 
            c=colors,               # 미리 계산된 색상 배열
            s=1.2,                  # [팁1] 점 크기를 0.5에서 1.2~1.5 정도로 살짝 키웁니다.
            marker='o',             # [팁2] 원형 마커임을 명시
            alpha=0.9,              # 약간의 투명도는 입자들이 겹칠 때 입체감을 줍니다.
            edgecolors='black',     # [팁3] 아주 얇은 검은색 테두리가 입자 하나하나를 구분해줍니다.
            linewidths=0.05,        # 테두리 두께를 매우 얇게 설정 (입자 느낌 극대화)
            depthshade=True)        # [팁4] 멀리 있는 입자를 어둡게 처리하여 공간감을 만듭니다.

    # 축 범위 고정
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # 라벨 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y (Height)')
    ax.set_title(f"Step {step}")

    # 뷰 각도 조정
    ax.view_init(elev=20, azim=45)

    # 저장
    img_path = os.path.join(out_dir, f"frame_{step:04d}.png")
    # dpi를 조금 낮추면 저장 속도가 빨라집니다. (예: 100 or 120)
    plt.savefig(img_path, dpi=150) 
    plt.close(fig)


def compute_height_map(particles, nx=64, nz=64, domain=(0, 1)):
    """
    [수정] mat_id == 0 (엘라스토머) 입자만을 대상으로 Y축 하이트맵을 추출합니다.
    """
    data = particles.numpy()
    x = data["x"]
    
    # mat_id 필드 확인
    if "mat_id" in data.dtype.names:
        mat_id = data["mat_id"]
    else:
        mat_id = np.zeros(len(x), dtype=np.int32)

    # 엘라스토머 입자만 필터링
    elastomer_x = x[mat_id == 0]
    
    hmap = np.zeros((nx, nz))
    if len(elastomer_x) == 0:
        return hmap

    dx = (domain[1] - domain[0]) / (nx - 1)
    
    # 실수 좌표 계산 (예를 들어 elastomer_x[:,0]이  가까운 grid 중 내림하여 해당하는 grid의 숫자와 같은 번호를 부여 받도록 함.(.astype(int)와 함께 이용하여) , 0~nx-1사이 값 )
    fx = (elastomer_x[:, 0] - domain[0]) / dx
    fz = (elastomer_x[:, 2] - domain[0]) / dx
    '''
    fx = elastomer_x[:,0]
    fz = elastomer_x[:,2]
    '''
    # NaN 및 범위 밖 입자 제거 (캐스팅 에러 방지)
    mask = np.isfinite(fx) & np.isfinite(fz)
    #관심 범위 설정
    mask &= (fx >= 0) & (fx < nx) & (fz >= 0) & (fz < nz)
    
    valid_ix = fx[mask].astype(int)
    valid_iz = fz[mask].astype(int)
    valid_y = elastomer_x[mask, 1] # Y축이 높이

    # 각 XZ 그리드 칸에서 최대 Y값(높이) 추출
    if len(valid_y) > 0:
        np.maximum.at(hmap, (valid_ix, valid_iz), valid_y)

    return hmap


def save_height_map(hmap, step, out_dir="height"):
    """하이트맵을 이미지로 저장합니다."""
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"height_{step:04d}.png")
    # 하이트맵 수치 가시성을 위해 색상 지도 최적화
    plt.imsave(img_path, hmap, origin="lower", cmap="viridis")


def save_height_map_3d(hmap, step, domain=(0, 1), out_dir="height_3d"):
    """
    지정된 영역(0.3~0.7) 이외의 데이터를 NaN으로 처리하여 
    블록 이외의 바닥면을 완전히 제거하고 시각화합니다.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    nx, nz = hmap.shape
    x_grid = np.linspace(domain[0], domain[1], nx)
    z_grid = np.linspace(domain[0], domain[1], nz)
    X, Z = np.meshgrid(x_grid, z_grid)
    
    # [수정] hmap.T를 복사한 후, 특정 범위를 벗어나는 영역을 NaN으로 마스킹합니다.
    #의 block_min, block_max 범위를 기준으로 설정
    display_hmap = hmap.T.copy()
    mask = (X < 0.3) | (X > 0.7) | (Z < 0.3) | (Z > 0.7)
    display_hmap[mask] = np.nan 

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # [핵심 수정] vmin과 vmax를 명시하여 색상 범위를 고정합니다.
    # 엘라스토머의 기본 높이가 0.25이므로 0~0.5 범위가 적당합니다.
    VMIN = 0.1
    VMAX = 0.3
    
    surf = ax.plot_surface(X, Z, display_hmap, cmap='viridis', 
                           vmin=VMIN, vmax=VMAX,  # 이 부분이 색상을 고정합니다.
                           rcount=nx,
                           ccount=nz,
                           linewidth=0, antialiased=True, alpha=0.9)
    
    # 뷰포트 설정
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.3, 0.7)
    ax.set_zlim(0.0, 0.5) # 높이(Y) 축 범위 최적화
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Height (Y)')
    ax.set_title(f"Elastomer Surface Mesh - Step {step}")
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.view_init(elev=30, azim=45)
    
    img_path = os.path.join(out_dir, f"hmap_3d_{step:04d}.png")
    plt.savefig(img_path, dpi=150)
    plt.close(fig)

def save_height_map_3d_with_indenter(particles, hmap, step, domain=(0, 1), out_dir="height_3d_w_indenter"):
    """
    엘라스토머 하이트맵과 강체 인덴터를 동일한 Figure에 3D로 시각화합니다.
   
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 입자 데이터에서 인덴터 중심 추출
    data = particles.numpy()    
    x = data["x"]
    mat_id = data["mat_id"] if "mat_id" in data.dtype.names else np.zeros(len(x))
    
    indenter_pts = x[mat_id == 1]
    has_indenter = len(indenter_pts) > 0
    
    if has_indenter:
        # 인덴터 입자들의 평균 위치를 현재 중심(Center)으로 간주
        current_center = np.mean(indenter_pts, axis=0)
        radius = 0.08  # 설정값
    
    # 2. 엘라스토머 메쉬 데이터 준비
    nx, nz = hmap.shape
    x_grid = np.linspace(domain[0], domain[1], nx)
    z_grid = np.linspace(domain[0], domain[1], nz)
    X, Z = np.meshgrid(x_grid, z_grid)
    
    display_hmap = hmap.T.copy()
    # 블록 영역(0.3~0.7) 외에는 NaN 처리하여 투명화
    mask = (X < 0.3) | (X > 0.7) | (Z < 0.3) | (Z > 0.7)
    display_hmap[mask] = np.nan 

    # 3. 시각화 시작
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 색상 범위 고정
    VMIN, VMAX = 0.1, 0.3
    
    # 엘라스토머 표면 플롯
    surf = ax.plot_surface(X, Z, display_hmap, cmap='viridis', 
                           vmin=VMIN, vmax=VMAX,
                           rcount = nx,
                           ccount = nz,
                           linewidth=0, antialiased=True, alpha=0.7)

    # 4. [추가] 인덴터(구체) 기하학적 렌더링
    if has_indenter:
        # 구체 좌표 생성을 위한 파라미터 (u: 경도, v: 위도)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        
        # 구의 수식: x = r*cos(u)*sin(v), y = r*sin(u)*sin(v), z = r*cos(v)
        # 하이트맵 좌표계(Y축이 높이)에 맞춰 매핑
        sphere_x = radius * np.outer(np.cos(u), np.sin(v)) + current_center[0]
        sphere_z = radius * np.outer(np.sin(u), np.sin(v)) + current_center[2]
        sphere_y = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + current_center[1]
        
        # 인덴터 플롯 (빨간색 구체)
        ax.plot_surface(sphere_x, sphere_z, sphere_y, color='red', alpha=0.9, shade=True)

    # 5. 뷰포트 및 라벨 설정
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.3, 0.7)
    ax.set_zlim(0.0, 0.5) 
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Height (Y)')
    ax.set_title(f"Visuo-Tactile Simulation - Step {step}")
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Elastomer Height')
    ax.view_init(elev=30, azim=45)
    
    img_path = os.path.join(out_dir, f"hmap_indenter_{step:04d}.png")
    plt.savefig(img_path, dpi=150)
    plt.close(fig)

def save_height_map_3d_surface_indenter(particles, hmap, step, domain=(0, 1), out_dir="height_3d_w_surf_indenter"):
    """
    실제 인덴터 입자들의 위치를 기반으로 표면(Convex Hull)을 재구성하여
    매끈한 3D 메쉬 형태로 엘라스토머 하이트맵 위에 함께 시각화합니다.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # --- 1. 데이터 준비 ---
    data = particles.numpy()
    x_coords = data["x"]
    mat_id = data["mat_id"] if "mat_id" in data.dtype.names else np.zeros(len(x_coords))
    
    # 인덴터 입자 추출
    indenter_pts = x_coords[mat_id == 1]
    
    # 엘라스토머 하이트맵 그리드 준비
    nx, nz = hmap.shape
    x_grid = np.linspace(domain[0], domain[1], nx)
    z_grid = np.linspace(domain[0], domain[1], nz)
    X, Z = np.meshgrid(x_grid, z_grid)
    
    # 엘라스토머 마스킹 (블록 영역 외 제거)
    display_hmap = hmap.T.copy()
    mask = (X < 0.3) | (X > 0.7) | (Z < 0.3) | (Z > 0.7)
    display_hmap[mask] = np.nan 

    # --- 2. 시각화 시작 ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 색상 고정
    VMIN, VMAX = 0.1, 0.3
    
    # [A] 엘라스토머 표면 그리기 (투명하게)
    surf = ax.plot_surface(X, Z, display_hmap, cmap='viridis', 
                           vmin=VMIN, vmax=VMAX,
                           rcount = nx,
                           ccount = nz,
                           linewidth=0, antialiased=True, alpha=0.6)

    # [B] 핵심 수정: 인덴터 입자를 이용한 표면 재구성 및 그리기
    # 3D Convex Hull을 계산하려면 최소 4개의 점이 필요합니다.
    if len(indenter_pts) > 4:
        try:
            # 1. 실제 입자 위치를 기반으로 볼록 껍질 계산
            hull = ConvexHull(indenter_pts)
            
            # 2. MPM 좌표계(X, Y_up, Z)를 Matplotlib 좌표계(X, Z, Y_up)로 매핑
            mp_x = indenter_pts[:, 0]
            mp_y = indenter_pts[:, 2] # Matplotlib의 Y축 자리에 Z좌표를 넣음
            mp_z = indenter_pts[:, 1] # Matplotlib의 Z축(높이) 자리에 Y좌표를 넣음
            
            # 3. plot_trisurf를 이용해 삼각형 메쉬 그리기
            # triangles=hull.simplices가 입자들을 연결하는 삼각형 인덱스 정보입니다.
            ax.plot_trisurf(mp_x, mp_y, mp_z, triangles=hull.simplices,
                            color='red',      # 인덴터 색상
                            alpha=0.9,        # 불투명도 (매끈한 느낌을 위해 높게)
                            shade=True,       # 음영 효과 켜기
                            edgecolor='none') # 와이어프레임 선 제거
        except Exception as e:
            print(f"인덴터 표면 재구성 실패 (Step {step}): {e}")
            # 실패 시 간단한 스캐터로 대체하거나 건너뜀
            pass

    # --- 3. 뷰포트 및 설정 ---
    # 뷰포트를 블록 중심으로 고정
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.3, 0.7)
    # 높이축은 압입 깊이를 잘 보기 위해 약간 낮게 설정
    ax.set_zlim(0.0, 0.4) 
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Height (Y)')
    ax.set_title(f"Surface Reconstruction Indentation - Step {step}")
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Elastomer Height')
    # 뷰 각도 조절 (압입되는 모습을 잘 보기 위한 각도)
    ax.view_init(elev=25, azim=35)
    
    img_path = os.path.join(out_dir, f"hmap_surf_{step:04d}.png")
    plt.savefig(img_path, dpi=150)
    plt.close(fig)


def make_gif(out_dir="frames", gif_name="mpm.gif"):
    """저장된 프레임들을 모아 GIF 애니메이션을 만듭니다."""
    file_list = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png")])
    if not file_list:
        return
    images = [imageio.v2.imread(f) for f in file_list]
    imageio.mimsave(gif_name, images, fps=15)