import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 데이터 불러오기
file_path = "mla_heightmap_512.npy"  # 파일 경로를 입력하세요
data = np.load(file_path)

# 2. 좌표 축 설정 (512x512 해상도 기준)
ny, nx = data.shape
x = np.linspace(0, nx, nx)
y = np.linspace(0, ny, ny)
X, Y = np.meshgrid(x, y)

# 3. 시각화 시작
fig = plt.figure(figsize=(14, 6))

# --- 2D Plot ---
ax1 = fig.add_subplot(121)
im = ax1.imshow(data, cmap='viridis', origin='lower')
ax1.set_title("2D Heightmap")
plt.colorbar(im, ax=ax1, label='Height')

# --- 3D Plot ---
ax2 = fig.add_subplot(122, projection='3d')
# rcount, ccount는 렌더링 속도를 위해 샘플링 밀도를 조절합니다 (512 전체는 무거울 수 있음)
surf = ax2.plot_surface(X, Y, data, cmap='viridis', 
                        rcount=256, ccount=256, antialiased=True)
ax2.set_title("3D Surface View")
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()