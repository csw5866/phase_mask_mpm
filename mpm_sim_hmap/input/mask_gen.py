import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 시스템 파라미터 최적화 (Aliasing 방지)
# ==========================================
res = 768
wavelength = 633e-9      # 633nm (He-Ne Laser)
k = 2 * np.pi / wavelength
n_glass = 1.5


# [Sampling 조건 설정] 
# 픽셀 사이즈를 2.5um 정도로 설정하여 Aliasing 방지
pixel_size = 2.6e-6 
Lx = res * pixel_size    # 약 1.92mm
Ly = Lx

# MLA 설계 (이전 대형 스케일에서 정밀 스케일로 다운사이징)
lens_pitch = 100e-6     # 약 100um
target_sag = 5e-6       # 5um (높이)

# 곡률 반경 및 초점 거리 계산
lens_radius = lens_pitch / 2
Radius_of_curvature = (lens_radius**2) / (2 * target_sag)
focal_length = Radius_of_curvature / (n_glass - 1)

print(f"--- Optical Design Specs ---")
print(f"Pixel Size: {pixel_size*1e6:.2f} um")
print(f"Focal Length (f): {focal_length*1000:.2f} mm")
print(f"Sampling Check (min pixels per lens): {lens_pitch/pixel_size:.1f} pts")

# ==========================================
# 2. Convex Heightmap (h - r^2/2R) 생성
# ==========================================
x = np.linspace(-Lx/2, Lx/2, res)
y = np.linspace(-Ly/2, Ly/2, res)
X, Y = np.meshgrid(x, y)

# 주기적 패턴 (사용자 제안 로직: h - r^2/2R)
X_local = (X % lens_pitch) - (lens_pitch / 2)
Y_local = (Y % lens_pitch) - (lens_pitch / 2)
R_sq = X_local**2 + Y_local**2

mask_aperture = R_sq <= (lens_radius**2)
# 볼록 렌즈 하이트맵
sag_map = np.zeros_like(X)
sag_map[mask_aperture] = target_sag - (R_sq[mask_aperture] / (2 * Radius_of_curvature))
sag_map = np.maximum(sag_map, 0.0)

# MPM용 데이터 저장
np.save("mla_heightmap_768.npy", sag_map.astype(np.float32))
print(f"✅ MPM Heightmap saved.")

# ==========================================
# 3. ASM (Angular Spectrum Method) 전파
# ==========================================
def propagate_asm_circ_conv(u_in, z, wl, px):
    Ny, Nx = u_in.shape
    dfx = 1.0 / (Nx * px)
    dfy = 1.0 / (Ny * px)
    
    fx = np.fft.fftfreq(Nx, d=px)
    fy = np.fft.fftfreq(Ny, d=px)
    FX, FY = np.meshgrid(fx, fy)
    
    # 전달 함수 (Transfer Function) 계산
    arg = 1 - (wl * FX)**2 - (wl * FY)**2
    mask = arg >= 0
    H = np.zeros_like(arg, dtype=complex)
    H[mask] = np.exp(1j * (2*np.pi/wl) * z * np.sqrt(arg[mask]))
    
    # 전파 수행
    U_f = np.fft.fft2(u_in)
    U_out = np.fft.ifft2(U_f * H)
    return U_out

def propagate_asm(u_in, z, wl, px):
    Ny, Nx = u_in.shape
    
    # 1. Zero Padding (2배 확장)
    # FFT의 Circular Convolution을 Linear Convolution처럼 작동하게 만듭니다.
    pad_y, pad_x = Ny // 2, Nx // 2
    u_padded = np.pad(u_in, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    
    # 확장된 크기에서의 파라미터 재설정
    Ny_p, Nx_p = u_padded.shape
    fx = np.fft.fftfreq(Nx_p, d=px)
    fy = np.fft.fftfreq(Ny_p, d=px)
    FX, FY = np.meshgrid(fx, fy)
    
    # 2. 전달 함수 (Transfer Function) 계산
    arg = 1 - (wl * FX)**2 - (wl * FY)**2
    mask = arg >= 0
    H = np.zeros_like(arg, dtype=complex)
    
    # [심화] Band-limited ASM을 위한 필터링 (선택 사항)
    # 전파 거리 z가 멀어질 때 생기는 에일리어싱을 방지하기 위해 주파수 대역을 제한하기도 합니다.
    H[mask] = np.exp(1j * (2*np.pi/wl) * z * np.sqrt(arg[mask]))
    
    # 3. 전파 수행 (확장된 도메인에서)
    U_f = np.fft.fft2(u_padded)
    U_out_padded = np.fft.ifft2(U_f * H)
    
    # 4. Cropping (중앙의 원래 영역만 추출)
    # 패딩된 영역 덕분에 경계면에서의 Wrap-around 영향이 중앙부까지 침범하지 못합니다.
    U_out = U_out_padded[pad_y:pad_y+Ny, pad_x:pad_x+Nx]
    
    return U_out

# 입사광 위상 마스크 (phi = k * (n-1) * height)
U_in = np.exp(1j * k * (n_glass - 1) * sag_map)

# 초점 거리(f)만큼 전파
print(f"🔄 Propagating to focus (z = {focal_length*1000:.2f} mm)...")
U_focus = propagate_asm(U_in, focal_length, wavelength, pixel_size)
Intensity = np.abs(U_focus)**2

# ==========================================
# 4. 결과 시각화
# ==========================================
plt.figure(figsize=(15, 5))

# 1) 설계된 하이트맵 (MPM 입력용)
plt.subplot(1, 3, 1)
plt.imshow(sag_map * 1e6, cmap='viridis', extent=[-Lx/2*1e3, Lx/2*1e3, -Ly/2*1e3, Ly/2*1e3])
plt.title("Designed Heightmap (um)")
plt.colorbar(fraction=0.046, pad=0.04)

# 2) 초점에서의 광강도 (PSF)
plt.subplot(1, 3, 2)
plt.imshow(Intensity, cmap='hot', extent=[-Lx/2*1e3, Lx/2*1e3, -Ly/2*1e3, Ly/2*1e3])
plt.title(f"Intensity at z={focal_length*1000:.1f}mm")
plt.colorbar(fraction=0.046, pad=0.04)

# 3) 중앙부 확대 (초점 확인)
plt.subplot(1, 3, 3)
zoom = res // 6
center = res // 2
plt.imshow(Intensity[center-zoom:center+zoom, center-zoom:center+zoom], cmap='hot')
plt.title("Zoomed Focal Spots")
plt.axis('off')

plt.tight_layout()
plt.show()