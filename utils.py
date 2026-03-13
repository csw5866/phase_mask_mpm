import numpy as np
import matplotlib.pyplot as plt
import os

def propagate_asm(u_in, z, wl, px, step, out_dir="output", title="ASM Propagation"):
    Ny, Nx = u_in.shape
    
    # 1. Zero Padding (주변부 회절 노이즈 방지)
    pad_y, pad_x = Ny // 2, Nx // 2
    u_padded = np.pad(u_in, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    
    Ny_p, Nx_p = u_padded.shape
    fx = np.fft.fftfreq(Nx_p, d=px)
    fy = np.fft.fftfreq(Ny_p, d=px)
    FX, FY = np.meshgrid(fx, fy)
    
    # 2. Transfer Function 계산
    arg = 1 - (wl * FX)**2 - (wl * FY)**2
    mask = arg >= 0
    H = np.zeros_like(arg, dtype=complex)
    H[mask] = np.exp(1j * (2*np.pi/wl) * z * np.sqrt(arg[mask]))
    
    # 3. FFT 전파
    U_f = np.fft.fft2(u_padded)
    U_out_padded = np.fft.ifft2(U_f * H)
    
    # 4. Cropping 및 강도 계산
    U_out = U_out_padded[pad_y:pad_y+Ny, pad_x:pad_x+Nx]
    intensity = np.abs(U_out)**2

    # --- [버그 수정: 경로 생성 로직] ---
    save_path = os.path.join(out_dir, "asm_results")
    if not os.path.exists(save_path):
        os.makedirs(save_path) # save_dir이 아니라 save_path를 생성해야 함
        
    save_file = os.path.join(save_path, f"asm_{step:04d}.png")

    plt.figure(figsize=(8, 6))
    # Intensity 분포가 좁을 수 있으므로 vmin, vmax 설정을 고려하세요.
    im = plt.imshow(intensity, extent=[0, Nx*px*1e6, 0, Ny*px*1e6], cmap='magma', origin='lower')
    plt.colorbar(im, label='Intensity (a.u.)')
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.title(f"{title} (Step {step}, z={z*1e6:.1f}μm)")
    
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()

    return intensity


def save_visuals(el_map, in_map, step, pixel_size, out_dir="output"):
    dir_2d = os.path.join(out_dir, "2d_hmap")
    dir_3d_el = os.path.join(out_dir, "3d_elastomer")
    dir_3d_comb = os.path.join(out_dir, "3d_combined")
    
    for d in [dir_2d, dir_3d_el, dir_3d_comb]:
        os.makedirs(d, exist_ok=True)
    
    nx, nz = el_map.shape
    x = np.linspace(0, nx * pixel_size, nx)
    z = np.linspace(0, nz * pixel_size, nz)
    X, Z = np.meshgrid(x, z)
    
    # 높이 스케일 자동 조정
    valid_el = el_map[el_map > 1e-7] 
    if len(valid_el) > 0:
        z_min, z_max = np.min(valid_el), np.max(valid_el)
        margin = (z_max - z_min) * 0.1 if z_max != z_min else 1e-6
        auto_zlim = [z_min - margin, z_max + margin]
    else:
        auto_zlim = [0, 1e-5]


    # 1. 2D Heightmap (vmin, vmax 적용으로 대비 최적화)
    plt.figure(figsize=(8, 6))
    plt.imshow(el_map.T, origin='lower', extent=[0, x[-1], 0, z[-1]], 
               cmap='viridis', vmin=auto_zlim[0], vmax=auto_zlim[1])
    plt.colorbar(label='Height (m)')
    plt.title(f"2D Heightmap - Step {step}")
    plt.savefig(os.path.join(dir_2d, f"2d_hmap_{step:04d}.png"))
    plt.close()

    # 3D 시각화 공통 설정
    elev, azim = 35, 45

    # 2. 3D Elastomer Only (linewidth=0 추가로 뾰족함 방지)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Z, el_map.T, cmap='viridis', 
                           antialiased=True, alpha=0.9,
                           rcount=512, ccount=512, # 해상도 적절히 조절
                           linewidth=0, shade=True) # 격자선 제거
    ax.set_zlim(auto_zlim)
    ax.set_title(f"Elastomer Surface - Step {step}")
    plt.savefig(os.path.join(dir_3d_el, f"3d_el_{step:04d}.png"))
    plt.close()

    # 3. 3D Combined (Elastomer + Indenter)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 엘라스토머 (반투명)
    ax.plot_surface(X, Z, el_map.T, cmap='viridis', 
                    antialiased=True, alpha=0.4,
                    rcount=512, ccount=512, linewidth=0)
    
    in_data = in_map.T
    if not np.all(np.isnan(in_data)):
        # 인덴터 (빨간색) - 0(또는 이전 단계에서 처리된 값) 제외하고 플롯
        in_mask = np.isfinite(in_data)
        if np.any(in_mask):
            ax.plot_surface(X, Z, in_data, color='red', alpha=0.7, 
                            rcount=512, ccount=512, linewidth=0, shade=True)
            
            # 인덴터까지 포함하도록 Z축 범위 재확장
            in_min = np.nanmin(in_data)
            ax.set_zlim([min(auto_zlim[0], in_min), auto_zlim[1]])
        else:
            ax.set_zlim(auto_zlim)
    else:
        ax.set_zlim(auto_zlim)
            
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Combined View - Step {step}")
    plt.savefig(os.path.join(dir_3d_comb, f"3d_comb_{step:04d}.png"))
    plt.close()