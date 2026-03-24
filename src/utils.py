import numpy as np
import matplotlib.pyplot as plt
import os

def propagate_asm(u_in, z, wl, px, step, out_dir="output", title="ASM Propagation"):
    Ny, Nx = u_in.shape
    
    # 1. Zero Padding (circular convolution 방지)
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

    
    save_path = os.path.join(out_dir, "asm_results")
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
        
    save_file = os.path.join(save_path, f"asm_{step:04d}.png")

    plt.figure(figsize=(8, 6))
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
    x = np.linspace(0, (nx -1) * pixel_size, nx)
    z = np.linspace(0, (nz - 1) * pixel_size, nz)
    X, Z = np.meshgrid(x, z)
    
    
    # 높이 스케일 자동 조정 (--> 미사용, *.gif 파일 제작 시 동일한 zlim 유지 위해 수동 설정)
    valid_el = el_map[el_map > 1e-7] 
    if len(valid_el) > 0:
        z_min, z_max = np.min(valid_el), np.max(valid_el)
        margin = (z_max - z_min) * 0.1 if z_max != z_min else 1e-6
        auto_zlim = [z_min - margin, z_max + margin]
    else:
        auto_zlim = [0, 1e-5]


    # 1. 2D Heightmap 
    plt.figure(figsize=(8, 6))
    plt.imshow(el_map.T, origin='lower',
               cmap='viridis',
               vmin = 3.0e-5, vmax = 5.6e-5)
    plt.colorbar(label='Height (m)')
    plt.title(f"2D Heightmap - Step {step}")
    plt.savefig(os.path.join(dir_2d, f"2d_hmap_{step:04d}.png"))
    plt.close()

    # 3D view 공통 설정
    elev, azim = 55, 45

    # 2. 3D Elastomer Only 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Z, el_map.T, cmap='viridis', 
                           antialiased=True, alpha=0.9,
                           rcount=512, ccount=512, # 최대 해상도로 설정하기
                           linewidth=0, shade=True) # 격자선 제거
    ax.set_zlim([3.5e-5, 10.0e-5]) # Z축 범위 수동 설정
    ax.set_title(f"Elastomer Surface - Step {step}")
    plt.savefig(os.path.join(dir_3d_el, f"3d_el_{step:04d}.png"))
    plt.close()

    # 3. 3D Combined (Elastomer + Indenter)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 엘라스토머
    ax.plot_surface(X, Z, el_map.T, cmap='viridis', 
                    antialiased=True, alpha=0.6,
                    rcount=512, ccount=512, linewidth=0)
    
    in_data = in_map.T
    if not np.all(np.isnan(in_data)):
        # 인덴터
        in_mask = np.isfinite(in_data)
        if np.any(in_mask):
            ax.plot_surface(X, Z, in_data, color='red', alpha=0.9, 
                            rcount=512, ccount=512, linewidth=0, shade=True)

            ax.set_zlim([3.5e-5, 10.0e-5])
        else:
            ax.set_zlim([3.5e-5, 10.0e-5])
    else:
        ax.set_zlim([3.5e-5, 10.0e-5])
            
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Combined View - Step {step}")
    plt.savefig(os.path.join(dir_3d_comb, f"3d_comb_{step:04d}.png"))
    plt.close()