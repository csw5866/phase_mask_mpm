import os
from PIL import Image

def create_gif(input_folder, output_path, duration=100):
    # --- [디버깅 추가] 현재 작업 디렉토리 확인 ---
    print(f"현재 작업 경로: {os.getcwd()}")
    print(f"찾으려는 폴더: {os.path.abspath(input_folder)}")

    # 1. 폴더 존재 여부 확인
    if not os.path.exists(input_folder):
        print(f"❌ 에러: 폴더를 찾을 수 없습니다 -> {input_folder}")
        return

    # 2. 이미지 파일 목록 가져오기 (대소문자 구분 없이 .png, .PNG 모두 가져오도록 수정)
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"찾은 이미지 개수: {len(images)}개")
    
    images.sort()

    if not images:
        print("⚠️ 폴더에 이미지 파일이 없습니다. 확장자(.png)를 확인하세요.")
        return

    frames = []
    print("이미지 읽는 중...")
    for image_name in images:
        img_path = os.path.join(input_folder, image_name)
        new_frame = Image.open(img_path)
        frames.append(new_frame)

    print(f"GIF 생성 중... ({output_path})")
    frames[0].save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )
    
    print(f"✅ 성공! GIF 저장 완료: {os.path.abspath(output_path)}")

# --- 사용 예시 ---
# 괄호()가 포함된 경로는 오타가 나기 쉬우니 주의하세요!
input_dir = "output/output(4e-4radii)/3d_combined"
output_name = "3d_combined_(4e-4).gif"

create_gif(input_dir, output_name, duration=100)