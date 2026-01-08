import gradio as gr
import cv2
import numpy as np
import tempfile
import os
import subprocess
import json
import zipfile
from pathlib import Path
import re

MAX_SIZE = 1920
VIDEO_MAX_SIZE = 720

def natural_sort_key(s):
    """자연 정렬을 위한 키 함수"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def resize_and_crop_to_match(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    ratio1 = w1 / h1
    ratio2 = w2 / h2
    
    if ratio1 > ratio2:
        new_h = h2
        new_w = int(h2 * ratio1)
    else:
        new_w = w2
        new_h = int(w2 / ratio1)
    
    img1_resized = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    start_x = (new_w - w2) // 2
    start_y = (new_h - h2) // 2
    return img1_resized[start_y:start_y+h2, start_x:start_x+w2]

def resize_if_needed(img, max_size=MAX_SIZE):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img

def get_face_info(img):
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                    min_detection_confidence=0.1, min_tracking_confidence=0.1) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                h, w = img.shape[:2]
                
                left_eye_idx = [33, 133, 160, 159, 158, 144, 145, 153]
                left_eye_x = np.mean([lm[i].x for i in left_eye_idx]) * w
                left_eye_y = np.mean([lm[i].y for i in left_eye_idx]) * h
                
                right_eye_idx = [362, 263, 387, 386, 385, 373, 374, 380]
                right_eye_x = np.mean([lm[i].x for i in right_eye_idx]) * w
                right_eye_y = np.mean([lm[i].y for i in right_eye_idx]) * h
                
                eyes_center_x = (left_eye_x + right_eye_x) / 2
                eyes_center_y = (left_eye_y + right_eye_y) / 2
                
                eye_dist = np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
                eye_angle = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
                
                nose_tip = lm[4]
                nose_x, nose_y = nose_tip.x * w, nose_tip.y * h
                
                upper_lip = lm[0]
                upper_lip_x, upper_lip_y = upper_lip.x * w, upper_lip.y * h
                
                lower_lip = lm[17]
                lower_lip_x, lower_lip_y = lower_lip.x * w, lower_lip.y * h
                
                chin = lm[152]
                chin_x, chin_y = chin.x * w, chin.y * h
                
                midline_length = np.sqrt((chin_x - nose_x)**2 + (chin_y - nose_y)**2)
                midline_angle = np.arctan2(chin_y - nose_y, chin_x - nose_x)
                
                midline_center_x = (upper_lip_x + lower_lip_x) / 2
                midline_center_y = (upper_lip_y + lower_lip_y) / 2
                
                idx = [10, 152, 234, 454]
                xs = [lm[i].x * w for i in idx]
                ys = [lm[i].y * h for i in idx]
                face_h = max(ys) - min(ys)
                face_w = max(xs) - min(xs)
                aspect = face_h / face_w if face_w > 0 else 1.0
                is_half_face = aspect < 0.9
                
                return {
                    'eye_dist': eye_dist,
                    'eye_angle': eye_angle,
                    'eyes_center_x': eyes_center_x,
                    'eyes_center_y': eyes_center_y,
                    'midline_length': midline_length,
                    'midline_angle': midline_angle,
                    'midline_center_x': midline_center_x,
                    'midline_center_y': midline_center_y,
                    'is_half_face': is_half_face,
                    'detected': True
                }
    except Exception as e:
        print(f"MediaPipe error: {e}")
    
    h, w = img.shape[:2]
    return {
        'eye_dist': w * 0.3,
        'eye_angle': 0,
        'eyes_center_x': w / 2,
        'eyes_center_y': h * 0.35,
        'midline_length': h * 0.3,
        'midline_angle': np.pi / 2,
        'midline_center_x': w / 2,
        'midline_center_y': h * 0.6,
        'is_half_face': True,
        'detected': False
    }

def match_brightness(img1, img2):
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg = (np.mean(lab1[:,:,0]) + np.mean(lab2[:,:,0])) / 2
    lab1[:,:,0] = np.clip(lab1[:,:,0] + (avg - np.mean(lab1[:,:,0])), 0, 255)
    lab2[:,:,0] = np.clip(lab2[:,:,0] + (avg - np.mean(lab2[:,:,0])), 0, 255)
    return cv2.cvtColor(lab1.astype(np.uint8), cv2.COLOR_LAB2BGR), cv2.cvtColor(lab2.astype(np.uint8), cv2.COLOR_LAB2BGR)

def add_logo(frame, logo, margin_ratio=0.03, width_ratio=0.27):
    if logo is None:
        return frame
    
    h, w = frame.shape[:2]
    logo_w = int(w * width_ratio)
    logo_h = int(logo.shape[0] * (logo_w / logo.shape[1]))
    logo_resized = cv2.resize(logo, (logo_w, logo_h), interpolation=cv2.INTER_LANCZOS4)
    margin = int(w * margin_ratio)
    
    if len(logo_resized.shape) == 3 and logo_resized.shape[2] == 4:
        alpha = logo_resized[:, :, 3] / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        logo_bgr = logo_resized[:, :, :3]
        roi = frame[margin:margin+logo_h, margin:margin+logo_w]
        blended = (logo_bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
        frame[margin:margin+logo_h, margin:margin+logo_w] = blended
    else:
        frame[margin:margin+logo_h, margin:margin+logo_w] = logo_resized
    
    return frame

def align_images(before, after, bi, ai):
    """사진용: after 기준으로 before 정렬 (기존 방식 유지)"""
    is_half = ai['is_half_face']
    
    if is_half:
        scale = ai['midline_length'] / bi['midline_length']
        scale = np.clip(scale, 0.5, 2.0)
        
        angle_rad = ai['midline_angle'] - bi['midline_angle']
        angle_deg = -np.degrees(angle_rad)
        angle_deg = np.clip(angle_deg, -15, 15)
        
        ax, ay = bi['midline_center_x'], bi['midline_center_y']
        tx, ty = ai['midline_center_x'], ai['midline_center_y']
    else:
        scale = ai['eye_dist'] / bi['eye_dist']
        scale = np.clip(scale, 0.5, 2.0)
        
        angle_rad = ai['eye_angle'] - bi['eye_angle']
        angle_deg = -np.degrees(angle_rad)
        angle_deg = np.clip(angle_deg, -15, 15)
        
        ax, ay = bi['eyes_center_x'], bi['eyes_center_y']
        tx, ty = ai['eyes_center_x'], ai['eyes_center_y']
    
    M = cv2.getRotationMatrix2D((ax, ay), angle_deg, scale)
    
    nax = M[0,0]*ax + M[0,1]*ay + M[0,2]
    nay = M[1,0]*ax + M[1,1]*ay + M[1,2]
    
    M[0,2] += tx - nax
    M[1,2] += ty - nay
    
    before_aligned = cv2.warpAffine(before, M, (after.shape[1], after.shape[0]), 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(255, 255, 255))
    
    return before_aligned, "반모(정중선)" if is_half else "안모(눈)"


# ============ 사진 비교 ============

def process_single_photo(before_path, after_path, logo=None):
    """단일 사진 쌍 처리"""
    before_img = cv2.imread(before_path)
    after_img = cv2.imread(after_path)
    
    if before_img is None or after_img is None:
        return None, "이미지를 읽을 수 없습니다"
    
    after = resize_if_needed(after_img)
    before = resize_if_needed(before_img)
    
    if before.shape[:2] != after.shape[:2]:
        before = resize_and_crop_to_match(before, after)
    
    bi = get_face_info(before)
    ai = get_face_info(after)
    
    before_aligned, align_type = align_images(before, after, bi, ai)
    
    h, w = after.shape[:2]
    margin = 0.05
    cx1, cy1 = int(w * margin), int(h * margin)
    cx2, cy2 = int(w * (1 - margin)), int(h * (1 - margin))
    before_crop = before_aligned[cy1:cy2, cx1:cx2]
    after_crop = after[cy1:cy2, cx1:cx2]
    
    bf, af = match_brightness(before_crop, after_crop)
    
    th, tw = af.shape[:2]
    fw, fh = (tw // 16) * 16, (th // 16) * 16
    bf = cv2.resize(bf, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    af = cv2.resize(af, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    
    fps = 30
    before_frames, dissolve_frames, after_frames = 39, 12, 39
    total_frames = before_frames + dissolve_frames + after_frames
    
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
    
    for i in range(total_frames):
        if i < before_frames:
            frame = bf.copy()
        elif i < before_frames + dissolve_frames:
            alpha = (i - before_frames) / dissolve_frames
            frame = cv2.addWeighted(bf, 1-alpha, af, alpha, 0)
        else:
            frame = af.copy()
        if logo is not None:
            frame = add_logo(frame, logo)
        out.write(frame)
    
    out.release()
    
    detect_b = "✓" if bi['detected'] else "✗"
    detect_a = "✓" if ai['detected'] else "✗"
    info = f"{fw}×{fh} | {align_type} | 전({detect_b}) 후({detect_a})"
    
    return output_path, info

def create_photo_comparison(before_files, after_files, logo_img=None):
    """사진 배치 처리"""
    if not before_files or not after_files:
        return None, None, "BEFORE와 AFTER 파일을 모두 업로드해주세요"
    
    # 파일 정렬
    before_list = sorted(before_files, key=lambda x: natural_sort_key(Path(x).name))
    after_list = sorted(after_files, key=lambda x: natural_sort_key(Path(x).name))
    
    if len(before_list) != len(after_list):
        return None, None, f"파일 수가 다릅니다: BEFORE {len(before_list)}개, AFTER {len(after_list)}개"
    
    # 로고 처리
    logo = None
    if logo_img is not None:
        if len(logo_img.shape) == 3 and logo_img.shape[2] == 4:
            logo = cv2.cvtColor(logo_img, cv2.COLOR_RGBA2BGRA)
        else:
            logo = cv2.cvtColor(logo_img, cv2.COLOR_RGB2BGR)
    
    results = []
    logs = []
    
    for idx, (bf, af) in enumerate(zip(before_list, after_list)):
        bf_name = Path(bf).stem
        af_name = Path(af).stem
        logs.append(f"[{idx+1}/{len(before_list)}] {bf_name} ↔ {af_name}")
        
        try:
            output_path, info = process_single_photo(bf, af, logo)
            if output_path:
                results.append((output_path, f"{bf_name}_{af_name}.mp4"))
                logs[-1] += f" ✓ {info}"
            else:
                logs[-1] += f" ✗ {info}"
        except Exception as e:
            logs[-1] += f" ✗ 오류: {str(e)}"
    
    log_text = "\n".join(logs)
    
    if len(results) == 0:
        return None, None, log_text + "\n\n처리된 파일이 없습니다"
    elif len(results) == 1:
        # 1개: 바로 비디오 표시
        return results[0][0], None, log_text
    else:
        # 2개 이상: ZIP 생성
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "photo_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for video_path, name in results:
                zf.write(video_path, name)
        return None, zip_path, log_text + f"\n\n총 {len(results)}개 파일이 ZIP으로 저장되었습니다"


# ============ 영상 비교 ============

def get_video_rotation(video_path):
    """ffprobe를 사용해 영상의 회전 메타데이터 확인"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                if 'side_data_list' in stream:
                    for side_data in stream['side_data_list']:
                        if 'rotation' in side_data:
                            return int(side_data['rotation'])
                if 'tags' in stream and 'rotate' in stream['tags']:
                    return int(stream['tags']['rotate'])
    except Exception as e:
        print(f"ffprobe error: {e}")
    return 0

def rotate_frame(frame, rotation):
    """회전 각도에 따라 프레임 회전"""
    if rotation == 90 or rotation == -270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == -90 or rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def add_label(frame, text, position='top'):
    """프레임에 라벨 추가"""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(w, h) / 500)
    thickness = max(2, int(font_scale * 2))
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    padding = 10
    if position == 'top':
        x = (w - text_w) // 2
        y = text_h + padding + 20
    else:
        x = (w - text_w) // 2
        y = h - padding - 20
    
    cv2.rectangle(frame, (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + baseline + padding), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return frame

def get_middle_frame(cap, rotation):
    """중간 프레임 추출"""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_idx = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 리셋
    
    if ret:
        frame = rotate_frame(frame, rotation)
        return frame
    return None

def calculate_balanced_transforms(bi, ai):
    """양쪽 반반씩 조절하는 변환 계산"""
    is_half = ai['is_half_face']
    
    if is_half:
        # 반모: 정중선 기준
        b_size = bi['midline_length']
        a_size = ai['midline_length']
        b_angle = bi['midline_angle']
        a_angle = ai['midline_angle']
        b_cx, b_cy = bi['midline_center_x'], bi['midline_center_y']
        a_cx, a_cy = ai['midline_center_x'], ai['midline_center_y']
    else:
        # 안모: 눈 기준
        b_size = bi['eye_dist']
        a_size = ai['eye_dist']
        b_angle = bi['eye_angle']
        a_angle = ai['eye_angle']
        b_cx, b_cy = bi['eyes_center_x'], bi['eyes_center_y']
        a_cx, a_cy = ai['eyes_center_x'], ai['eyes_center_y']
    
    # 중간값 계산
    target_size = (b_size + a_size) / 2
    target_angle = (b_angle + a_angle) / 2
    
    # 각각의 스케일 계산 (중간 크기로 맞추기)
    scale_b = target_size / b_size if b_size > 0 else 1.0
    scale_a = target_size / a_size if a_size > 0 else 1.0
    
    # 스케일 제한 (0.5 ~ 2.0)
    scale_b = np.clip(scale_b, 0.5, 2.0)
    scale_a = np.clip(scale_a, 0.5, 2.0)
    
    # 회전 각도 (중간 각도로 맞추기)
    angle_b = -np.degrees(target_angle - b_angle)
    angle_a = -np.degrees(target_angle - a_angle)
    
    # 회전 제한
    angle_b = np.clip(angle_b, -15, 15)
    angle_a = np.clip(angle_a, -15, 15)
    
    return {
        'scale_b': scale_b,
        'scale_a': scale_a,
        'angle_b': angle_b,
        'angle_a': angle_a,
        'b_center': (b_cx, b_cy),
        'a_center': (a_cx, a_cy),
        'is_half': is_half
    }

def apply_transform(frame, scale, angle, center):
    """프레임에 변환 적용"""
    h, w = frame.shape[:2]
    cx, cy = center
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    
    # 변환 후 중심이 프레임 중앙에 오도록 조정
    new_cx = M[0,0]*cx + M[0,1]*cy + M[0,2]
    new_cy = M[1,0]*cx + M[1,1]*cy + M[1,2]
    
    M[0,2] += w/2 - new_cx
    M[1,2] += h/2 - new_cy
    
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def process_single_video(before_path, after_path, add_labels_flag=True):
    """단일 영상 쌍 처리 (중간 프레임 분석, 양쪽 반반 스케일)"""
    
    rotation_b = get_video_rotation(before_path)
    rotation_a = get_video_rotation(after_path)
    
    cap_before = cv2.VideoCapture(before_path)
    cap_after = cv2.VideoCapture(after_path)
    
    if not cap_before.isOpened() or not cap_after.isOpened():
        return None, "영상을 열 수 없습니다"
    
    fps_before = cap_before.get(cv2.CAP_PROP_FPS)
    fps_after = cap_after.get(cv2.CAP_PROP_FPS)
    fps = min(fps_before, fps_after, 30)
    
    frame_count_before = int(cap_before.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_after = int(cap_after.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 중간 프레임에서 얼굴 정보 추출
    mid_frame_b = get_middle_frame(cap_before, rotation_b)
    mid_frame_a = get_middle_frame(cap_after, rotation_a)
    
    if mid_frame_b is None or mid_frame_a is None:
        return None, "영상 프레임을 읽을 수 없습니다"
    
    # 리사이즈
    mid_frame_b = resize_if_needed(mid_frame_b, VIDEO_MAX_SIZE)
    mid_frame_a = resize_if_needed(mid_frame_a, VIDEO_MAX_SIZE)
    
    h_b, w_b = mid_frame_b.shape[:2]
    h_a, w_a = mid_frame_a.shape[:2]
    
    # 얼굴 정보 추출
    before_info = get_face_info(mid_frame_b)
    after_info = get_face_info(mid_frame_a)
    
    # 양쪽 반반 변환 계산
    transforms = calculate_balanced_transforms(before_info, after_info)
    
    # 최종 출력 크기 (둘 다 변환 후 같은 크기로)
    # 변환 후 크기를 고려해서 공통 크기 결정
    target_h = max(h_b, h_a)
    target_w = max(w_b, w_a)
    
    # 구분선 너비
    divider_width = 4
    
    # 최종 크기 (16의 배수)
    panel_w = (target_w // 16) * 16
    panel_h = (target_h // 16) * 16
    final_w = panel_w * 2 + divider_width
    final_h = panel_h
    
    # 출력 설정
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "side_by_side.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (final_w, final_h))
    
    max_frames = max(frame_count_before, frame_count_after)
    
    frame_idx = 0
    last_frame_b = None
    last_frame_a = None
    
    while frame_idx < max_frames:
        # Before 프레임
        if frame_idx < frame_count_before:
            ret_b, frame_b = cap_before.read()
            if ret_b:
                frame_b = rotate_frame(frame_b, rotation_b)
                last_frame_b = frame_b.copy()
        else:
            frame_b = last_frame_b
        
        # After 프레임
        if frame_idx < frame_count_after:
            ret_a, frame_a = cap_after.read()
            if ret_a:
                frame_a = rotate_frame(frame_a, rotation_a)
                last_frame_a = frame_a.copy()
        else:
            frame_a = last_frame_a
        
        if frame_b is None or frame_a is None:
            break
        
        # 리사이즈
        frame_b = resize_if_needed(frame_b, VIDEO_MAX_SIZE)
        frame_a = resize_if_needed(frame_a, VIDEO_MAX_SIZE)
        
        # 현재 프레임의 얼굴 정보로 중심점 업데이트 (옵션 - 더 정확하지만 느림)
        # 여기서는 첫 프레임 기준 사용
        
        # 변환 적용
        frame_b_transformed = apply_transform(frame_b, transforms['scale_b'], 
                                               transforms['angle_b'], transforms['b_center'])
        frame_a_transformed = apply_transform(frame_a, transforms['scale_a'], 
                                               transforms['angle_a'], transforms['a_center'])
        
        # 패널 크기로 조정
        frame_b_panel = cv2.resize(frame_b_transformed, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4)
        frame_a_panel = cv2.resize(frame_a_transformed, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 라벨 추가
        if add_labels_flag:
            frame_b_panel = add_label(frame_b_panel, "BEFORE", 'top')
            frame_a_panel = add_label(frame_a_panel, "AFTER", 'top')
        
        # 합치기
        combined = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        combined[:, :panel_w] = frame_b_panel
        combined[:, panel_w:panel_w+divider_width] = [255, 255, 255]
        combined[:, panel_w+divider_width:] = frame_a_panel
        
        out.write(combined)
        frame_idx += 1
    
    cap_before.release()
    cap_after.release()
    out.release()
    
    duration = max_frames / fps
    detect_b = "✓" if before_info['detected'] else "✗"
    detect_a = "✓" if after_info['detected'] else "✗"
    align_type = "반모(정중선)" if transforms['is_half'] else "안모(눈)"
    scale_info = f"스케일: B×{transforms['scale_b']:.2f}, A×{transforms['scale_a']:.2f}"
    rot_info = f" | 회전보정: B({rotation_b}°) A({rotation_a}°)" if rotation_b != 0 or rotation_a != 0 else ""
    
    info = f"{final_w}×{final_h} | {duration:.1f}초 | {align_type}\n{scale_info}{rot_info}\n얼굴검출: 전({detect_b}) 후({detect_a})"
    
    return output_path, info

def create_video_comparison(before_files, after_files, add_labels_flag=True):
    """영상 배치 처리"""
    if not before_files or not after_files:
        return None, None, "BEFORE와 AFTER 영상을 모두 업로드해주세요"
    
    # 파일 정렬
    before_list = sorted(before_files, key=lambda x: natural_sort_key(Path(x).name))
    after_list = sorted(after_files, key=lambda x: natural_sort_key(Path(x).name))
    
    if len(before_list) != len(after_list):
        return None, None, f"파일 수가 다릅니다: BEFORE {len(before_list)}개, AFTER {len(after_list)}개"
    
    results = []
    logs = []
    
    for idx, (bf, af) in enumerate(zip(before_list, after_list)):
        bf_name = Path(bf).stem
        af_name = Path(af).stem
        logs.append(f"[{idx+1}/{len(before_list)}] {bf_name} ↔ {af_name}")
        
        try:
            output_path, info = process_single_video(bf, af, add_labels_flag)
            if output_path:
                results.append((output_path, f"{bf_name}_{af_name}.mp4"))
                logs[-1] += f" ✓"
                logs.append(f"    {info}")
            else:
                logs[-1] += f" ✗ {info}"
        except Exception as e:
            logs[-1] += f" ✗ 오류: {str(e)}"
    
    log_text = "\n".join(logs)
    
    if len(results) == 0:
        return None, None, log_text + "\n\n처리된 파일이 없습니다"
    elif len(results) == 1:
        return results[0][0], None, log_text
    else:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "video_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for video_path, name in results:
                zf.write(video_path, name)
        return None, zip_path, log_text + f"\n\n총 {len(results)}개 파일이 ZIP으로 저장되었습니다"


# ============ UI ============

custom_css = """
.gradio-container { max-width: 1000px !important; margin: auto !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Dental B&A", css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align:center'>🦷 치과 전후 비교</h1>")
    gr.Markdown("<p style='text-align:center;color:#666'>파일명 순서대로 자동 매칭 (1↔1, 2↔2, ...)</p>")
    
    with gr.Tabs():
        # ===== 탭 1: 사진 비교 =====
        with gr.TabItem("📷 사진 비교"):
            gr.Markdown("<p style='text-align:center;color:#888'>사진 → 디졸브 MP4 생성</p>")
            
            with gr.Row():
                photo_before = gr.File(label="BEFORE 사진들", file_count="multiple", file_types=["image"])
                photo_after = gr.File(label="AFTER 사진들", file_count="multiple", file_types=["image"])
            
            with gr.Accordion("로고 추가 (선택)", open=False):
                logo_input = gr.Image(label="PNG 투명 배경 지원", type="numpy")
            
            photo_btn = gr.Button("🎬 영상 생성", variant="primary", size="lg")
            
            with gr.Row():
                photo_video_out = gr.Video(label="결과 (1개일 때)")
                photo_zip_out = gr.File(label="ZIP 다운로드 (2개 이상)")
            
            photo_log = gr.Textbox(label="처리 로그", lines=6)
            
            photo_btn.click(
                fn=create_photo_comparison,
                inputs=[photo_before, photo_after, logo_input],
                outputs=[photo_video_out, photo_zip_out, photo_log]
            )
        
        # ===== 탭 2: 영상 비교 =====
        with gr.TabItem("🎬 영상 비교"):
            gr.Markdown("<p style='text-align:center;color:#888'>영상 → 좌우 비교 (중간 프레임 분석, 양쪽 반반 스케일)</p>")
            
            with gr.Row():
                video_before = gr.File(label="BEFORE 영상들", file_count="multiple", file_types=["video"])
                video_after = gr.File(label="AFTER 영상들", file_count="multiple", file_types=["video"])
            
            add_labels_check = gr.Checkbox(label="BEFORE/AFTER 라벨 표시", value=True)
            
            video_btn = gr.Button("🎬 영상 합치기", variant="primary", size="lg")
            
            with gr.Row():
                video_video_out = gr.Video(label="결과 (1개일 때)")
                video_zip_out = gr.File(label="ZIP 다운로드 (2개 이상)")
            
            video_log = gr.Textbox(label="처리 로그", lines=6)
            
            video_btn.click(
                fn=create_video_comparison,
                inputs=[video_before, video_after, add_labels_check],
                outputs=[video_video_out, video_zip_out, video_log]
            )

demo.launch()
