import gradio as gr
import cv2
import numpy as np
import tempfile
import os

MAX_SIZE = 1920

def resize_if_needed(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        if w > h:
            new_w = MAX_SIZE
            new_h = int(h * MAX_SIZE / w)
        else:
            new_h = MAX_SIZE
            new_w = int(w * MAX_SIZE / h)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img

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

def get_oral_teeth_info(img):
    """구강 사진 치아 정보 (Otsu 기반)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img.shape[:2]
    
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    margin = 0.12
    roi = np.zeros_like(otsu_mask)
    roi[int(h_img*margin):int(h_img*(1-margin)), 
        int(w_img*margin):int(w_img*(1-margin))] = 255
    teeth_mask = cv2.bitwise_and(otsu_mask, roi)
    
    kernel = np.ones((5,5), np.uint8)
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel)
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, kernel)
    
    row_sum = np.sum(teeth_mask > 0, axis=1)
    row_smooth = np.convolve(row_sum, np.ones(11)/11, mode='same')
    
    search_start = int(h_img * 0.35)
    search_end = int(h_img * 0.65)
    search_region = row_smooth[search_start:search_end]
    
    occlusal_y = search_start + np.argmin(search_region) if len(search_region) > 0 else h_img // 2
    
    upper_mask = teeth_mask.copy()
    upper_mask[int(occlusal_y):, :] = 0
    
    contours, _ = cv2.findContours(upper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(teeth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (w_img * h_img) * 0.01:
        return None
    
    x, y, bw, bh = cv2.boundingRect(largest)
    M = cv2.moments(largest)
    cx = M['m10'] / M['m00'] if M['m00'] > 0 else x + bw/2
    cy = M['m01'] / M['m00'] if M['m00'] > 0 else y + bh/2
    
    return {'cx': float(cx), 'cy': float(cy), 'bw': float(bw), 'bh': float(bh), 'detected': True, 'type': 'oral'}

def get_face_info(img):
    """얼굴 또는 구강 감지"""
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                    min_detection_confidence=0.1) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                h, w = img.shape[:2]
                
                left_eye_idx = [33, 133, 160, 159, 158, 144, 145, 153]
                right_eye_idx = [362, 263, 387, 386, 385, 373, 374, 380]
                
                left_eye_x = np.mean([lm[i].x for i in left_eye_idx]) * w
                left_eye_y = np.mean([lm[i].y for i in left_eye_idx]) * h
                right_eye_x = np.mean([lm[i].x for i in right_eye_idx]) * w
                right_eye_y = np.mean([lm[i].y for i in right_eye_idx]) * h
                
                eye_dist = np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
                eye_angle = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x)
                eyes_cx = (left_eye_x + right_eye_x) / 2
                eyes_cy = (left_eye_y + right_eye_y) / 2
                
                return {
                    'cx': eyes_cx, 'cy': eyes_cy,
                    'eye_dist': eye_dist, 'eye_angle': eye_angle,
                    'detected': True, 'type': 'face'
                }
    except:
        pass
    
    oral = get_oral_teeth_info(img)
    if oral:
        return oral
    
    h, w = img.shape[:2]
    return {'cx': w/2, 'cy': h/2, 'bw': w*0.6, 'detected': False, 'type': 'fallback'}

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
        alpha = logo_resized[:, :, 3:4] / 255.0
        logo_bgr = logo_resized[:, :, :3]
        roi = frame[margin:margin+logo_h, margin:margin+logo_w]
        frame[margin:margin+logo_h, margin:margin+logo_w] = (logo_bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        frame[margin:margin+logo_h, margin:margin+logo_w] = logo_resized
    return frame

def add_label(img, text, position='top'):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, min(w, h) / 900)
    thickness = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    
    pad = 8
    bx = (w - tw - pad*2) // 2
    by = 12 if position == 'top' else h - th - pad*2 - 12
    
    cv2.rectangle(img, (bx, by), (bx + tw + pad*2, by + th + pad*2), (0,0,0), -1)
    cv2.putText(img, text, (bx + pad, by + th + pad), font, scale, (255,255,255), thickness)
    return img

def align_balanced(before, after, bi, ai):
    """균형 스케일링: 두 이미지를 중간 크기로"""
    h_img, w_img = before.shape[:2]
    target_cx, target_cy = w_img / 2, h_img / 2
    
    # 구강 사진
    if bi.get('type') == 'oral' and ai.get('type') == 'oral':
        ratio = ai['bw'] / bi['bw']
        ratio = np.clip(ratio, 0.5, 2.0)
        
        mid_scale = np.sqrt(ratio)
        b_scale = np.clip(mid_scale, 0.7, 1.4)
        a_scale = np.clip(1.0 / mid_scale, 0.7, 1.4)
        
        # Before 변환
        M_b = cv2.getRotationMatrix2D((bi['cx'], bi['cy']), 0, b_scale)
        new_bx = M_b[0,0]*bi['cx'] + M_b[0,1]*bi['cy'] + M_b[0,2]
        new_by = M_b[1,0]*bi['cx'] + M_b[1,1]*bi['cy'] + M_b[1,2]
        M_b[0,2] += target_cx - new_bx
        M_b[1,2] += target_cy - new_by
        
        # After 변환
        M_a = cv2.getRotationMatrix2D((ai['cx'], ai['cy']), 0, a_scale)
        new_ax = M_a[0,0]*ai['cx'] + M_a[0,1]*ai['cy'] + M_a[0,2]
        new_ay = M_a[1,0]*ai['cx'] + M_a[1,1]*ai['cy'] + M_a[1,2]
        M_a[0,2] += target_cx - new_ax
        M_a[1,2] += target_cy - new_ay
        
        before_out = cv2.warpAffine(before, M_b, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)
        after_out = cv2.warpAffine(after, M_a, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)
        
        return before_out, after_out, f"구강(균형 {b_scale:.2f}/{a_scale:.2f})"
    
    # 얼굴 사진
    if bi.get('type') == 'face' and ai.get('type') == 'face':
        ratio = ai['eye_dist'] / bi['eye_dist']
        ratio = np.clip(ratio, 0.5, 2.0)
        
        mid_scale = np.sqrt(ratio)
        b_scale = np.clip(mid_scale, 0.8, 1.25)
        a_scale = np.clip(1.0 / mid_scale, 0.8, 1.25)
        
        # 각도 차이
        angle_diff = ai['eye_angle'] - bi['eye_angle']
        b_angle = np.degrees(angle_diff / 2)
        a_angle = -np.degrees(angle_diff / 2)
        b_angle = np.clip(b_angle, -10, 10)
        a_angle = np.clip(a_angle, -10, 10)
        
        M_b = cv2.getRotationMatrix2D((bi['cx'], bi['cy']), b_angle, b_scale)
        new_bx = M_b[0,0]*bi['cx'] + M_b[0,1]*bi['cy'] + M_b[0,2]
        new_by = M_b[1,0]*bi['cx'] + M_b[1,1]*bi['cy'] + M_b[1,2]
        M_b[0,2] += target_cx - new_bx
        M_b[1,2] += target_cy - new_by
        
        M_a = cv2.getRotationMatrix2D((ai['cx'], ai['cy']), a_angle, a_scale)
        new_ax = M_a[0,0]*ai['cx'] + M_a[0,1]*ai['cy'] + M_a[0,2]
        new_ay = M_a[1,0]*ai['cx'] + M_a[1,1]*ai['cy'] + M_a[1,2]
        M_a[0,2] += target_cx - new_ax
        M_a[1,2] += target_cy - new_ay
        
        before_out = cv2.warpAffine(before, M_b, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)
        after_out = cv2.warpAffine(after, M_a, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)
        
        return before_out, after_out, f"안모(균형 {b_scale:.2f}/{a_scale:.2f})"
    
    # 폴백: 변환 없음
    return before.copy(), after.copy(), "미감지"

def process_images(before_img, after_img, logo_img=None):
    """이미지 전처리 및 정렬"""
    before = cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR)
    after = cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR)
    
    after = resize_if_needed(after)
    before = resize_if_needed(before)
    
    if before.shape[:2] != after.shape[:2]:
        before = resize_and_crop_to_match(before, after)
    
    logo = None
    if logo_img is not None:
        if len(logo_img.shape) == 3 and logo_img.shape[2] == 4:
            logo = cv2.cvtColor(logo_img, cv2.COLOR_RGBA2BGRA)
        else:
            logo = cv2.cvtColor(logo_img, cv2.COLOR_RGB2BGR)
    
    bi = get_face_info(before)
    ai = get_face_info(after)
    
    before_aligned, after_aligned, align_type = align_balanced(before, after, bi, ai)
    
    # 크롭 및 밝기 보정
    h, w = after_aligned.shape[:2]
    margin = 0.05
    cx1, cy1 = int(w * margin), int(h * margin)
    cx2, cy2 = int(w * (1 - margin)), int(h * (1 - margin))
    
    bf = before_aligned[cy1:cy2, cx1:cx2]
    af = after_aligned[cy1:cy2, cx1:cx2]
    
    bf, af = match_brightness(bf, af)
    
    # 16배수 맞춤
    th, tw = af.shape[:2]
    fw, fh = (tw // 16) * 16, (th // 16) * 16
    bf = cv2.resize(bf, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    af = cv2.resize(af, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    
    return bf, af, logo, bi, ai, align_type, fw, fh

def create_dissolve_video(before_img, after_img, logo_img=None):
    """디졸브 영상"""
    if before_img is None or after_img is None:
        return None, "전/후 사진을 모두 업로드해주세요"
    
    try:
        bf, af, logo, bi, ai, align_type, fw, fh = process_images(before_img, after_img, logo_img)
        
        fps = 30
        before_frames, dissolve_frames, after_frames = 39, 12, 39
        
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "dissolve.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
        
        for i in range(before_frames + dissolve_frames + after_frames):
            if i < before_frames:
                frame = bf.copy()
            elif i < before_frames + dissolve_frames:
                alpha = (i - before_frames) / dissolve_frames
                frame = cv2.addWeighted(bf, 1-alpha, af, alpha, 0)
            else:
                frame = af.copy()
            if logo:
                frame = add_logo(frame, logo)
            out.write(frame)
        
        out.release()
        
        return output_path, f"{fw}×{fh} | 3.0초 | {align_type}"
    except Exception as e:
        return None, f"오류: {str(e)}"

def create_sidebyside(before_img, after_img, logo_img=None):
    """좌우 비교 이미지"""
    if before_img is None or after_img is None:
        return None, "전/후 사진을 모두 업로드해주세요"
    
    try:
        bf, af, logo, bi, ai, align_type, fw, fh = process_images(before_img, after_img, logo_img)
        
        bf_labeled = add_label(bf.copy(), "BEFORE")
        af_labeled = add_label(af.copy(), "AFTER")
        
        divider = np.ones((fh, 4, 3), dtype=np.uint8) * 255
        combined = np.hstack([bf_labeled, divider, af_labeled])
        
        if logo:
            combined = add_logo(combined, logo, width_ratio=0.15)
        
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        return combined_rgb, f"{combined.shape[1]}×{combined.shape[0]} | {align_type}"
    except Exception as e:
        return None, f"오류: {str(e)}"

with gr.Blocks(title="Dental B&A") as demo:
    gr.Markdown("<h1 style='text-align:center'>🦷 치과 전후 비교</h1>")
    gr.Markdown("<p style='text-align:center;color:#666'>얼굴/구강 자동 감지 → 균형 스케일 정렬</p>")
    
    with gr.Row():
        before_input = gr.Image(label="BEFORE", type="numpy")
        after_input = gr.Image(label="AFTER", type="numpy")
    
    with gr.Accordion("로고 (선택)", open=False):
        logo_input = gr.Image(label="PNG 투명 배경 지원", type="numpy")
    
    with gr.Row():
        dissolve_btn = gr.Button("🎬 디졸브 영상", variant="primary")
        sidebyside_btn = gr.Button("🖼️ 좌우 비교")
    
    with gr.Row():
        video_output = gr.Video(label="영상")
        image_output = gr.Image(label="이미지")
    
    status_output = gr.Textbox(label="정보", lines=1)
    
    dissolve_btn.click(create_dissolve_video, [before_input, after_input, logo_input], [video_output, status_output])
    sidebyside_btn.click(create_sidebyside, [before_input, after_input, logo_input], [image_output, status_output])

demo.launch()
