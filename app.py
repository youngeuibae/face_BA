import gradio as gr
import cv2
import numpy as np
import tempfile
import os

MAX_SIZE = 1920

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

def get_oral_teeth_info(img):
    """구강 사진에서 치아 영역 감지 (밝기+채도 기반)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img, w_img = img.shape[:2]
    
    v = hsv[:,:,2]
    s = hsv[:,:,1]
    
    # 치아 마스크: 밝고 채도 낮음
    teeth_mask = ((v > 180) & (s < 60)).astype(np.uint8) * 255
    
    # 중앙 영역만 (테두리 15% 제외)
    roi = np.zeros_like(teeth_mask)
    margin = 0.15
    roi[int(h_img*margin):int(h_img*(1-margin)), 
        int(w_img*margin):int(w_img*(1-margin))] = 255
    teeth_mask = cv2.bitwise_and(teeth_mask, roi)
    
    # 모폴로지
    kernel = np.ones((7,7), np.uint8)
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel)
    teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, kernel)
    
    # 컨투어
    contours, _ = cv2.findContours(teeth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 컨투어
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # 최소 면적 체크 (이미지의 1% 이상)
        if area > (w_img * h_img) * 0.01:
            x, y, bw, bh = cv2.boundingRect(largest)
            
            M = cv2.moments(largest)
            cx = M['m10'] / M['m00'] if M['m00'] > 0 else x + bw/2
            cy = M['m01'] / M['m00'] if M['m00'] > 0 else y + bh/2
            
            return {
                'cx': float(cx), 'cy': float(cy),
                'bw': float(bw), 'bh': float(bh),
                'area': float(area),
                'detected': True,
                'type': 'oral'
            }
    
    return None

def get_face_info(img):
    """얼굴 감지 (MediaPipe) 또는 구강 사진 감지"""
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
                
                # 눈 좌표
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
                
                # 정중선
                nose_tip = lm[4]
                nose_x, nose_y = nose_tip.x * w, nose_tip.y * h
                
                upper_lip = lm[0]
                lower_lip = lm[17]
                chin = lm[152]
                
                midline_center_x = (upper_lip.x * w + lower_lip.x * w) / 2
                midline_center_y = (upper_lip.y * h + lower_lip.y * h) / 2
                midline_length = np.sqrt((chin.x*w - nose_x)**2 + (chin.y*h - nose_y)**2)
                midline_angle = np.arctan2(chin.y*h - nose_y, chin.x*w - nose_x)
                
                # 안모/반모 판별
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
                    'detected': True,
                    'type': 'face'
                }
    except Exception as e:
        print(f"MediaPipe error: {e}")
    
    # 얼굴 감지 실패 → 구강 사진 시도
    oral = get_oral_teeth_info(img)
    if oral:
        return oral
    
    # 완전 실패
    h, w = img.shape[:2]
    return {
        'cx': w / 2, 'cy': h / 2,
        'bw': w * 0.6, 'bh': h * 0.6,
        'detected': False,
        'type': 'fallback'
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
    """이미지 유형에 따른 정렬"""
    
    # 구강 사진 (oral type)
    if bi.get('type') == 'oral' or ai.get('type') == 'oral':
        # 치아 너비 기준 스케일
        scale = ai.get('bw', 1) / bi.get('bw', 1)
        scale = np.clip(scale, 0.7, 1.4)
        
        # 중심점 기준 정렬
        ax, ay = bi['cx'], bi['cy']
        tx, ty = ai['cx'], ai['cy']
        
        M = cv2.getRotationMatrix2D((ax, ay), 0, float(scale))
        nax = M[0,0]*ax + M[0,1]*ay + M[0,2]
        nay = M[1,0]*ax + M[1,1]*ay + M[1,2]
        M[0,2] += tx - nax
        M[1,2] += ty - nay
        
        before_aligned = cv2.warpAffine(before, M, (after.shape[1], after.shape[0]), 
                                         borderMode=cv2.BORDER_REPLICATE)
        return before_aligned, f"구강(스케일:{scale:.2f})"
    
    # 얼굴 사진 - 반모
    if ai.get('is_half_face', False):
        scale = ai['midline_length'] / bi['midline_length']
        scale = np.clip(scale, 0.8, 1.25)
        
        angle_rad = ai['midline_angle'] - bi['midline_angle']
        angle_deg = -np.degrees(angle_rad)
        angle_deg = np.clip(angle_deg, -10, 10)
        
        ax, ay = bi['midline_center_x'], bi['midline_center_y']
        tx, ty = ai['midline_center_x'], ai['midline_center_y']
        align_type = "반모(정중선)"
    else:
        # 얼굴 사진 - 안모
        scale = ai['eye_dist'] / bi['eye_dist']
        scale = np.clip(scale, 0.8, 1.25)
        
        angle_rad = ai['eye_angle'] - bi['eye_angle']
        angle_deg = -np.degrees(angle_rad)
        angle_deg = np.clip(angle_deg, -10, 10)
        
        ax, ay = bi['eyes_center_x'], bi['eyes_center_y']
        tx, ty = ai['eyes_center_x'], ai['eyes_center_y']
        align_type = "안모(눈)"
    
    M = cv2.getRotationMatrix2D((ax, ay), angle_deg, scale)
    nax = M[0,0]*ax + M[0,1]*ay + M[0,2]
    nay = M[1,0]*ax + M[1,1]*ay + M[1,2]
    M[0,2] += tx - nax
    M[1,2] += ty - nay
    
    before_aligned = cv2.warpAffine(before, M, (after.shape[1], after.shape[0]), 
                                     borderMode=cv2.BORDER_REPLICATE)
    
    return before_aligned, align_type

def create_video(before_img, after_img, logo_img=None):
    if before_img is None or after_img is None:
        return None, "전/후 사진을 모두 업로드해주세요"
    
    try:
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
        output_path = os.path.join(temp_dir, "dental_comparison.mp4")
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
        type_b = bi.get('type', '?')
        type_a = ai.get('type', '?')
        return output_path, f"{fw}×{fh} | 3.0초 | {align_type}\n감지: 전({detect_b}/{type_b}) 후({detect_a}/{type_a})"
    
    except Exception as e:
        import traceback
        return None, f"오류: {str(e)}\n{traceback.format_exc()}"

with gr.Blocks(title="Dental B&A") as demo:
    gr.Markdown("<h1 style='text-align:center'>🦷 치과 전후 비교</h1>")
    gr.Markdown("<p style='text-align:center;color:#666'>얼굴/구강 사진 자동 감지 → 정렬 → MP4 생성</p>")
    
    with gr.Row():
        before_input = gr.Image(label="BEFORE", type="numpy")
        after_input = gr.Image(label="AFTER", type="numpy")
    
    with gr.Accordion("로고 추가 (선택)", open=False):
        logo_input = gr.Image(label="PNG 투명 배경 지원", type="numpy")
    
    generate_btn = gr.Button("영상 생성", variant="primary")
    
    with gr.Row():
        video_output = gr.Video(label="결과")
        status_output = gr.Textbox(label="정보", lines=3)
    
    generate_btn.click(
        fn=create_video, 
        inputs=[before_input, after_input, logo_input], 
        outputs=[video_output, status_output]
    )

demo.launch()
