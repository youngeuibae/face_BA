import gradio as gr
import cv2
import numpy as np
import tempfile
import os
import zipfile

MAX_SIZE = 1920

# ===== 유틸리티 =====
def resize_if_needed(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return img

def resize_and_crop_to_match(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    ratio1, ratio2 = w1/h1, w2/h2
    
    if ratio1 > ratio2:
        new_w = int(h2 * ratio1)
        new_h = h2
    else:
        new_w = w2
        new_h = int(w2 / ratio1)
    
    img1_resized = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    sx, sy = (new_w - w2) // 2, (new_h - h2) // 2
    return img1_resized[sy:sy+h2, sx:sx+w2]

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
    
    if logo_resized.shape[2] == 4:
        alpha = logo_resized[:, :, 3:4] / 255.0
        roi = frame[margin:margin+logo_h, margin:margin+logo_w]
        frame[margin:margin+logo_h, margin:margin+logo_w] = (logo_resized[:,:,:3] * alpha + roi * (1-alpha)).astype(np.uint8)
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

# ===== 감지 =====
def detect_face(img):
    """MediaPipe 얼굴 감지 → 안모/반모 판별"""
    try:
        import mediapipe as mp
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.3) as fm:
            results = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None
            
            lm = results.multi_face_landmarks[0].landmark
            h, w = img.shape[:2]
            
            left_eye = np.mean([[lm[i].x*w, lm[i].y*h] for i in [33,133,160,159,158,144,145,153]], axis=0)
            right_eye = np.mean([[lm[i].x*w, lm[i].y*h] for i in [362,263,387,386,385,373,374,380]], axis=0)
            
            eye_dist = np.linalg.norm(right_eye - left_eye)
            eye_angle = np.arctan2(right_eye[1]-left_eye[1], right_eye[0]-left_eye[0])
            eye_center = (left_eye + right_eye) / 2
            
            nose = np.array([lm[4].x*w, lm[4].y*h])
            chin = np.array([lm[152].x*w, lm[152].y*h])
            upper_lip = np.array([lm[0].x*w, lm[0].y*h])
            lower_lip = np.array([lm[17].x*w, lm[17].y*h])
            
            midline_len = np.linalg.norm(chin - nose)
            midline_angle = np.arctan2(chin[1]-nose[1], chin[0]-nose[0])
            midline_center = (upper_lip + lower_lip) / 2
            
            face_pts = [lm[i] for i in [10, 152, 234, 454]]
            xs = [p.x*w for p in face_pts]
            ys = [p.y*h for p in face_pts]
            aspect = (max(ys)-min(ys)) / (max(xs)-min(xs)) if max(xs)!=min(xs) else 1
            
            is_half = aspect < 0.9
            
            return {
                'type': 'half_face' if is_half else 'full_face',
                'eye_dist': eye_dist, 'eye_angle': eye_angle, 'eye_center': eye_center,
                'midline_len': midline_len, 'midline_angle': midline_angle, 'midline_center': midline_center
            }
    except Exception as e:
        print(f"Face detection error: {e}")
        return None

def detect_oral(img):
    """구강(치아) 감지 - 무게중심으로 정중선(11-21 사이) 찾기"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    roi = np.zeros_like(mask)
    m = 0.12
    roi[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))] = 255
    mask = cv2.bitwise_and(mask, roi)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 상하악 경계 (교합면)
    row_sum = np.convolve(np.sum(mask>0, axis=1), np.ones(11)/11, mode='same')
    search = row_sum[int(h*0.35):int(h*0.65)]
    occlusal_y = int(h*0.35) + np.argmin(search) if len(search) > 0 else h//2
    
    # 상악 영역
    upper = mask.copy()
    upper[occlusal_y:, :] = 0
    
    contours, _ = cv2.findContours(upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < w*h*0.01:
        return None
    
    x, y, bw, bh = cv2.boundingRect(largest)
    
    # 무게중심으로 정중선 X 좌표 (11-21 사이)
    M = cv2.moments(largest)
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
    else:
        cx = x + bw / 2
    
    return {
        'type': 'oral',
        'cx': cx,
        'cy': occlusal_y,
        'width': bw,
        'height': bh
    }

def detect_features(img):
    """통합 감지: 얼굴 → 구강 → 폴백"""
    face = detect_face(img)
    if face:
        return face
    
    oral = detect_oral(img)
    if oral:
        return oral
    
    h, w = img.shape[:2]
    return {'type': 'fallback', 'cx': w/2, 'cy': h/2, 'width': w*0.5}

# ===== 정렬 =====
def align_balanced(before, after, bi, ai):
    """균형 스케일링: 두 이미지를 중간 크기로, 치아 중앙을 이미지 중앙으로"""
    h, w = before.shape[:2]
    target = np.array([w/2, h/2])  # 이미지 중앙
    
    b_type, a_type = bi['type'], ai['type']
    
    # 안모 (Full Face)
    if b_type == 'full_face' and a_type == 'full_face':
        ratio = ai['eye_dist'] / bi['eye_dist']
        ratio = np.clip(ratio, 0.5, 2.0)
        mid_scale = np.sqrt(ratio)
        b_scale = np.clip(mid_scale, 0.8, 1.25)
        a_scale = np.clip(1/mid_scale, 0.8, 1.25)
        
        angle_diff = ai['eye_angle'] - bi['eye_angle']
        b_angle = np.degrees(angle_diff/2)
        a_angle = -np.degrees(angle_diff/2)
        
        b_anchor = bi['eye_center']
        a_anchor = ai['eye_center']
        align_type = "안모"
    
    # 반모 (Half Face)
    elif b_type == 'half_face' and a_type == 'half_face':
        ratio = ai['midline_len'] / bi['midline_len']
        ratio = np.clip(ratio, 0.5, 2.0)
        mid_scale = np.sqrt(ratio)
        b_scale = np.clip(mid_scale, 0.8, 1.25)
        a_scale = np.clip(1/mid_scale, 0.8, 1.25)
        
        angle_diff = ai['midline_angle'] - bi['midline_angle']
        b_angle = np.degrees(angle_diff/2)
        a_angle = -np.degrees(angle_diff/2)
        
        b_anchor = bi['midline_center']
        a_anchor = ai['midline_center']
        align_type = "반모"
    
    # 구강 (Oral) - 치아 중앙을 이미지 중앙으로
    elif b_type == 'oral' or a_type == 'oral':
        b_w = bi.get('width', w*0.5)
        a_w = ai.get('width', w*0.5)
        ratio = a_w / b_w
        ratio = np.clip(ratio, 0.5, 2.0)
        mid_scale = np.sqrt(ratio)
        b_scale = np.clip(mid_scale, 0.7, 1.4)
        a_scale = np.clip(1/mid_scale, 0.7, 1.4)
        
        b_anchor = np.array([bi.get('cx', w/2), bi.get('cy', h/2)])
        a_anchor = np.array([ai.get('cx', w/2), ai.get('cy', h/2)])
        b_angle, a_angle = 0, 0
        align_type = "구강"
    
    # 폴백
    else:
        return before.copy(), after.copy(), "미감지"
    
    # 변환 적용
    def transform(img, anchor, angle, scale, target):
        M = cv2.getRotationMatrix2D(tuple(anchor), angle, scale)
        new_anchor = M @ np.array([anchor[0], anchor[1], 1])
        M[0,2] += target[0] - new_anchor[0]
        M[1,2] += target[1] - new_anchor[1]
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    
    before_out = transform(before, b_anchor, b_angle, b_scale, target)
    after_out = transform(after, a_anchor, a_angle, a_scale, target)
    
    return before_out, after_out, f"{align_type}({b_scale:.2f}/{a_scale:.2f})"

# ===== 처리 =====
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
        logo = cv2.cvtColor(logo_img, cv2.COLOR_RGBA2BGRA if logo_img.shape[2]==4 else cv2.COLOR_RGB2BGR)
    
    bi = detect_features(before)
    ai = detect_features(after)
    
    before_aligned, after_aligned, align_type = align_balanced(before, after, bi, ai)
    
    # 크롭
    h, w = after_aligned.shape[:2]
    m = 0.05
    bf = before_aligned[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))]
    af = after_aligned[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))]
    
    bf, af = match_brightness(bf, af)
    
    # 16배수
    th, tw = af.shape[:2]
    fw, fh = (tw//16)*16, (th//16)*16
    bf = cv2.resize(bf, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    af = cv2.resize(af, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    
    return bf, af, logo, align_type, fw, fh

# ===== 출력 생성 =====
def create_dissolve(before_img, after_img, logo_img=None):
    """디졸브 영상"""
    if before_img is None or after_img is None:
        return None, "전/후 사진을 모두 업로드해주세요"
    
    try:
        bf, af, logo, align_type, fw, fh = process_images(before_img, after_img, logo_img)
        
        fps = 30
        frames = [39, 12, 39]
        
        temp = tempfile.mkdtemp()
        path = os.path.join(temp, "dissolve.mp4")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
        
        for i in range(sum(frames)):
            if i < frames[0]:
                frame = bf.copy()
            elif i < frames[0] + frames[1]:
                alpha = (i - frames[0]) / frames[1]
                frame = cv2.addWeighted(bf, 1-alpha, af, alpha, 0)
            else:
                frame = af.copy()
            if logo is not None:
                frame = add_logo(frame, logo)
            out.write(frame)
        
        out.release()
        return path, f"{fw}×{fh} | 3.0초 | {align_type}"
    except Exception as e:
        return None, f"오류: {e}"

def create_sidebyside(before_img, after_img, logo_img=None):
    """좌우 비교"""
    if before_img is None or after_img is None:
        return None, "전/후 사진을 모두 업로드해주세요"
    
    try:
        bf, af, logo, align_type, fw, fh = process_images(before_img, after_img, logo_img)
        
        bf_l = add_label(bf.copy(), "BEFORE")
        af_l = add_label(af.copy(), "AFTER")
        
        divider = np.ones((fh, 4, 3), dtype=np.uint8) * 255
        combined = np.hstack([bf_l, divider, af_l])
        
        if logo is not None:
            combined = add_logo(combined, logo, width_ratio=0.15)
        
        return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), f"{combined.shape[1]}×{combined.shape[0]} | {align_type}"
    except Exception as e:
        return None, f"오류: {e}"

# ===== 배치 처리 =====
def process_batch(files, output_type, logo_img=None):
    """배치 처리"""
    if not files or len(files) < 2:
        return None, "최소 2개 파일 필요 (before/after 쌍)"
    
    files = sorted(files, key=lambda x: x.name if hasattr(x, 'name') else x)
    
    if len(files) % 2 != 0:
        return None, "파일 개수가 짝수여야 합니다"
    
    temp = tempfile.mkdtemp()
    results = []
    
    for i in range(0, len(files), 2):
        before_path = files[i].name if hasattr(files[i], 'name') else files[i]
        after_path = files[i+1].name if hasattr(files[i+1], 'name') else files[i+1]
        
        before_img = cv2.cvtColor(cv2.imread(before_path), cv2.COLOR_BGR2RGB)
        after_img = cv2.cvtColor(cv2.imread(after_path), cv2.COLOR_BGR2RGB)
        
        idx = i // 2 + 1
        
        if output_type == "디졸브":
            result, _ = create_dissolve(before_img, after_img, logo_img)
            if result:
                new_path = os.path.join(temp, f"dissolve_{idx:03d}.mp4")
                os.rename(result, new_path)
                results.append(new_path)
        else:
            result, _ = create_sidebyside(before_img, after_img, logo_img)
            if result is not None:
                new_path = os.path.join(temp, f"sidebyside_{idx:03d}.png")
                cv2.imwrite(new_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                results.append(new_path)
    
    if not results:
        return None, "처리 실패"
    
    zip_path = os.path.join(temp, "batch_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for r in results:
            zf.write(r, os.path.basename(r))
    
    return zip_path, f"{len(results)}개 완료"

# ===== UI =====
with gr.Blocks(title="Dental B&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦷 치과 전후 비교")
    gr.Markdown("안모/반모/구강 자동 감지 → 균형 스케일 + 중앙 정렬")
    
    with gr.Tabs():
        with gr.Tab("단일 처리"):
            with gr.Row():
                before_input = gr.Image(label="BEFORE", type="numpy")
                after_input = gr.Image(label="AFTER", type="numpy")
            
            with gr.Accordion("로고 (선택)", open=False):
                logo_input = gr.Image(label="PNG 투명 배경", type="numpy")
            
            with gr.Row():
                dissolve_btn = gr.Button("🎬 디졸브", variant="primary")
                sidebyside_btn = gr.Button("🖼️ 좌우비교")
            
            with gr.Row():
                video_out = gr.Video(label="영상")
                image_out = gr.Image(label="이미지")
            
            status_out = gr.Textbox(label="정보")
            
            dissolve_btn.click(create_dissolve, [before_input, after_input, logo_input], [video_out, status_out])
            sidebyside_btn.click(create_sidebyside, [before_input, after_input, logo_input], [image_out, status_out])
        
        with gr.Tab("배치 처리"):
            gr.Markdown("파일명 순서대로 before/after 쌍으로 처리\n\n예: `01_B.jpg, 01_A.jpg, 02_B.jpg, 02_A.jpg`")
            
            batch_files = gr.File(label="이미지들", file_count="multiple", file_types=["image"])
            batch_type = gr.Radio(["디졸브", "좌우비교"], value="디졸브", label="출력")
            batch_logo = gr.Image(label="로고", type="numpy")
            
            batch_btn = gr.Button("배치 시작", variant="primary")
            batch_out = gr.File(label="결과 ZIP")
            batch_status = gr.Textbox(label="상태")
            
            batch_btn.click(process_batch, [batch_files, batch_type, batch_logo], [batch_out, batch_status])

demo.launch()
