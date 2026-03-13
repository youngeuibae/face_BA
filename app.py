import gradio as gr
import cv2
import numpy as np
import tempfile
import os

TARGET_W = 1080   # 9:16 기준 너비
TARGET_H = 1920   # 9:16 기준 높이
MAX_SIZE = 1920

# ── 유틸 ──────────────────────────────────────────────────────────────────────

def resize_if_needed(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
    return img

def pad_to_9_16(img, target_w=TARGET_W, target_h=TARGET_H):
    """원본 비율 유지 + 9:16 캔버스에 맞게 패딩(흰색)"""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

# ── 얼굴 랜드마크 ─────────────────────────────────────────────────────────────

def get_face_info(img):
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                    min_detection_confidence=0.1,
                                    min_tracking_confidence=0.1) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                h, w = img.shape[:2]

                # 눈
                left_eye_idx  = [33, 133, 160, 159, 158, 144, 145, 153]
                right_eye_idx = [362, 263, 387, 386, 385, 373, 374, 380]
                le_x = np.mean([lm[i].x for i in left_eye_idx]) * w
                le_y = np.mean([lm[i].y for i in left_eye_idx]) * h
                re_x = np.mean([lm[i].x for i in right_eye_idx]) * w
                re_y = np.mean([lm[i].y for i in right_eye_idx]) * h
                eye_cx = (le_x + re_x) / 2
                eye_cy = (le_y + re_y) / 2
                eye_dist  = np.hypot(re_x - le_x, re_y - le_y)
                eye_angle = np.arctan2(re_y - le_y, re_x - le_x)

                # 입
                mouth_idx = [61, 291, 0, 17]
                mouth_cx = np.mean([lm[i].x for i in mouth_idx]) * w
                mouth_cy = np.mean([lm[i].y for i in mouth_idx]) * h

                # 코 끝
                nose_x = lm[4].x * w
                nose_y = lm[4].y * h

                # 안모/반모 판별
                xs = [lm[i].x * w for i in [10, 152, 234, 454]]
                ys = [lm[i].y * h for i in [10, 152, 234, 454]]
                face_h = max(ys) - min(ys)
                face_w = max(xs) - min(xs)
                is_half = (face_h / face_w if face_w > 0 else 1.0) < 0.9

                return dict(
                    eye_dist=eye_dist, eye_angle=eye_angle,
                    eye_cx=eye_cx, eye_cy=eye_cy,
                    mouth_cx=mouth_cx, mouth_cy=mouth_cy,
                    nose_x=nose_x, nose_y=nose_y,
                    is_half=is_half, detected=True
                )
    except Exception as e:
        print(f"MediaPipe error: {e}")

    h, w = img.shape[:2]
    return dict(
        eye_dist=w*0.3, eye_angle=0,
        eye_cx=w/2, eye_cy=h*0.35,
        mouth_cx=w/2, mouth_cy=h*0.6,
        nose_x=w/2, nose_y=h*0.5,
        is_half=True, detected=False
    )

# ── 정렬 ──────────────────────────────────────────────────────────────────────

def align_before_to_after(before_bgr, after_bgr, bi, ai):
    """
    before 이미지를 after 기준으로 정렬.
    - 눈 간격(scale), 눈 기울기(rotate), 눈 중심(translate) 일치
    - 입 위치도 검증용으로 사용
    """
    # 스케일: after 눈 간격 / before 눈 간격
    scale = np.clip(ai['eye_dist'] / bi['eye_dist'], 0.7, 1.4)

    # 회전: 눈 기울기 차이
    angle_deg = np.clip(-np.degrees(ai['eye_angle'] - bi['eye_angle']), -15, 15)

    # 앵커: before 눈 중심 → after 눈 중심으로
    ax, ay = bi['eye_cx'], bi['eye_cy']
    tx, ty = ai['eye_cx'], ai['eye_cy']

    M = cv2.getRotationMatrix2D((ax, ay), angle_deg, scale)
    # 변환 후 앵커 좌표
    nax = M[0,0]*ax + M[0,1]*ay + M[0,2]
    nay = M[1,0]*ax + M[1,1]*ay + M[1,2]
    # 타겟으로 평행이동
    M[0,2] += tx - nax
    M[1,2] += ty - nay

    h, w = after_bgr.shape[:2]
    aligned = cv2.warpAffine(before_bgr, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return aligned

# ── 밝기 보정 ──────────────────────────────────────────────────────────────────

def match_brightness(img1, img2):
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg = (np.mean(lab1[:,:,0]) + np.mean(lab2[:,:,0])) / 2
    lab1[:,:,0] = np.clip(lab1[:,:,0] + (avg - np.mean(lab1[:,:,0])), 0, 255)
    lab2[:,:,0] = np.clip(lab2[:,:,0] + (avg - np.mean(lab2[:,:,0])), 0, 255)
    r1 = cv2.cvtColor(lab1.astype(np.uint8), cv2.COLOR_LAB2BGR)
    r2 = cv2.cvtColor(lab2.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return r1, r2

# ── 로고 ───────────────────────────────────────────────────────────────────────

def add_logo(frame, logo, margin_ratio=0.03, width_ratio=0.27):
    if logo is None:
        return frame
    h, w = frame.shape[:2]
    lw = int(w * width_ratio)
    lh = int(logo.shape[0] * (lw / logo.shape[1]))
    logo_r = cv2.resize(logo, (lw, lh), interpolation=cv2.INTER_LANCZOS4)
    m = int(w * margin_ratio)
    if logo_r.shape[2] == 4:
        alpha = logo_r[:,:,3:4] / 255.0
        roi = frame[m:m+lh, m:m+lw]
        frame[m:m+lh, m:m+lw] = (logo_r[:,:,:3] * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        frame[m:m+lh, m:m+lw] = logo_r
    return frame

# ── 메인 ───────────────────────────────────────────────────────────────────────

def create_video(before_img, after_img, logo_img=None):
    if before_img is None or after_img is None:
        return None, "전/후 사진을 모두 업로드해주세요"

    try:
        before = cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR)
        after  = cv2.cvtColor(after_img,  cv2.COLOR_RGB2BGR)

        # 1) 최대 크기 제한
        before = resize_if_needed(before)
        after  = resize_if_needed(after)

        # 2) 각각 9:16 캔버스로 패딩 (원본 해상도 최대 활용)
        #    after 기준 해상도 결정
        ah, aw = after.shape[:2]
        # 9:16 캔버스 크기: after 원본을 담을 수 있는 최소 9:16
        if aw / ah < 9/16:
            cw = int(ah * 9 / 16)
            ch = ah
        else:
            cw = aw
            ch = int(aw * 16 / 9)
        # 짝수 맞춤
        cw = (cw // 2) * 2
        ch = (ch // 2) * 2

        after_canvas  = pad_to_9_16(after,  cw, ch)
        before_canvas = pad_to_9_16(before, cw, ch)

        # 3) 얼굴 랜드마크 (패딩 후 이미지 기준)
        bi = get_face_info(before_canvas)
        ai = get_face_info(after_canvas)

        # 4) before → after 기준으로 정렬
        if bi['detected'] and ai['detected']:
            before_aligned = align_before_to_after(before_canvas, after_canvas, bi, ai)
            align_note = "눈 기준 정렬 ✓"
        else:
            before_aligned = before_canvas
            align_note = f"얼굴 미검출(before:{bi['detected']} after:{ai['detected']}) – 정렬 생략"

        # 5) 밝기 보정
        bf, af = match_brightness(before_aligned, after_canvas)

        # 6) 영상 생성
        fps = 30
        before_frames, dissolve_frames, after_frames = 39, 12, 39
        total_frames = before_frames + dissolve_frames + after_frames

        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "dental_comparison.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (cw, ch))

        logo = None
        if logo_img is not None:
            logo = (cv2.cvtColor(logo_img, cv2.COLOR_RGBA2BGRA)
                    if logo_img.shape[2] == 4
                    else cv2.cvtColor(logo_img, cv2.COLOR_RGB2BGR))

        for i in range(total_frames):
            if i < before_frames:
                frame = bf.copy()
            elif i < before_frames + dissolve_frames:
                a = (i - before_frames) / dissolve_frames
                frame = cv2.addWeighted(bf, 1-a, af, a, 0)
            else:
                frame = af.copy()
            if logo is not None:
                frame = add_logo(frame, logo)
            out.write(frame)

        out.release()
        return output_path, f"해상도: {cw}×{ch} | 비율: 9:16 | {align_note}"

    except Exception as e:
        import traceback
        return None, f"오류: {str(e)}\n{traceback.format_exc()}"

# ── UI ────────────────────────────────────────────────────────────────────────

custom_css = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Dental B&A", css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align:center'>🦷 치과 전후 비교</h1>")
    gr.Markdown("<p style='text-align:center;color:#666'>9:16 비율 · 원본 해상도 유지 · 얼굴 자동 정렬</p>")

    with gr.Row():
        before_input = gr.Image(label="BEFORE", type="numpy")
        after_input  = gr.Image(label="AFTER",  type="numpy")

    with gr.Accordion("로고 추가 (선택)", open=False):
        logo_input = gr.Image(label="PNG 투명 배경 지원", type="numpy")

    generate_btn = gr.Button("영상 생성", variant="primary")

    with gr.Row():
        video_output  = gr.Video(label="결과")
        status_output = gr.Textbox(label="정보", lines=4)

    generate_btn.click(
        fn=create_video,
        inputs=[before_input, after_input, logo_input],
        outputs=[video_output, status_output]
    )

demo.launch(css=custom_css)
