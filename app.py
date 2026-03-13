import gradio as gr
import cv2
import numpy as np
import tempfile
import os

MAX_SIZE = 4096  # 극단적 고해상도만 제한


# ─────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────

def resize_if_needed(img, max_size=MAX_SIZE):
    """극단적으로 큰 이미지만 리사이즈"""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    if w >= h:
        new_w, new_h = max_size, int(h * max_size / w)
    else:
        new_h, new_w = max_size, int(w * max_size / h)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def scale_before_to_after_resolution(before, after):
    """
    비포를 애프터와 동일한 높이로 스케일 (비율 유지).
    랜드마크 좌표계를 통일하기 위해 필요.
    """
    bh, bw = before.shape[:2]
    ah = after.shape[0]
    if bh == ah:
        return before
    new_w = int(round(bw * ah / bh))
    return cv2.resize(before, (new_w, ah), interpolation=cv2.INTER_LANCZOS4)


def crop_to_9_16(img, face_cx, face_cy):
    """
    9:16 비율(세로형)로 크롭 영역 계산. 얼굴 중심 기준.
    정렬된 두 이미지에 동일한 (x1, y1) 크롭을 적용하면 정렬이 유지됨.
    반환: (x1, y1, crop_w, crop_h)
    """
    h, w = img.shape[:2]
    ratio = 9.0 / 16.0  # width / height

    if w / h > ratio:
        # 현재 더 가로 → 너비 크롭
        new_h = h
        new_w = int(round(h * ratio))
    else:
        # 현재 더 세로 → 높이 크롭
        new_w = w
        new_h = int(round(w / ratio))

    # 짝수 보장
    new_w = (new_w // 2) * 2
    new_h = (new_h // 2) * 2

    # 얼굴 중심 기준 크롭 위치
    x1 = int(face_cx - new_w / 2)
    y1 = int(face_cy - new_h * 0.38)  # 얼굴을 상단 38% 지점에 배치

    # 이미지 경계 클램핑
    x1 = max(0, min(x1, w - new_w))
    y1 = max(0, min(y1, h - new_h))

    return x1, y1, new_w, new_h


# ─────────────────────────────────────────
# 얼굴 랜드마크 검출
# ─────────────────────────────────────────

def get_face_info(img):
    """
    MediaPipe FaceMesh로 랜드마크 검출.
    안모(정면): 눈 간격·기울기·중심 사용
    반모(측면): 코끝~턱 정중선 사용
    + 얼굴 중심점 (9:16 크롭용)
    """
    try:
        import mediapipe as mp
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            min_detection_confidence=0.1, min_tracking_confidence=0.1
        ) as fm:
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                raise RuntimeError("no face")

            lm = res.multi_face_landmarks[0].landmark

            # ── 눈 (안모) ──
            L_IDX = [33, 133, 160, 159, 158, 144, 145, 153]
            R_IDX = [362, 263, 387, 386, 385, 373, 374, 380]
            lx = np.mean([lm[i].x for i in L_IDX]) * w
            ly = np.mean([lm[i].y for i in L_IDX]) * h
            rx = np.mean([lm[i].x for i in R_IDX]) * w
            ry = np.mean([lm[i].y for i in R_IDX]) * h

            eye_dist  = np.hypot(rx - lx, ry - ly)
            eye_angle = np.arctan2(ry - ly, rx - lx)
            eyes_cx   = (lx + rx) / 2
            eyes_cy   = (ly + ry) / 2

            # ── 정중선 (반모) ──
            nose_x, nose_y = lm[4].x * w,   lm[4].y * h
            chin_x, chin_y = lm[152].x * w, lm[152].y * h
            ul_x,   ul_y   = lm[0].x * w,   lm[0].y * h
            ll_x,   ll_y   = lm[17].x * w,  lm[17].y * h

            midline_len = np.hypot(chin_x - nose_x, chin_y - nose_y)
            midline_ang = np.arctan2(chin_y - nose_y, chin_x - nose_x)
            mid_cx = (ul_x + ll_x) / 2
            mid_cy = (ul_y + ll_y) / 2

            # ── 얼굴 중심 (크롭용) ──
            face_cx = eyes_cx * 0.5 + nose_x * 0.3 + chin_x * 0.2
            face_cy = eyes_cy * 0.3 + nose_y * 0.3 + chin_y * 0.4

            # ── 안모 / 반모 판별 ──
            pts   = [10, 152, 234, 454]
            xs    = [lm[i].x * w for i in pts]
            ys    = [lm[i].y * h for i in pts]
            face_w = max(xs) - min(xs)
            face_h = max(ys) - min(ys)
            is_half = (face_h / (face_w + 1e-6)) < 0.85

            return dict(
                eye_dist=eye_dist, eye_angle=eye_angle,
                eyes_cx=eyes_cx, eyes_cy=eyes_cy,
                midline_len=midline_len, midline_ang=midline_ang,
                mid_cx=mid_cx, mid_cy=mid_cy,
                face_cx=face_cx, face_cy=face_cy,
                is_half=is_half, detected=True
            )

    except Exception as e:
        print(f"[FaceInfo] {e}")

    # 검출 실패 시 기본값 (이미지 중앙 기준)
    h, w = img.shape[:2]
    return dict(
        eye_dist=w * 0.30, eye_angle=0.0,
        eyes_cx=w / 2,     eyes_cy=h * 0.35,
        midline_len=h * 0.30, midline_ang=np.pi / 2,
        mid_cx=w / 2,      mid_cy=h * 0.60,
        face_cx=w / 2,     face_cy=h * 0.45,
        is_half=False, detected=False
    )


# ─────────────────────────────────────────
# 정렬 (비포 → 애프터 기준)
# ─────────────────────────────────────────

def align_before_to_after(before, after, bi, ai):
    """
    비포 이미지를 애프터 이미지의 얼굴 위치에 맞게 변환.
    변환: 스케일 + 회전 (비포 앵커 기준) + 이동 (애프터 앵커로)

    [안모] 눈 간격(스케일), 눈 기울기(회전), 눈 중심(이동)
    [반모] 정중선 길이(스케일), 정중선 각도(회전), 입술 중심(이동)
    """
    is_half = ai['is_half']

    if is_half:
        scale    = float(np.clip(ai['midline_len'] / (bi['midline_len'] + 1e-6), 0.6, 1.6))
        d_angle  = ai['midline_ang'] - bi['midline_ang']
        bx, by   = bi['mid_cx'],   bi['mid_cy']
        tx, ty   = ai['mid_cx'],   ai['mid_cy']
        label    = "반모(정중선)"
    else:
        scale    = float(np.clip(ai['eye_dist'] / (bi['eye_dist'] + 1e-6), 0.6, 1.6))
        d_angle  = ai['eye_angle'] - bi['eye_angle']
        bx, by   = bi['eyes_cx'],  bi['eyes_cy']
        tx, ty   = ai['eyes_cx'],  ai['eyes_cy']
        label    = "안모(눈)"

    angle_deg = float(np.clip(-np.degrees(d_angle), -25.0, 25.0))

    # 비포 앵커 기준 회전+스케일 → 애프터 앵커로 이동
    M = cv2.getRotationMatrix2D((bx, by), angle_deg, scale)

    new_bx = M[0, 0] * bx + M[0, 1] * by + M[0, 2]
    new_by = M[1, 0] * bx + M[1, 1] * by + M[1, 2]
    M[0, 2] += tx - new_bx
    M[1, 2] += ty - new_by

    h, w = after.shape[:2]
    aligned = cv2.warpAffine(
        before, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return aligned, label


# ─────────────────────────────────────────
# 밝기 맞춤
# ─────────────────────────────────────────

def match_brightness(img1, img2):
    """LAB 색공간에서 두 이미지의 평균 밝기를 동일하게 조정"""
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_l = (np.mean(lab1[:, :, 0]) + np.mean(lab2[:, :, 0])) / 2
    lab1[:, :, 0] = np.clip(lab1[:, :, 0] + (avg_l - np.mean(lab1[:, :, 0])), 0, 255)
    lab2[:, :, 0] = np.clip(lab2[:, :, 0] + (avg_l - np.mean(lab2[:, :, 0])), 0, 255)
    out1 = cv2.cvtColor(lab1.astype(np.uint8), cv2.COLOR_LAB2BGR)
    out2 = cv2.cvtColor(lab2.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out1, out2


# ─────────────────────────────────────────
# 로고 합성
# ─────────────────────────────────────────

def add_logo(frame, logo, margin_ratio=0.03, width_ratio=0.27):
    if logo is None:
        return frame
    h, w = frame.shape[:2]
    lw = int(w * width_ratio)
    lh = int(logo.shape[0] * lw / logo.shape[1])
    logo_r = cv2.resize(logo, (lw, lh), interpolation=cv2.INTER_LANCZOS4)
    m = int(w * margin_ratio)
    y1, y2 = m, min(m + lh, h)
    x1, x2 = m, min(m + lw, w)
    lh_c, lw_c = y2 - y1, x2 - x1

    if lh_c <= 0 or lw_c <= 0:
        return frame

    if logo_r.ndim == 3 and logo_r.shape[2] == 4:
        alpha = logo_r[:lh_c, :lw_c, 3:4] / 255.0
        rgb   = logo_r[:lh_c, :lw_c, :3]
        roi   = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = logo_r[:lh_c, :lw_c]
    return frame


# ─────────────────────────────────────────
# 메인 처리 함수
# ─────────────────────────────────────────

def process_images(before_img, after_img, logo_img=None):
    """
    1. 얼굴 검출 → 2. 정렬 → 3. 밝기 맞춤 → 4. 9:16 크롭 → 5. PNG 저장
    반환: (before_path, after_path, preview_numpy, status_str)
    """
    if before_img is None or after_img is None:
        return None, None, None, "전/후 사진을 모두 업로드해주세요"

    try:
        # ── BGR 변환 ──
        before = cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR)
        after  = cv2.cvtColor(after_img,  cv2.COLOR_RGB2BGR)

        orig_h, orig_w = after.shape[:2]

        # ── 극단적 고해상도만 리사이즈 ──
        before = resize_if_needed(before)
        after  = resize_if_needed(after)

        # ── 로고 처리 ──
        logo = None
        if logo_img is not None:
            ch = logo_img.shape[2] if logo_img.ndim == 3 else 1
            conv = cv2.COLOR_RGBA2BGRA if ch == 4 else cv2.COLOR_RGB2BGR
            logo = cv2.cvtColor(logo_img, conv)

        # ── 비포를 애프터와 같은 높이로 스케일 (좌표계 통일) ──
        before_s = scale_before_to_after_resolution(before, after)

        # ── 얼굴 랜드마크 검출 ──
        bi = get_face_info(before_s)
        ai = get_face_info(after)

        # ── 비포를 애프터 기준으로 정렬 ──
        before_aligned, align_type = align_before_to_after(before_s, after, bi, ai)

        # ── 밝기 통일 ──
        before_b, after_b = match_brightness(before_aligned, after.copy())

        # ── 9:16 크롭 (얼굴 중심 기준, 두 이미지에 동일 적용 → 정렬 유지) ──
        x1, y1, cw, ch = crop_to_9_16(after_b, ai['face_cx'], ai['face_cy'])
        before_out = before_b[y1:y1+ch, x1:x1+cw].copy()
        after_out  = after_b[y1:y1+ch, x1:x1+cw].copy()

        # ── 로고 합성 ──
        if logo is not None:
            before_out = add_logo(before_out, logo)
            after_out  = add_logo(after_out,  logo)

        # ── PNG 저장 ──
        tmp = tempfile.mkdtemp()
        bp  = os.path.join(tmp, "before_aligned.png")
        ap  = os.path.join(tmp, "after_aligned.png")
        cv2.imwrite(bp, before_out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(ap, after_out,  [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # ── 미리보기 (좌우 합성, 최대 높이 960px) ──
        ph   = min(960, before_out.shape[0])
        pw_b = int(before_out.shape[1] * ph / before_out.shape[0])
        pw_a = int(after_out.shape[1]  * ph / after_out.shape[0])
        b_p  = cv2.resize(before_out, (pw_b, ph))
        a_p  = cv2.resize(after_out,  (pw_a, ph))

        sep = 4  # 구분선 두께
        label_h = 44
        canvas = np.full((ph + label_h, pw_b + sep + pw_a, 3), 255, dtype=np.uint8)
        canvas[label_h:, :pw_b] = b_p
        canvas[label_h:, pw_b:pw_b+sep] = (200, 200, 200)  # 회색 구분선
        canvas[label_h:, pw_b+sep:] = a_p

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "BEFORE",
                    (pw_b // 2 - 55, 30), font, 1.0, (80, 80, 80), 2, cv2.LINE_AA)
        cv2.putText(canvas, "AFTER",
                    (pw_b + sep + pw_a // 2 - 44, 30), font, 1.0, (80, 80, 80), 2, cv2.LINE_AA)

        preview_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # ── 상태 메시지 ──
        out_h, out_w = after_out.shape[:2]
        db = "✓" if bi['detected'] else "✗"
        da = "✓" if ai['detected'] else "✗"
        status = (
            f"원본: {orig_w}×{orig_h} → 출력: {out_w}×{out_h} (9:16)\n"
            f"정렬: {align_type} | 얼굴 검출: 전({db}) 후({da})"
        )

        return bp, ap, preview_rgb, status

    except Exception as e:
        import traceback
        return None, None, None, f"오류: {str(e)}\n{traceback.format_exc()}"


# ─────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────

custom_css = """
.gradio-container { max-width: 1000px !important; margin: auto !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Dental B&A", css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align:center'>🦷 치과 전후 비교</h1>")
    gr.Markdown(
        "<p style='text-align:center;color:#666'>"
        "사진 업로드 → 얼굴 자동 정렬 → 9:16 PNG 저장"
        "</p>"
    )

    with gr.Row():
        before_input = gr.Image(label="BEFORE", type="numpy")
        after_input  = gr.Image(label="AFTER",  type="numpy")

    with gr.Accordion("로고 추가 (선택)", open=False):
        logo_input = gr.Image(label="PNG 투명 배경 지원", type="numpy", image_mode="RGBA")

    generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")

    preview_output = gr.Image(label="미리보기 (BEFORE | AFTER)", type="numpy")
    status_output  = gr.Textbox(label="정보", lines=2)

    with gr.Row():
        before_dl = gr.File(label="📥 BEFORE 다운로드 (PNG)")
        after_dl  = gr.File(label="📥 AFTER 다운로드 (PNG)")

    generate_btn.click(
        fn=process_images,
        inputs=[before_input, after_input, logo_input],
        outputs=[before_dl, after_dl, preview_output, status_output]
    )

demo.launch()
