import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# 키포인트 인덱스 정의 (FaceMesh 기준)
OUTER_LEFT, OUTER_RIGHT, OUTER_TOP, OUTER_BOTTOM = 61, 291, 0, 17
INNER_LEFT, INNER_RIGHT, INNER_TOP, INNER_BOTTOM = 78, 308, 13, 14

KEY_IDXS = [
    OUTER_LEFT, OUTER_RIGHT, OUTER_TOP, OUTER_BOTTOM,
    INNER_LEFT, INNER_RIGHT, INNER_TOP, INNER_BOTTOM
]

# ➜ 어느 쪽 입술 기준으로 볼지 선택: "right" 또는 "left"
USE_SIDE = "left"   # 필요하면 "left" 로 바꿔서 테스트

img_path = "D:/python/RoIDetection/TestData/Frames_45/1/1/20250926_130544_000001.jpg"
img = cv2.imread(img_path)
if img is None:
    raise RuntimeError("이미지를 못 읽었습니다.")

h, w, _ = img.shape
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.01
) as face_mesh:
    result = face_mesh.process(img_rgb)
    if not result.multi_face_landmarks:
        print("얼굴 못 찾음")
    else:
        face_landmarks = result.multi_face_landmarks[0]

        # 전체 랜드마크 → 픽셀 좌표
        points = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

        # 8개 키포인트만 뽑기
        pts = np.array([points[i] for i in KEY_IDXS], dtype=np.float32)

        outer_left, outer_right, outer_top, outer_bottom = pts[0], pts[1], pts[2], pts[3]
        inner_left, inner_right, inner_top, inner_bottom = pts[4], pts[5], pts[6], pts[7]

        # 길이 계산 함수
        def dist(a, b):
            return float(np.linalg.norm(a - b))

        # ───────── 기존 전체 폭 (양쪽 다 쓰는 버전) ─────────
        OW_full = dist(outer_left,  outer_right)
        OH      = dist(outer_top,   outer_bottom)
        IW_full = dist(inner_left,  inner_right)
        IH      = dist(inner_top,   inner_bottom)

        # ───────── 입술 중심 계산 ─────────
        center_outer_x = 0.5 * (outer_top[0] + outer_bottom[0])  # 상/하 x평균
        center_outer_y = 0.5 * (outer_left[1] + outer_right[1])  # 좌/우 y평균
        center_outer = np.array([center_outer_x, center_outer_y], dtype=np.float32)

        # 안쪽 입술 중심
        center_inner_x = 0.5 * (inner_top[0] + inner_bottom[0])
        center_inner_y = 0.5 * (inner_left[1] + inner_right[1])
        center_inner = np.array([center_inner_x, center_inner_y], dtype=np.float32)

        # ───────── 반쪽 폭 (중심 → 한쪽 끝) ─────────
        if USE_SIDE == "right":
            OW_half = dist(center_outer, outer_right)
            IW_half = dist(center_inner, inner_right)
            side_label = "RIGHT"
            side_outer_pt = outer_right
            side_inner_pt = inner_right
        else:  # "left"
            OW_half = dist(center_outer, outer_left)
            IW_half = dist(center_inner, inner_left)
            side_label = "LEFT"
            side_outer_pt = outer_left
            side_inner_pt = inner_left

        print(f"[전체 폭 기준]")
        print(f"  OW_full={OW_full:.2f}, OH={OH:.2f}, IW_full={IW_full:.2f}, IH={IH:.2f}")
        print(f"[반쪽({side_label}) 기준]")
        print(f"  OW_half(center→{side_label})={OW_half:.2f}, IW_half(center→{side_label})={IW_half:.2f}")

        # ───────── 시각화 ─────────
        vis = img.copy()

        # 바깥 입술 점 (파란색)
        key_points_outer = [outer_left, outer_right, outer_top, outer_bottom]
        for (x, y) in key_points_outer:
            cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)

        # 안쪽 입술 점 (초록색)
        key_points_inner = [inner_left, inner_right, inner_top, inner_bottom]
        for (x, y) in key_points_inner:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

        # 중심 점 표시 (노란색)
        cx_o, cy_o = int(center_outer[0]), int(center_outer[1])
        cx_i, cy_i = int(center_inner[0]), int(center_inner[1])
        cv2.circle(vis, (cx_o, cy_o), 4, (0, 255, 255), -1)
        cv2.circle(vis, (cx_i, cy_i), 4, (0, 255, 255), -1)

        # 중심 → 선택된 한쪽 끝까지 선 그리기 (바깥/안쪽 각각)
        sx_o, sy_o = int(side_outer_pt[0]), int(side_outer_pt[1])
        sx_i, sy_i = int(side_inner_pt[0]), int(side_inner_pt[1])

        # 바깥 입술: 하늘색 선
        cv2.line(vis, (cx_o, cy_o), (sx_o, sy_o), (255, 255, 0), 2)
        # 안쪽 입술: 분홍색 선
        cv2.line(vis, (cx_i, cy_i), (sx_i, sy_i), (255, 0, 255), 2)

        # 텍스트로도 어느 쪽 반인지 표시
        cv2.putText(
            vis,
            f"Side: {side_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.namedWindow("mouth_points_half_side", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mouth_points_half_side", 800, 600)
        cv2.imshow("mouth_points_half_side", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
