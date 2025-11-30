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

        # 길이 계산 (원하면 그대로 사용)
        def dist(a, b):
            return float(np.linalg.norm(a - b))

        OW = dist(outer_left,  outer_right)
        OH = dist(outer_top,   outer_bottom)
        IW = dist(inner_left,  inner_right)
        IH = dist(inner_top,   inner_bottom)

        print(f"OW={OW:.2f}, OH={OH:.2f}, IW={IW:.2f}, IH={IH:.2f}")

        # ───────── 점만 찍기 ─────────
        vis = img.copy()
        key_points_outer = [outer_left, outer_right, outer_top, outer_bottom]
        key_points_inner = [inner_left, inner_right, inner_top, inner_bottom]

        # 바깥 입술 점 (파란색)
        for (x, y) in key_points_outer:
            cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)

        # 안쪽 입술 점 (초록색)
        for (x, y) in key_points_inner:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

        cv2.namedWindow("mouth_points", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mouth_points", 800, 600)
        cv2.imshow("mouth_points", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
