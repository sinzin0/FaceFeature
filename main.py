import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────
# 0. 입술 인덱스 정의 (MediaPipe FaceMesh 기준)
# ─────────────────────
OUTER_LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]

# 안쪽 입술(inner)
INNER_LIPS = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

MOUTH_IDX = OUTER_LIPS + INNER_LIPS

mp_face_mesh = mp.solutions.face_mesh

# 1) 이미지 읽기
img_path = r"D:/python/RoIDetection/TestData/Frames_45/4/2/20250928_131709_000012.jpg"
img = cv2.imread(img_path)
if img is None:
    raise RuntimeError("이미지를 못 읽었습니다. 경로를 다시 확인해 주세요.")

h, w, _ = img.shape
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2) FaceMesh 모델 생성 (정적 이미지라서 static_image_mode=True)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,   # 눈/입술 주위 더 촘촘한 랜드마크
    min_detection_confidence=0.5
) as face_mesh:

    result = face_mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        print("얼굴 못 찾음")
    else:
        face_landmarks = result.multi_face_landmarks[0]

        # 3) 전체 랜드마크 (468개) → 픽셀 좌표로 변환
        points = []
        for lm in face_landmarks.landmark:
            x = lm.x * w   # 정규화 좌표(0~1)를 픽셀 좌표로 변환
            y = lm.y * h
            points.append((x, y))

        print("랜드마크 개수:", len(points))  # 보통 468개
        print("첫 5개:", points[:5])

        # 4) 입술 인덱스만 골라서 mouth_pts 만들기
        mouth_pts = [points[i] for i in MOUTH_IDX]

        # 5) 원본 이미지 위에 입술 랜드마크 시각화
        vis = img.copy()
        for (x, y) in mouth_pts:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)  # 초록색 점

        cv2.namedWindow("mouth_landmarks", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mouth_landmarks", 800, 600)
        cv2.imshow("mouth_landmarks", vis)
        cv2.waitKey(0)   # 한 장만 볼 거면 0
        cv2.destroyAllWindows()
