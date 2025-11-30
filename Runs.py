import os
import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict

# ========= 경로 설정 =========
MATCH_ROOT = r"C:/Users/jyshin/Desktop/신진영/대학/대학자료/졸업논문/신진영_졸업논문_구현/데이터/match image/45"
FRAMES_ROOT = r"D:/python/RoIDetection/TestData/Frames_45"   # 전체 얼굴 프레임 폴더 (긴 파일명)
OUTPUT_DIR = r"C:/Users/jyshin/Desktop/신진영/대학/대학자료/졸업논문/신진영_졸업논문_구현/데이터/FaceFeature"      # 결과 txt 저장 폴더

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SPEAKERS = 5
NUM_WORDS = 10

# ========= MediaPipe FaceMesh 설정 =========
mp_face_mesh = mp.solutions.face_mesh

# 입술 관련 랜드마크 인덱스 (MediaPipe FaceMesh 기준)
OUTER_LEFT, OUTER_RIGHT, OUTER_TOP, OUTER_BOTTOM = 61, 291, 0, 17
INNER_LEFT, INNER_RIGHT, INNER_TOP, INNER_BOTTOM = 78, 308, 13, 14

KEY_IDXS = [
    OUTER_LEFT, OUTER_RIGHT, OUTER_TOP, OUTER_BOTTOM,
    INNER_LEFT, INNER_RIGHT, INNER_TOP, INNER_BOTTOM
]

USE_SIDE = "left"

def compute_mouth_geom(face_mesh, img_bgr):
    """
    한 장의 BGR 이미지에서 (OW, OH, IW, IH) 계산.
    실패하면 None 리턴.
    """
    if img_bgr is None:
        return None

    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        return None

    face_landmarks = result.multi_face_landmarks[0]
    points = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
    pts = np.array([points[i] for i in KEY_IDXS], dtype=np.float32)

    outer_left, outer_right, outer_top, outer_bottom = pts[0], pts[1], pts[2], pts[3]
    inner_left, inner_right, inner_top, inner_bottom = pts[4], pts[5], pts[6], pts[7]

    # 중신 찾기
    center_outer_x = 0.5 * (outer_top[0] + outer_bottom[0])  # 상/하 x평균
    center_outer_y = 0.5 * (outer_left[1] + outer_right[1])  # 좌/우 y평균
    center_outer = np.array([center_outer_x, center_outer_y], dtype=np.float32)

    # 안쪽 입술 중심
    center_inner_x = 0.5 * (inner_top[0] + inner_bottom[0])
    center_inner_y = 0.5 * (inner_left[1] + inner_right[1])
    center_inner = np.array([center_inner_x, center_inner_y], dtype=np.float32)

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    if USE_SIDE == "right":
        OW = dist(center_outer, outer_right)  # 중심 -> 오른쪽 바깥 입술
        IW = dist(center_inner, inner_right)  # 중심 -> 오른쪽 안쪽 입술
    else:  # "left"
        OW = dist(center_outer, outer_left)  # 중심 -> 왼쪽 바깥 입술
        IW = dist(center_inner, inner_left)  # 중심 -> 왼쪽 안쪽 입술

    # OW = dist(outer_left, outer_right)
    OH = dist(outer_top,  outer_bottom)
    # IW = dist(inner_left, inner_right)
    IH = dist(inner_top,  inner_bottom)

    return OW, OH, IW, IH


# ========= 메인 루프 =========
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3   # 조금 낮춰서 더 잘 잡히게
) as face_mesh:

    # (spk, word)마다 프레임 파일 리스트 캐시
    frames_cache = {}

    for spk in range(1, NUM_SPEAKERS + 1):
        match_path = os.path.join(MATCH_ROOT, f"match_{spk}.txt")
        out_path   = os.path.join(OUTPUT_DIR, f"shape_feature_45_{spk}.txt")

        print(f"[Speaker {spk}] match: {match_path}")
        if not os.path.exists(match_path):
            print(f"  [경고] match 파일 없음: {match_path}")
            continue

        # (word, utter) → [idx1, idx2, ...] (speech 프레임만, 6열을 index로 해석)
        utter_frame_indices = defaultdict(list)

        # ===== 1단계: match_*에서 speech 프레임 index 리스트 만들기 =====
        with open(match_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 6:
                    continue

                try:
                    word = int(parts[0])
                    utter = int(parts[1])
                    speech_flag = int(parts[4])
                    img_name = parts[5]   # 예: "1.jpg"
                except ValueError:
                    continue

                if word < 1 or word > NUM_WORDS:
                    continue

                if speech_flag != 1:
                    continue

                # "1.jpg" -> 1, "23.jpg" -> 23
                try:
                    idx = int(os.path.splitext(img_name)[0])
                except ValueError:
                    # 파일명이 숫자가 아니면 패스
                    print(f"  [경고] 숫자가 아닌 이미지 인덱스: {img_name} (skip)")
                    continue

                utter_frame_indices[(word, utter)].append(idx)

        # ===== 2단계: utter 단위로 Mediapipe shape 계산 & txt 저장 =====
        with open(out_path, "w", encoding="utf-8") as f_out:
            keys = sorted(utter_frame_indices.keys(), key=lambda x: (x[0], x[1]))

            for (word, utter) in keys:
                idx_list = utter_frame_indices[(word, utter)]
                if not idx_list:
                    continue

                # 이 (spk, word)에 해당하는 전체 프레임 파일 리스트 불러오기 (캐싱)
                cache_key = (spk, word)
                if cache_key not in frames_cache:
                    frames_dir = os.path.join(FRAMES_ROOT, str(spk), str(word))
                    if not os.path.isdir(frames_dir):
                        print(f"  [경고] 프레임 폴더 없음: {frames_dir}")
                        frames_cache[cache_key] = []
                    else:
                        frames = sorted(
                            [fn for fn in os.listdir(frames_dir)
                             if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
                        )
                        frames_cache[cache_key] = frames
                frames_all = frames_cache[cache_key]

                if not frames_all:
                    print(f"  [경고] 프레임 리스트 비어있음: spk={spk}, word={word}")
                    continue

                geoms = []
                last_geom = None  # ★ 최근 성공한 값 저장

                # --- 각 speech 프레임 index에 대해 (OW,OH,IW,IH) 계산 ---
                for idx in idx_list:
                    # match의 index는 1부터 시작한다고 가정 → 리스트 인덱스는 idx-1
                    if idx < 1 or idx > len(frames_all):
                        print(f"  [경고] idx {idx} out of range (1~{len(frames_all)}), skip")
                        geoms.append(last_geom)
                        continue

                    frame_name = frames_all[idx - 1]
                    img_path = os.path.join(FRAMES_ROOT, str(spk), str(word), frame_name)

                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"  [경고] 이미지 로드 실패: {img_path}")
                        geoms.append(last_geom)
                        continue

                    geom = compute_mouth_geom(face_mesh, img)

                    if geom is None:
                        # ★ 얼굴 못 찾으면 가장 최근 성공값으로 대체
                        if last_geom is not None:
                            print(f"  [정보] 얼굴 못 찾음 → 이전 값으로 대체: {img_path}")
                            geom = last_geom
                        else:
                            print(f"  [정보] 얼굴 못 찾음(이전값 없음): {img_path}")
                            # 일단 None으로 두고, 나중에 fill 단계에서 처리
                    else:
                        last_geom = geom

                    geoms.append(geom)

                # 이 발화에서 전부 실패하면 그냥 스킵
                if all(g is None for g in geoms):
                    print(f"  [utter skip] 모든 프레임에서 얼굴 못 찾음: spk={spk}, word={word}, utter={utter}")
                    continue

                # --- 추가 안전장치: 아직 남아있는 None은 앞/뒤 값으로 보간 ---
                # forward fill
                for i in range(1, len(geoms)):
                    if geoms[i] is None and geoms[i-1] is not None:
                        geoms[i] = geoms[i-1]
                # backward fill
                for i in range(len(geoms)-2, -1, -1):
                    if geoms[i] is None and geoms[i+1] is not None:
                        geoms[i] = geoms[i+1]

                # 최종적으로 남은 None은 건너뜀
                frame_idx = 0
                for geom in geoms:
                    if geom is None:
                        continue
                    frame_idx += 1
                    OW, OH, IW, IH = geom
                    # word  utter  frame  OW  OH  IW  IH
                    line = f"{word} {utter} {frame_idx} {OW:.6f} {OH:.6f} {IW:.6f} {IH:.6f}\n"
                    f_out.write(line)

        print(f"  → saved: {out_path}")
