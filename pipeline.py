######
# Este ficheiro contém as funções genéricas: carregar imagens, SIFT, matching, RANSAC, homografia, panorama, etc.
######

import cv2
import numpy as np
from pathlib import Path


# ============================================================
# 1. CARREGAR IMAGEM
# ============================================================

def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {path}")

    return image


# ============================================================
# 2. CONVERTER PARA GRAYSCALE
# ============================================================

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# 3. SIFT
# ============================================================

def detect_sift(image_gray):
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        image_gray,
        None
    )

    return keypoints, descriptors


# ============================================================
# 4. LOWE RATIO MATCHING
# ============================================================

def match_sift(des1, des2, ratio=0.8):

    # SIFT usa descritores floating-point
    # portanto usamos distância L2
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return matches, good_matches


# ============================================================
# 5. HOMOGRAFIA + RANSAC
# ============================================================

def estimate_homography(
    kp1,
    kp2,
    good_matches,
    ransac_threshold=3.0
):

    if len(good_matches) < 4:
        raise ValueError(
            "São necessários pelo menos 4 matches para calcular a homografia."
        )

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransac_threshold
    )

    if H is None:
        raise ValueError("Não foi possível calcular a homografia.")

    return H, mask


# ============================================================
# 6. DESENHAR KEYPOINTS
# ============================================================

def draw_keypoints(image, keypoints):

    result = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return result


# ============================================================
# 7. DESENHAR MATCHES
# ============================================================

def draw_matches(
    img1,
    kp1,
    img2,
    kp2,
    matches,
    mask=None
):

    matches_mask = None

    if mask is not None:
        matches_mask = mask.ravel().tolist()

    result = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return result


# ============================================================
# 8. WARP SIMPLES PARA A IMAGEM DE REFERÊNCIA
# ============================================================

def warp_to_reference(img1, img2, H):

    height, width = img2.shape[:2]

    warped = cv2.warpPerspective(
        img1,
        H,
        (width, height)
    )

    return warped


# ============================================================
# 9. CRIAR PANORAMA
# ============================================================

def create_panorama(img1, img2, H):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Cantos da imagem 1
    corners1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    # Transformar os cantos
    warped_corners1 = cv2.perspectiveTransform(
        corners1,
        H
    )

    # Cantos da imagem 2
    corners2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 1, 2)

    # Determinar tamanho do canvas
    all_corners = np.concatenate(
        (warped_corners1, corners2),
        axis=0
    )

    xmin, ymin = np.floor(
        all_corners.min(axis=0).ravel()
    ).astype(int)

    xmax, ymax = np.ceil(
        all_corners.max(axis=0).ravel()
    ).astype(int)

    # Translação necessária
    tx = -xmin
    ty = -ymin

    translation = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    panorama_width = xmax - xmin
    panorama_height = ymax - ymin

    # Warp da imagem 1
    warped_img1 = cv2.warpPerspective(
        img1,
        translation @ H,
        (panorama_width, panorama_height)
    )

    # Máscara imagem 1
    mask1 = cv2.warpPerspective(
        np.ones((h1, w1), dtype=np.uint8),
        translation @ H,
        (panorama_width, panorama_height),
        flags=cv2.INTER_NEAREST
    ).astype(bool)

    # Canvas imagem 2
    canvas_img2 = np.zeros_like(warped_img1)

    canvas_img2[
        ty:ty + h2,
        tx:tx + w2
    ] = img2

    # Máscara imagem 2
    mask2 = np.zeros(
        (panorama_height, panorama_width),
        dtype=bool
    )

    mask2[
        ty:ty + h2,
        tx:tx + w2
    ] = True

    # Começar com a imagem 1
    panorama = warped_img1.copy()

    # Zona onde só existe imagem 2
    only_img2 = mask2 & ~mask1

    panorama[only_img2] = canvas_img2[only_img2]

    # Zona onde existem as duas
    overlap = mask1 & mask2

    # Baseline blending:
    # média dos pixels
    panorama[overlap] = (
        (
            warped_img1[overlap].astype(np.float32)
            +
            canvas_img2[overlap].astype(np.float32)
        )
        / 2
    ).astype(np.uint8)

    return panorama


# ============================================================
# 10. GUARDAR IMAGEM
# ============================================================

def save_image(image, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(path),
        image
    )


# ============================================================
# 11. MOSTRAR IMAGEM
# ============================================================

def show_image(title, image):

    cv2.imshow(
        title,
        image
    )