######
# Este ficheiro contém as funções genéricas:
# carregar imagens, detetores, descritores, matching,
# RANSAC, homografia, ground truth, warp e panorama.
######

import cv2
import numpy as np
from pathlib import Path


# ============================================================
# 1. CARREGAR IMAGEM
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Não foi possível abrir a imagem: {path}"
        )

    return image


# ============================================================
# 2. CONVERTER PARA GRAYSCALE
# ============================================================

def to_gray(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


# ============================================================
# 3. DETETORES + DESCRITORES
# ============================================================


# ------------------------------------------------------------
# SIFT
# ------------------------------------------------------------

def detect_sift(image_gray):

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        image_gray,
        None
    )

    return keypoints, descriptors


# ------------------------------------------------------------
# ORB
# ------------------------------------------------------------

def detect_orb(image_gray):

    orb = cv2.ORB_create(
        nfeatures=1500
    )

    keypoints, descriptors = orb.detectAndCompute(
        image_gray,
        None
    )

    return keypoints, descriptors


# ------------------------------------------------------------
# KAZE
# ------------------------------------------------------------

def detect_kaze(image_gray):

    kaze = cv2.KAZE_create()

    keypoints, descriptors = kaze.detectAndCompute(
        image_gray,
        None
    )

    return keypoints, descriptors


# ============================================================
# 4. MATCHING
# ============================================================


# ------------------------------------------------------------
# SIFT MATCHING
#
# SIFT:
# descriptor floating-point
# -> distância L2
# ------------------------------------------------------------

def match_sift(
    des1,
    des2,
    ratio=0.8
):

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for m, n in matches:

        if m.distance < ratio * n.distance:

            good_matches.append(
                m
            )

    return matches, good_matches


# ------------------------------------------------------------
# ORB MATCHING
#
# ORB:
# descriptor binário
# -> distância Hamming
# ------------------------------------------------------------

def match_orb(
    des1,
    des2,
    ratio=0.8
):

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for m, n in matches:

        if m.distance < ratio * n.distance:

            good_matches.append(
                m
            )

    return matches, good_matches


# ------------------------------------------------------------
# KAZE MATCHING
#
# KAZE:
# descriptor floating-point
# -> distância L2
# ------------------------------------------------------------

def match_kaze(
    des1,
    des2,
    ratio=0.8
):

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for m, n in matches:

        if m.distance < ratio * n.distance:

            good_matches.append(
                m
            )

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
            "São necessários pelo menos 4 matches "
            "para calcular a homografia."
        )

    # Pontos da imagem 1
    src_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good_matches
        ]
    ).reshape(
        -1,
        1,
        2
    )

    # Pontos correspondentes da imagem 2
    dst_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good_matches
        ]
    ).reshape(
        -1,
        1,
        2
    )

    # Calcular homografia usando RANSAC
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransac_threshold
    )

    if H is None:

        raise ValueError(
            "Não foi possível calcular a homografia."
        )

    return H, mask


# ============================================================
# 6. DESENHAR KEYPOINTS
# ============================================================

def draw_keypoints(
    image,
    keypoints
):

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

        matches_mask = (
            mask
            .ravel()
            .tolist()
        )

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
# 8. WARP PARA A IMAGEM DE REFERÊNCIA
# ============================================================

def warp_to_reference(
    img1,
    img2,
    H
):

    height, width = (
        img2.shape[:2]
    )

    warped = cv2.warpPerspective(
        img1,
        H,
        (
            width,
            height
        )
    )

    return warped


# ============================================================
# 9. CRIAR PANORAMA
# ============================================================

def create_panorama(
    img1,
    img2,
    H
):

    h1, w1 = (
        img1.shape[:2]
    )

    h2, w2 = (
        img2.shape[:2]
    )


    # --------------------------------------------------------
    # Cantos da imagem 1
    # --------------------------------------------------------

    corners1 = np.float32(
        [
            [0, 0],
            [w1, 0],
            [w1, h1],
            [0, h1]
        ]
    ).reshape(
        -1,
        1,
        2
    )


    # Transformar os cantos da imagem 1
    warped_corners1 = cv2.perspectiveTransform(
        corners1,
        H
    )


    # --------------------------------------------------------
    # Cantos da imagem 2
    # --------------------------------------------------------

    corners2 = np.float32(
        [
            [0, 0],
            [w2, 0],
            [w2, h2],
            [0, h2]
        ]
    ).reshape(
        -1,
        1,
        2
    )


    # --------------------------------------------------------
    # Determinar tamanho do canvas
    # --------------------------------------------------------

    all_corners = np.concatenate(
        (
            warped_corners1,
            corners2
        ),
        axis=0
    )


    xmin, ymin = np.floor(
        all_corners
        .min(axis=0)
        .ravel()
    ).astype(
        int
    )


    xmax, ymax = np.ceil(
        all_corners
        .max(axis=0)
        .ravel()
    ).astype(
        int
    )


    # --------------------------------------------------------
    # Translação
    # --------------------------------------------------------

    tx = -xmin
    ty = -ymin


    translation = np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ],
        dtype=np.float64
    )


    panorama_width = (
        xmax
        -
        xmin
    )

    panorama_height = (
        ymax
        -
        ymin
    )


    # --------------------------------------------------------
    # Warp da imagem 1
    # --------------------------------------------------------

    warped_img1 = cv2.warpPerspective(
        img1,
        translation @ H,
        (
            panorama_width,
            panorama_height
        )
    )


    # --------------------------------------------------------
    # Máscara da imagem 1
    # --------------------------------------------------------

    mask1 = cv2.warpPerspective(
        np.ones(
            (
                h1,
                w1
            ),
            dtype=np.uint8
        ),
        translation @ H,
        (
            panorama_width,
            panorama_height
        ),
        flags=cv2.INTER_NEAREST
    ).astype(
        bool
    )


    # --------------------------------------------------------
    # Canvas para imagem 2
    # --------------------------------------------------------

    canvas_img2 = np.zeros_like(
        warped_img1
    )


    canvas_img2[
        ty:ty + h2,
        tx:tx + w2
    ] = img2


    # --------------------------------------------------------
    # Máscara imagem 2
    # --------------------------------------------------------

    mask2 = np.zeros(
        (
            panorama_height,
            panorama_width
        ),
        dtype=bool
    )


    mask2[
        ty:ty + h2,
        tx:tx + w2
    ] = True


    # --------------------------------------------------------
    # Criar panorama
    # --------------------------------------------------------

    panorama = warped_img1.copy()


    # Zona onde só existe imagem 2
    only_img2 = (
        mask2
        &
        ~mask1
    )


    panorama[
        only_img2
    ] = canvas_img2[
        only_img2
    ]


    # --------------------------------------------------------
    # Zona de overlap
    # --------------------------------------------------------

    overlap = (
        mask1
        &
        mask2
    )


    # Baseline blending:
    # média simples dos pixels
    panorama[
        overlap
    ] = (
        (
            warped_img1[
                overlap
            ].astype(
                np.float32
            )
            +
            canvas_img2[
                overlap
            ].astype(
                np.float32
            )
        )
        /
        2
    ).astype(
        np.uint8
    )


    return panorama


# ============================================================
# 10. GUARDAR IMAGEM
# ============================================================

def save_image(
    image,
    path
):

    path = Path(
        path
    )

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

def show_image(
    title,
    image
):

    cv2.imshow(
        title,
        image
    )


# ============================================================
# 12. CALCULAR ERRO MÉDIO DE REPROJEÇÃO
# ============================================================

def calculate_reprojection_error(
    kp1,
    kp2,
    matches,
    H,
    mask
):

    src_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in matches
        ]
    ).reshape(
        -1,
        1,
        2
    )


    dst_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in matches
        ]
    ).reshape(
        -1,
        1,
        2
    )


    # Projetar pontos da imagem 1
    projected_pts = cv2.perspectiveTransform(
        src_pts,
        H
    )


    # Erro em pixels
    errors = np.linalg.norm(
        projected_pts.reshape(
            -1,
            2
        )
        -
        dst_pts.reshape(
            -1,
            2
        ),
        axis=1
    )


    # Apenas inliers do RANSAC
    inliers = (
        mask
        .ravel()
        .astype(bool)
    )


    inlier_errors = errors[
        inliers
    ]


    if len(inlier_errors) == 0:

        return float("nan")


    average_error = np.mean(
        inlier_errors
    )


    return average_error


# ============================================================
# 13. CARREGAR HOMOGRAFIA GROUND TRUTH
# ============================================================

def load_ground_truth_homography(
    path
):

    H_gt = np.loadtxt(
        str(path)
    )

    return H_gt


# ============================================================
# 14. GARANTIR MATCHES ONE-TO-ONE
# ============================================================

def make_matches_one_to_one(
    matches
):

    # Ordenar melhores matches primeiro
    sorted_matches = sorted(
        matches,
        key=lambda m: m.distance
    )


    used_query = set()
    used_train = set()

    unique_matches = []


    for match in sorted_matches:

        # Não reutilizar o mesmo keypoint
        if (
            match.queryIdx not in used_query
            and
            match.trainIdx not in used_train
        ):

            unique_matches.append(
                match
            )

            used_query.add(
                match.queryIdx
            )

            used_train.add(
                match.trainIdx
            )


    return unique_matches


# ============================================================
# 15. AVALIAR MATCHES COM GROUND TRUTH
# ============================================================

def evaluate_matches_ground_truth(
    kp1,
    kp2,
    matches,
    H_gt,
    img2_shape,
    threshold=3.0
):

    if len(matches) == 0:

        raise ValueError(
            "Não existem matches para avaliar."
        )


    # --------------------------------------------------------
    # Pontos dos matches
    # --------------------------------------------------------

    src_pts = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in matches
        ]
    ).reshape(
        -1,
        1,
        2
    )


    dst_pts = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in matches
        ]
    ).reshape(
        -1,
        1,
        2
    )


    # --------------------------------------------------------
    # Aplicar ground-truth homography
    # --------------------------------------------------------

    projected_pts = cv2.perspectiveTransform(
        src_pts,
        H_gt
    )


    # --------------------------------------------------------
    # Calcular erro
    # --------------------------------------------------------

    errors = np.linalg.norm(
        projected_pts.reshape(
            -1,
            2
        )
        -
        dst_pts.reshape(
            -1,
            2
        ),
        axis=1
    )


    # Match correto se erro <= threshold
    correct_mask = (
        errors
        <=
        threshold
    )


    num_correct = int(
        correct_mask.sum()
    )


    num_putative = len(
        matches
    )


    # --------------------------------------------------------
    # Número de features
    # --------------------------------------------------------

    num_features = len(
        kp1
    )


    # --------------------------------------------------------
    # PMR
    # --------------------------------------------------------

    pmr = (
        num_putative
        /
        num_features
    )


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    precision = (
        num_correct
        /
        num_putative
    )


    # --------------------------------------------------------
    # MATCHING SCORE
    # --------------------------------------------------------

    matching_score = (
        num_correct
        /
        num_features
    )


    # --------------------------------------------------------
    # RECALL
    #
    # Contar keypoints de img1 que,
    # depois da transformação H_gt,
    # ficam dentro da img2.
    # --------------------------------------------------------

    all_src_pts = np.float32(
        [
            kp.pt
            for kp in kp1
        ]
    ).reshape(
        -1,
        1,
        2
    )


    all_projected = cv2.perspectiveTransform(
        all_src_pts,
        H_gt
    ).reshape(
        -1,
        2
    )


    height2, width2 = (
        img2_shape[:2]
    )


    inside_img2 = (
        (all_projected[:, 0] >= 0)
        &
        (all_projected[:, 0] < width2)
        &
        (all_projected[:, 1] >= 0)
        &
        (all_projected[:, 1] < height2)
    )


    num_possible_correspondences = int(
        inside_img2.sum()
    )


    if num_possible_correspondences > 0:

        recall = (
            num_correct
            /
            num_possible_correspondences
        )

    else:

        recall = 0.0


    # Para usar depois em cv2.drawMatches
    correct_mask = correct_mask.astype(
        np.uint8
    )


    return {

        "putative_matches":
            num_putative,

        "correct_matches":
            num_correct,

        "num_features":
            num_features,

        "possible_correspondences":
            num_possible_correspondences,

        "pmr":
            pmr,

        "precision":
            precision,

        "matching_score":
            matching_score,

        "recall":
            recall,

        "correct_mask":
            correct_mask,

        "errors":
            errors
    }