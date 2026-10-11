######
# Este ficheiro contém as funções genéricas:
# carregar imagens, detetores, descritores, matching,
# RANSAC, homografia, ground truth, warp e panorama.
######

import cv2
import numpy as np
import time
import csv
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


# ------------------------------------------------------------
# BRISK (detetor + descritor próprios)
# ------------------------------------------------------------

def detect_brisk(image_gray):

    brisk = cv2.BRISK_create()

    keypoints, descriptors = brisk.detectAndCompute(
        image_gray,
        None
    )

    return keypoints, descriptors


# ------------------------------------------------------------
# FAST + descritor externo (BRISK/FREAK sabem calcular
# descritores para keypoints dados por outro detetor)
# ------------------------------------------------------------

def _detect_then_describe(
    image_gray,
    detector,
    descriptor_extractor
):

    keypoints = detector.detect(
        image_gray,
        None
    )

    keypoints, descriptors = descriptor_extractor.compute(
        image_gray,
        keypoints
    )

    return keypoints, descriptors


def detect_fast_freak(image_gray):

    fast = cv2.FastFeatureDetector_create()
    freak = cv2.xfeatures2d.FREAK_create()

    return _detect_then_describe(
        image_gray,
        fast,
        freak
    )


# ------------------------------------------------------------
# BRIEF (implementação própria, não a do opencv-contrib)
#
# Para cada keypoint, compara intensidades (numa versão
# ligeiramente desfocada da imagem, como no paper original)
# em pares de pontos amostrados dentro de um patch à volta
# do keypoint. O padrão de pares é fixo (seed fixa) e reusado
# em todos os keypoints/imagens, tal como no BRIEF original.
#
# Duas estratégias de amostragem (parâmetro `pattern`):
# - "uniform":  pares uniformemente distribuídos no patch.
# - "gaussian": pares amostrados de uma Gaussiana isotrópica
#               centrada no keypoint (aprox. à estratégia
#               "G I" do paper original).
#
# Nota: sem informação de orientação (ao contrário do rBRIEF
# usado no ORB) -> não é invariante a rotação. Limitação
# conhecida e esperada do BRIEF "vanilla".
# ------------------------------------------------------------

def compute_brief(
    image_gray,
    keypoints,
    pattern="uniform",
    patch_size=31,
    n_bits=256,
    seed=42
):

    if len(keypoints) == 0:
        return keypoints, np.zeros((0, n_bits // 8), dtype=np.uint8)

    blurred = cv2.GaussianBlur(
        image_gray,
        (5, 5),
        2.0
    )

    height, width = blurred.shape

    half = patch_size // 2

    rng = np.random.default_rng(seed)

    if pattern == "uniform":

        offsets = rng.integers(
            -half,
            half + 1,
            size=(n_bits, 2, 2)
        )

    elif pattern == "gaussian":

        sigma = patch_size / 5.0

        offsets = rng.normal(
            0,
            sigma,
            size=(n_bits, 2, 2)
        )

        offsets = np.clip(
            np.round(offsets),
            -half,
            half
        ).astype(int)

    else:

        raise ValueError(
            f"Padrão de amostragem BRIEF desconhecido: {pattern}"
        )

    # Descartar keypoints demasiado perto da margem para
    # caberem o patch completo
    valid_keypoints = [
        kp for kp in keypoints
        if half <= kp.pt[0] <= width - 1 - half
        and half <= kp.pt[1] <= height - 1 - half
    ]

    if len(valid_keypoints) == 0:
        return valid_keypoints, np.zeros((0, n_bits // 8), dtype=np.uint8)

    centers = np.array(
        [kp.pt for kp in valid_keypoints]
    )

    pts1 = np.round(
        centers[:, None, :] + offsets[None, :, 0, :]
    ).astype(int)

    pts2 = np.round(
        centers[:, None, :] + offsets[None, :, 1, :]
    ).astype(int)

    pts1[..., 0] = np.clip(pts1[..., 0], 0, width - 1)
    pts1[..., 1] = np.clip(pts1[..., 1], 0, height - 1)
    pts2[..., 0] = np.clip(pts2[..., 0], 0, width - 1)
    pts2[..., 1] = np.clip(pts2[..., 1], 0, height - 1)

    intensities1 = blurred[pts1[..., 1], pts1[..., 0]]
    intensities2 = blurred[pts2[..., 1], pts2[..., 0]]

    bits = (intensities1 < intensities2).astype(np.uint8)

    descriptors = np.packbits(bits, axis=1)

    return valid_keypoints, descriptors


def detect_fast_brief_uniform(image_gray):

    fast = cv2.FastFeatureDetector_create()

    keypoints = fast.detect(image_gray, None)

    return compute_brief(
        image_gray,
        keypoints,
        pattern="uniform"
    )


def detect_fast_brief_gaussian(image_gray):

    fast = cv2.FastFeatureDetector_create()

    keypoints = fast.detect(image_gray, None)

    return compute_brief(
        image_gray,
        keypoints,
        pattern="gaussian"
    )


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


# ============================================================
# 16. REPEATABILITY
#
# Projeta kp1 para o referencial de img2 via H_gt e conta a
# fração desses pontos (os que caem dentro de img2) que têm
# um keypoint de kp2 a <= threshold pixels de distância.
# ============================================================

def compute_repeatability(
    kp1,
    kp2,
    H_gt,
    img2_shape,
    threshold=3.0
):

    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0, 0

    pts1 = np.float32(
        [kp.pt for kp in kp1]
    ).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(
        pts1,
        H_gt
    ).reshape(-1, 2)

    height2, width2 = img2_shape[:2]

    inside_img2 = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < width2)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height2)
    )

    projected_inside = projected[inside_img2]

    if len(projected_inside) == 0:
        return 0.0, 0

    pts2 = np.float32([kp.pt for kp in kp2])

    # Distância de cada ponto projetado ao keypoint mais
    # próximo em img2 (broadcasting: N_inside x N_kp2)
    diffs = projected_inside[:, None, :] - pts2[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    min_dists = dists.min(axis=1)

    num_repeated = int((min_dists <= threshold).sum())
    num_projected_inside = len(projected_inside)

    repeatability = num_repeated / num_projected_inside

    return repeatability, num_projected_inside


# ============================================================
# 17. CORRER UMA EXPERIÊNCIA COMPLETA (deteção -> matching ->
#     RANSAC -> warp -> panorama -> métricas), parametrizada
#     por detetor/descritor/matcher.
# ============================================================

def run_experiment(
    name,
    detect_fn,
    match_fn,
    distance_label,
    img1_path,
    img2_path,
    ground_truth_path,
    results_folder,
    ratio=0.8,
    ransac_threshold=3.0,
    gt_threshold=3.0,
    show=False,
):

    results_folder = Path(results_folder)

    results_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    label = name.upper()


    # --------------------------------------------------------
    # 1. CARREGAR IMAGENS
    # --------------------------------------------------------

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    print("Imagens carregadas com sucesso!")


    # --------------------------------------------------------
    # 2. CONVERTER PARA GRAYSCALE
    # --------------------------------------------------------

    gray1 = to_gray(img1)
    gray2 = to_gray(img2)


    # --------------------------------------------------------
    # 3. DETEÇÃO + DESCRIÇÃO
    # --------------------------------------------------------

    start_time = time.perf_counter()

    kp1, des1 = detect_fn(gray1)
    kp2, des2 = detect_fn(gray2)

    feature_time = time.perf_counter() - start_time


    if des1 is None or des2 is None:
        print(f"ERRO: {label} não conseguiu calcular descritores.")
        return None


    descriptor_dimension = des1.shape[1]

    descriptor_memory = (
        des1.nbytes + des2.nbytes
    ) / 1024


    print(f"\n--- {label} ---")

    print("Keypoints imagem 1:", len(kp1))
    print("Keypoints imagem 2:", len(kp2))

    print(
        "Tempo deteção + descrição:",
        round(feature_time * 1000, 2),
        "ms"
    )

    print("Descriptor dimension:", descriptor_dimension)

    print(
        "Descriptor memory:",
        round(descriptor_memory, 2),
        "KB"
    )


    # --------------------------------------------------------
    # 4. GUARDAR KEYPOINTS
    # --------------------------------------------------------

    keypoints_img1 = draw_keypoints(img1, kp1)
    keypoints_img2 = draw_keypoints(img2, kp2)

    save_image(keypoints_img1, results_folder / "01_keypoints_img1.jpg")
    save_image(keypoints_img2, results_folder / "02_keypoints_img2.jpg")


    # --------------------------------------------------------
    # 5. MATCHING
    # --------------------------------------------------------

    start_time = time.perf_counter()

    matches, good_matches = match_fn(des1, des2, ratio)

    matching_time = time.perf_counter() - start_time


    print("\n--- MATCHING ---")

    print("Matches iniciais:", len(matches))
    print("Good matches:", len(good_matches))

    print(
        "Tempo matching:",
        round(matching_time * 1000, 2),
        "ms"
    )


    if len(good_matches) < 4:
        print(
            "ERRO: não existem matches suficientes "
            "para calcular a homografia."
        )
        return None


    good_matches_image = draw_matches(img1, kp1, img2, kp2, good_matches)

    save_image(good_matches_image, results_folder / "03_good_matches.jpg")


    # --------------------------------------------------------
    # 6. AVALIAÇÃO COM GROUND TRUTH
    # --------------------------------------------------------

    H_gt = load_ground_truth_homography(ground_truth_path)

    repeatability, repeatability_projected = compute_repeatability(
        kp1,
        kp2,
        H_gt,
        img2.shape,
        gt_threshold
    )

    print("\n--- REPEATABILITY ---")
    print("Repeatability:", round(repeatability * 100, 2), "%")
    print("Pontos projetados dentro de img2:", repeatability_projected)

    evaluation_matches = make_matches_one_to_one(good_matches)

    gt_results = evaluate_matches_ground_truth(
        kp1,
        kp2,
        evaluation_matches,
        H_gt,
        img2.shape,
        gt_threshold
    )


    print("\n--- GROUND TRUTH EVALUATION ---")

    print("Putative matches (one-to-one):", gt_results["putative_matches"])
    print("Correct matches:", gt_results["correct_matches"])
    print("PMR:", round(gt_results["pmr"] * 100, 2), "%")
    print("Precision:", round(gt_results["precision"] * 100, 2), "%")
    print("Matching Score:", round(gt_results["matching_score"] * 100, 2), "%")
    print("Recall:", round(gt_results["recall"] * 100, 2), "%")


    correct_matches_image = draw_matches(
        img1, kp1, img2, kp2,
        evaluation_matches,
        gt_results["correct_mask"]
    )

    save_image(correct_matches_image, results_folder / "07_ground_truth_correct.jpg")


    incorrect_mask = 1 - gt_results["correct_mask"]

    incorrect_matches_image = draw_matches(
        img1, kp1, img2, kp2,
        evaluation_matches,
        incorrect_mask
    )

    save_image(incorrect_matches_image, results_folder / "08_ground_truth_incorrect.jpg")


    # --------------------------------------------------------
    # 7. HOMOGRAFIA + RANSAC
    # --------------------------------------------------------

    H, mask = estimate_homography(
        kp1,
        kp2,
        good_matches,
        ransac_threshold
    )

    average_reprojection_error = calculate_reprojection_error(
        kp1, kp2, good_matches, H, mask
    )

    num_inliers = int(mask.sum())
    num_matches = len(good_matches)
    num_outliers = num_matches - num_inliers
    inlier_ratio = num_inliers / num_matches


    print("\n--- RANSAC ---")

    print("Good matches:", num_matches)
    print("Inliers:", num_inliers)
    print("Outliers:", num_outliers)
    print("Inlier ratio:", round(inlier_ratio * 100, 2), "%")

    print(
        "Average reprojection error:",
        round(average_reprojection_error, 4),
        "pixels"
    )

    print("\nHomografia:")
    print(H)


    inliers_image = draw_matches(img1, kp1, img2, kp2, good_matches, mask)

    save_image(inliers_image, results_folder / "04_ransac_inliers.jpg")


    # --------------------------------------------------------
    # 8. WARP
    # --------------------------------------------------------

    warped_img1 = warp_to_reference(img1, img2, H)

    save_image(warped_img1, results_folder / "05_warped_img1.jpg")


    # --------------------------------------------------------
    # 9. PANORAMA
    # --------------------------------------------------------

    panorama = create_panorama(img1, img2, H)

    save_image(panorama, results_folder / "06_panorama.jpg")

    print("\n--- PANORAMA ---")
    print("Panorama guardado em:", results_folder / "06_panorama.jpg")


    # --------------------------------------------------------
    # 10. GUARDAR MÉTRICAS
    # --------------------------------------------------------

    metrics_path = results_folder / "metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as file:

        file.write(f"Metodo: {label}\n")
        file.write(f"Keypoints imagem 1: {len(kp1)}\n")
        file.write(f"Keypoints imagem 2: {len(kp2)}\n")
        file.write(f"Matches iniciais: {len(matches)}\n")
        file.write(f"Good matches: {num_matches}\n")
        file.write(f"Inliers: {num_inliers}\n")
        file.write(f"Outliers: {num_outliers}\n")
        file.write(f"Inlier ratio: {inlier_ratio * 100:.2f}%\n")
        file.write(
            f"Average reprojection error: "
            f"{average_reprojection_error:.4f} pixels\n"
        )
        file.write(f"Feature extraction time: {feature_time * 1000:.2f} ms\n")
        file.write(f"Matching time: {matching_time * 1000:.2f} ms\n")
        file.write(f"Descriptor dimension: {descriptor_dimension}\n")
        file.write(f"Descriptor memory: {descriptor_memory:.2f} KB\n")
        file.write(f"Lowe ratio: {ratio}\n")
        file.write(f"RANSAC threshold: {ransac_threshold}\n")
        file.write(f"Matching distance: {distance_label}\n")

        file.write("\n--- Ground Truth Evaluation ---\n")
        file.write(f"Ground truth threshold: {gt_threshold} pixels\n")
        file.write(f"Repeatability: {repeatability * 100:.2f}%\n")
        file.write(f"Repeatability projected points: {repeatability_projected}\n")
        file.write(f"Putative matches one-to-one: {gt_results['putative_matches']}\n")
        file.write(f"Correct matches: {gt_results['correct_matches']}\n")
        file.write(f"Possible correspondences: {gt_results['possible_correspondences']}\n")
        file.write(f"PMR: {gt_results['pmr'] * 100:.2f}%\n")
        file.write(f"Precision: {gt_results['precision'] * 100:.2f}%\n")
        file.write(f"Matching Score: {gt_results['matching_score'] * 100:.2f}%\n")
        file.write(f"Recall: {gt_results['recall'] * 100:.2f}%\n")

    print("Métricas guardadas em:", metrics_path)


    # --------------------------------------------------------
    # 11. MOSTRAR RESULTADOS (opcional)
    # --------------------------------------------------------

    if show:

        show_image(f"1 - {label} Keypoints imagem 1", keypoints_img1)
        show_image(f"2 - {label} Good Matches", good_matches_image)
        show_image(f"3 - {label} RANSAC Inliers", inliers_image)
        show_image(f"4 - {label} Warp", warped_img1)
        show_image(f"5 - {label} Panorama", panorama)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return {
        "name": name,
        "keypoints_img1": len(kp1),
        "keypoints_img2": len(kp2),
        "good_matches": num_matches,
        "inliers": num_inliers,
        "outliers": num_outliers,
        "inlier_ratio": inlier_ratio,
        "average_reprojection_error": average_reprojection_error,
        "feature_time_ms": feature_time * 1000,
        "matching_time_ms": matching_time * 1000,
        "descriptor_dimension": descriptor_dimension,
        "descriptor_memory_kb": descriptor_memory,
        "repeatability": repeatability,
        "repeatability_projected_points": repeatability_projected,
        **gt_results,
    }


# ============================================================
# 17. TABELA-RESUMO ENTRE MÉTODOS (comparison.csv)
# ============================================================

COMPARISON_FIELDS = [
    "name",
    "keypoints_img1",
    "keypoints_img2",
    "descriptor_dimension",
    "descriptor_memory_kb",
    "feature_time_ms",
    "matching_time_ms",
    "good_matches",
    "inliers",
    "outliers",
    "inlier_ratio",
    "average_reprojection_error",
    "repeatability",
    "repeatability_projected_points",
    "putative_matches",
    "correct_matches",
    "pmr",
    "precision",
    "matching_score",
    "recall",
]


def summarize_results(
    results_by_method,
    csv_path
):

    csv_path = Path(csv_path)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COMPARISON_FIELDS
        )

        writer.writeheader()

        for name, result in results_by_method.items():

            if result is None:
                continue

            row = {}

            for field in COMPARISON_FIELDS:

                value = result.get(field, "")

                if isinstance(value, float):
                    value = round(value, 4)

                row[field] = value

            writer.writerow(row)

    print("\nTabela-resumo guardada em:", csv_path)


# ============================================================
# 18. VARRIMENTO DE PARÂMETROS DO FAST (threshold x NMS),
#     comparado com SIFT — secção 2.3 do enunciado.
# ============================================================

def run_fast_threshold_sweep(
    img1_path,
    img2_path,
    ground_truth_path,
    csv_path,
    thresholds=(10, 20, 30, 40),
    nms_options=(True, False),
    gt_threshold=3.0,
):

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    gray1 = to_gray(img1)
    gray2 = to_gray(img2)

    H_gt = load_ground_truth_homography(ground_truth_path)

    rows = []

    # --------------------------------------------------------
    # Baseline: SIFT
    # --------------------------------------------------------

    start_time = time.perf_counter()
    kp1, _ = detect_sift(gray1)
    kp2, _ = detect_sift(gray2)
    detection_time = time.perf_counter() - start_time

    repeatability, _ = compute_repeatability(
        kp1, kp2, H_gt, img2.shape, gt_threshold
    )

    rows.append({
        "config": "sift (baseline)",
        "threshold": "",
        "nonmax_suppression": "",
        "keypoints_img1": len(kp1),
        "keypoints_img2": len(kp2),
        "repeatability": round(repeatability, 4),
        "detection_time_ms": round(detection_time * 1000, 4),
    })

    # --------------------------------------------------------
    # FAST: threshold x nonmaxSuppression
    # --------------------------------------------------------

    for threshold in thresholds:

        for nms in nms_options:

            fast = cv2.FastFeatureDetector_create(
                threshold=threshold,
                nonmaxSuppression=nms
            )

            start_time = time.perf_counter()
            kp1 = fast.detect(gray1, None)
            kp2 = fast.detect(gray2, None)
            detection_time = time.perf_counter() - start_time

            repeatability, _ = compute_repeatability(
                kp1, kp2, H_gt, img2.shape, gt_threshold
            )

            rows.append({
                "config": f"fast_t{threshold}_nms{nms}",
                "threshold": threshold,
                "nonmax_suppression": nms,
                "keypoints_img1": len(kp1),
                "keypoints_img2": len(kp2),
                "repeatability": round(repeatability, 4),
                "detection_time_ms": round(detection_time * 1000, 4),
            })

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "config", "threshold", "nonmax_suppression",
        "keypoints_img1", "keypoints_img2",
        "repeatability", "detection_time_ms",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Varrimento do FAST guardado em:", csv_path)


# ============================================================
# 19. SUPERPOINT (detetor + descritor aprendidos)
#
# Checkpoint pré-treinado via Hugging Face transformers.
# Encaixa no mesmo contrato detect_fn(gray) -> (kp, des) dos
# detetores clássicos, por isso entra no grid EXPERIMENTS
# normalmente (emparelhado com match_sift, já que os
# descritores são floating-point 256-d).
# ============================================================

_SUPERPOINT_PROCESSOR = None
_SUPERPOINT_MODEL = None


def _get_superpoint():

    global _SUPERPOINT_PROCESSOR, _SUPERPOINT_MODEL

    if _SUPERPOINT_MODEL is None:

        from transformers import AutoImageProcessor, SuperPointForKeypointDetection

        _SUPERPOINT_PROCESSOR = AutoImageProcessor.from_pretrained(
            "magic-leap-community/superpoint"
        )

        _SUPERPOINT_MODEL = SuperPointForKeypointDetection.from_pretrained(
            "magic-leap-community/superpoint"
        )

        _SUPERPOINT_MODEL.eval()

    return _SUPERPOINT_PROCESSOR, _SUPERPOINT_MODEL


def detect_superpoint(image_gray):

    import torch

    processor, model = _get_superpoint()

    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    inputs = processor(image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    image_size = [(image_gray.shape[0], image_gray.shape[1])]

    processed = processor.post_process_keypoint_detection(
        outputs,
        image_size
    )[0]

    coords = processed["keypoints"].numpy()
    scores = processed["scores"].numpy()
    descriptors = processed["descriptors"].numpy().astype(np.float32)

    keypoints = [
        cv2.KeyPoint(float(x), float(y), 8, -1, float(s))
        for (x, y), s in zip(coords, scores)
    ]

    return keypoints, descriptors


# ============================================================
# 20. SUPERPOINT + LIGHTGLUE
#
# Configuração learned completa (deteção + matching) — secção
# 2.6 do enunciado. O LightGlue deteta e empareilha num único
# passo conjunto (não "detect_fn" + "match_fn" separados como
# nos outros métodos), por isso tem a sua própria função de
# experiência em vez de passar por run_experiment(). Reutiliza
# as mesmas funções de baixo nível (RANSAC, avaliação ground
# truth, repeatability, warp, panorama) para que os resultados
# fiquem no mesmo formato dos outros métodos.
# ============================================================

_LIGHTGLUE_PROCESSOR = None
_LIGHTGLUE_MODEL = None


def _get_lightglue():

    global _LIGHTGLUE_PROCESSOR, _LIGHTGLUE_MODEL

    if _LIGHTGLUE_MODEL is None:

        from transformers import AutoImageProcessor, AutoModel

        _LIGHTGLUE_PROCESSOR = AutoImageProcessor.from_pretrained(
            "ETH-CVG/lightglue_superpoint"
        )

        _LIGHTGLUE_MODEL = AutoModel.from_pretrained(
            "ETH-CVG/lightglue_superpoint"
        )

        _LIGHTGLUE_MODEL.eval()

    return _LIGHTGLUE_PROCESSOR, _LIGHTGLUE_MODEL


def run_superpoint_lightglue_experiment(
    img1_path,
    img2_path,
    ground_truth_path,
    results_folder,
    ransac_threshold=3.0,
    gt_threshold=3.0,
    match_threshold=0.2,
    show=False,
):

    import torch
    from PIL import Image

    name = "superpoint_lightglue"
    label = name.upper()

    results_folder = Path(results_folder)

    results_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    print("Imagens carregadas com sucesso!")

    pil1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    pil2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    processor, model = _get_lightglue()

    start_time = time.perf_counter()

    inputs = processor([pil1, pil2], return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    image_sizes = [[(pil1.height, pil1.width), (pil2.height, pil2.width)]]

    processed = processor.post_process_keypoint_matching(
        outputs,
        image_sizes,
        threshold=match_threshold
    )[0]

    detection_and_matching_time = time.perf_counter() - start_time

    keypoints0 = processed["keypoints0"].numpy()
    keypoints1 = processed["keypoints1"].numpy()
    scores = processed["matching_scores"].numpy()

    kp1 = [cv2.KeyPoint(float(x), float(y), 8) for x, y in keypoints0]
    kp2 = [cv2.KeyPoint(float(x), float(y), 8) for x, y in keypoints1]

    good_matches = [
        cv2.DMatch(i, i, float(1.0 - s))
        for i, s in enumerate(scores)
    ]

    matches = good_matches

    descriptor_dimension = 256

    descriptor_memory = (
        (len(kp1) + len(kp2)) * descriptor_dimension * 4
    ) / 1024

    print(f"\n--- {label} ---")

    print("Keypoints/matches imagem 1:", len(kp1))
    print("Keypoints/matches imagem 2:", len(kp2))

    print(
        "Tempo deteção+matching (conjunto, LightGlue):",
        round(detection_and_matching_time * 1000, 2),
        "ms"
    )

    if len(good_matches) < 4:
        print(
            "ERRO: não existem matches suficientes "
            "para calcular a homografia."
        )
        return None

    keypoints_img1 = draw_keypoints(img1, kp1)
    keypoints_img2 = draw_keypoints(img2, kp2)

    save_image(keypoints_img1, results_folder / "01_keypoints_img1.jpg")
    save_image(keypoints_img2, results_folder / "02_keypoints_img2.jpg")

    good_matches_image = draw_matches(img1, kp1, img2, kp2, good_matches)

    save_image(good_matches_image, results_folder / "03_good_matches.jpg")

    H_gt = load_ground_truth_homography(ground_truth_path)

    repeatability, repeatability_projected = compute_repeatability(
        kp1, kp2, H_gt, img2.shape, gt_threshold
    )

    print("\n--- REPEATABILITY ---")
    print("Repeatability:", round(repeatability * 100, 2), "%")

    evaluation_matches = make_matches_one_to_one(good_matches)

    gt_results = evaluate_matches_ground_truth(
        kp1,
        kp2,
        evaluation_matches,
        H_gt,
        img2.shape,
        gt_threshold
    )

    print("\n--- GROUND TRUTH EVALUATION ---")

    print("Putative matches (one-to-one):", gt_results["putative_matches"])
    print("Correct matches:", gt_results["correct_matches"])
    print("Precision:", round(gt_results["precision"] * 100, 2), "%")
    print("Recall:", round(gt_results["recall"] * 100, 2), "%")

    correct_matches_image = draw_matches(
        img1, kp1, img2, kp2,
        evaluation_matches,
        gt_results["correct_mask"]
    )

    save_image(correct_matches_image, results_folder / "07_ground_truth_correct.jpg")

    incorrect_mask = 1 - gt_results["correct_mask"]

    incorrect_matches_image = draw_matches(
        img1, kp1, img2, kp2,
        evaluation_matches,
        incorrect_mask
    )

    save_image(incorrect_matches_image, results_folder / "08_ground_truth_incorrect.jpg")

    H, mask = estimate_homography(
        kp1,
        kp2,
        good_matches,
        ransac_threshold
    )

    average_reprojection_error = calculate_reprojection_error(
        kp1, kp2, good_matches, H, mask
    )

    num_inliers = int(mask.sum())
    num_matches = len(good_matches)
    num_outliers = num_matches - num_inliers
    inlier_ratio = num_inliers / num_matches

    print("\n--- RANSAC ---")

    print("Good matches:", num_matches)
    print("Inliers:", num_inliers)
    print("Outliers:", num_outliers)
    print("Inlier ratio:", round(inlier_ratio * 100, 2), "%")

    print(
        "Average reprojection error:",
        round(average_reprojection_error, 4),
        "pixels"
    )

    inliers_image = draw_matches(img1, kp1, img2, kp2, good_matches, mask)

    save_image(inliers_image, results_folder / "04_ransac_inliers.jpg")

    warped_img1 = warp_to_reference(img1, img2, H)

    save_image(warped_img1, results_folder / "05_warped_img1.jpg")

    panorama = create_panorama(img1, img2, H)

    save_image(panorama, results_folder / "06_panorama.jpg")

    print("\n--- PANORAMA ---")
    print("Panorama guardado em:", results_folder / "06_panorama.jpg")

    metrics_path = results_folder / "metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as file:

        file.write(f"Metodo: {label}\n")

        file.write(
            "Nota: deteção e matching são um único passo conjunto "
            "(LightGlue), não divididos como nos outros métodos. "
            "keypoints_img1/img2 contam só os pontos emparelhados "
            "devolvidos pelo LightGlue (acima do threshold), não a "
            "totalidade de pontos SuperPoint detetados -- essa "
            "contagem está na linha 'superpoint' (SuperPoint + BF) "
            "da mesma tabela.\n"
        )

        file.write(f"Keypoints/matches imagem 1: {len(kp1)}\n")
        file.write(f"Keypoints/matches imagem 2: {len(kp2)}\n")
        file.write(f"Good matches: {num_matches}\n")
        file.write(f"Inliers: {num_inliers}\n")
        file.write(f"Outliers: {num_outliers}\n")
        file.write(f"Inlier ratio: {inlier_ratio * 100:.2f}%\n")
        file.write(
            f"Average reprojection error: "
            f"{average_reprojection_error:.4f} pixels\n"
        )
        file.write(
            f"Deteção+matching time (conjunto): "
            f"{detection_and_matching_time * 1000:.2f} ms\n"
        )
        file.write(f"Descriptor dimension: {descriptor_dimension}\n")
        file.write(f"Descriptor memory: {descriptor_memory:.2f} KB\n")
        file.write(f"LightGlue match threshold: {match_threshold}\n")
        file.write(f"RANSAC threshold: {ransac_threshold}\n")
        file.write("Matching distance: n/a (matcher aprendido, LightGlue)\n")

        file.write("\n--- Ground Truth Evaluation ---\n")
        file.write(f"Ground truth threshold: {gt_threshold} pixels\n")
        file.write(f"Repeatability: {repeatability * 100:.2f}%\n")
        file.write(f"Repeatability projected points: {repeatability_projected}\n")
        file.write(f"Putative matches one-to-one: {gt_results['putative_matches']}\n")
        file.write(f"Correct matches: {gt_results['correct_matches']}\n")
        file.write(f"Possible correspondences: {gt_results['possible_correspondences']}\n")
        file.write(f"PMR: {gt_results['pmr'] * 100:.2f}%\n")
        file.write(f"Precision: {gt_results['precision'] * 100:.2f}%\n")
        file.write(f"Matching Score: {gt_results['matching_score'] * 100:.2f}%\n")
        file.write(f"Recall: {gt_results['recall'] * 100:.2f}%\n")

    print("Métricas guardadas em:", metrics_path)

    if show:

        show_image(f"1 - {label} Keypoints imagem 1", keypoints_img1)
        show_image(f"2 - {label} Good Matches", good_matches_image)
        show_image(f"3 - {label} RANSAC Inliers", inliers_image)
        show_image(f"4 - {label} Warp", warped_img1)
        show_image(f"5 - {label} Panorama", panorama)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "name": name,
        "keypoints_img1": len(kp1),
        "keypoints_img2": len(kp2),
        "good_matches": num_matches,
        "inliers": num_inliers,
        "outliers": num_outliers,
        "inlier_ratio": inlier_ratio,
        "average_reprojection_error": average_reprojection_error,
        "feature_time_ms": 0.0,
        "matching_time_ms": detection_and_matching_time * 1000,
        "descriptor_dimension": descriptor_dimension,
        "descriptor_memory_kb": descriptor_memory,
        "repeatability": repeatability,
        "repeatability_projected_points": repeatability_projected,
        **gt_results,
    }


# ============================================================
# 21. STITCHING MULTI-IMAGEM (N >= 2)
#
# Generaliza create_panorama (que só serve para 2 imagens) a
# uma sequência de N imagens. Estima homografias entre pares
# adjacentes, compõe-as até à imagem de referência, calcula o
# canvas-união a partir dos cantos de todas as imagens, e funde
# por média acumulada (cada pixel do canvas é a média das
# imagens que o cobrem).
# ============================================================

def stitch_sequence(
    images,
    detect_fn,
    match_fn,
    reference_index=None,
    ratio=0.8,
    ransac_threshold=3.0,
):

    n = len(images)

    if n < 2:
        raise ValueError(
            "São necessárias pelo menos 2 imagens para stitching."
        )

    if reference_index is None:
        reference_index = n // 2

    grays = [to_gray(img) for img in images]

    keypoints_list = []
    descriptors_list = []

    for gray in grays:

        kp, des = detect_fn(gray)

        keypoints_list.append(kp)
        descriptors_list.append(des)

    # --------------------------------------------------------
    # Homografias entre pares adjacentes:
    # H_adjacent[i] mapeia imagem i -> imagem i+1
    # --------------------------------------------------------

    H_adjacent = []

    for i in range(n - 1):

        matches, good_matches = match_fn(
            descriptors_list[i],
            descriptors_list[i + 1],
            ratio
        )

        H, mask = estimate_homography(
            keypoints_list[i],
            keypoints_list[i + 1],
            good_matches,
            ransac_threshold
        )

        H_adjacent.append(H)

    # --------------------------------------------------------
    # Compor homografias até à imagem de referência
    # --------------------------------------------------------

    H_to_ref = [None] * n
    H_to_ref[reference_index] = np.eye(3)

    for i in range(reference_index + 1, n):

        H_to_ref[i] = H_to_ref[i - 1] @ np.linalg.inv(H_adjacent[i - 1])

    for i in range(reference_index - 1, -1, -1):

        H_to_ref[i] = H_to_ref[i + 1] @ H_adjacent[i]

    # --------------------------------------------------------
    # Canvas-união a partir dos cantos de todas as imagens
    # --------------------------------------------------------

    all_corners = []

    for i, img in enumerate(images):

        h, w = img.shape[:2]

        corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]]
        ).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(corners, H_to_ref[i])

        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)

    xmin, ymin = np.floor(
        all_corners.min(axis=0).ravel()
    ).astype(int)

    xmax, ymax = np.ceil(
        all_corners.max(axis=0).ravel()
    ).astype(int)

    tx, ty = -xmin, -ymin

    translation = np.array(
        [[1, 0, tx], [0, 1, ty], [0, 0, 1]],
        dtype=np.float64
    )

    canvas_width = xmax - xmin
    canvas_height = ymax - ymin

    # --------------------------------------------------------
    # Warp + blending (média acumulada por pixel)
    # --------------------------------------------------------

    accumulator = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for i, img in enumerate(images):

        H_canvas = translation @ H_to_ref[i]

        warped = cv2.warpPerspective(
            img, H_canvas, (canvas_width, canvas_height)
        )

        h, w = img.shape[:2]

        mask = cv2.warpPerspective(
            np.ones((h, w), dtype=np.uint8),
            H_canvas,
            (canvas_width, canvas_height),
            flags=cv2.INTER_NEAREST
        ).astype(bool)

        accumulator[mask] += warped[mask].astype(np.float32)
        weight[mask] += 1.0

    covered = weight > 0

    panorama = np.zeros_like(accumulator, dtype=np.uint8)

    panorama[covered] = (
        accumulator[covered] / weight[covered, None]
    ).astype(np.uint8)

    return panorama