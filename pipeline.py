######
# Este ficheiro contém as funções genéricas:
# carregar imagens, detetores, descritores, matching,
# RANSAC, homografia, ground truth, warp e panorama.
######

import cv2
import numpy as np
import time
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


# ============================================================
# 16. CORRER UMA EXPERIÊNCIA COMPLETA (deteção -> matching ->
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
        **gt_results,
    }