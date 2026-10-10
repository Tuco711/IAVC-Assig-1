# ============================================================
# FICHEIRO PRINCIPAL PARA TESTAR KAZE
# ============================================================
#
# Resultados gerados:
#
# results/
# └── kaze/
#     ├── 01_keypoints_img1.jpg
#     ├── 02_keypoints_img2.jpg
#     ├── 03_good_matches.jpg
#     ├── 04_ransac_inliers.jpg
#     ├── 05_warped_img1.jpg
#     ├── 06_panorama.jpg
#     ├── 07_ground_truth_correct.jpg
#     ├── 08_ground_truth_incorrect.jpg
#     └── metrics.txt
# ============================================================


import cv2
import time
from pathlib import Path

from pipeline import (
    load_image,
    to_gray,
    detect_kaze,
    match_kaze,
    estimate_homography,
    calculate_reprojection_error,
    draw_keypoints,
    draw_matches,
    warp_to_reference,
    create_panorama,
    save_image,
    show_image,
    load_ground_truth_homography,
    make_matches_one_to_one,
    evaluate_matches_ground_truth
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "data" / "graf"

results_folder = (
    project_folder
    / "results"
    / "kaze"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)


img1_path = (
    data_folder
    / "img1.ppm"
)

img2_path = (
    data_folder
    / "img2.ppm"
)


# Homografia verdadeira img1 -> img2
ground_truth_path = (
    data_folder
    / "H1to2p.txt"
)


# Parâmetros
LOWE_RATIO = 0.8

RANSAC_THRESHOLD = 3.0

GROUND_TRUTH_THRESHOLD = 3.0


# ============================================================
# 1. CARREGAR IMAGENS
# ============================================================

img1 = load_image(
    img1_path
)

img2 = load_image(
    img2_path
)

print(
    "Imagens carregadas com sucesso!"
)


# ============================================================
# 2. CONVERTER PARA GRAYSCALE
# ============================================================

gray1 = to_gray(
    img1
)

gray2 = to_gray(
    img2
)


# ============================================================
# 3. KAZE - DETEÇÃO + DESCRIÇÃO
# ============================================================

start_time = time.perf_counter()


kp1, des1 = detect_kaze(
    gray1
)

kp2, des2 = detect_kaze(
    gray2
)


feature_time = (
    time.perf_counter()
    -
    start_time
)


# Segurança
if des1 is None or des2 is None:

    print(
        "ERRO: KAZE não conseguiu calcular descritores."
    )

    exit()


# Dimensão de cada descritor
descriptor_dimension = (
    des1.shape[1]
)


# Memória total dos descritores
descriptor_memory = (
    des1.nbytes
    +
    des2.nbytes
) / 1024


print(
    "\n--- KAZE ---"
)

print(
    "Keypoints imagem 1:",
    len(kp1)
)

print(
    "Keypoints imagem 2:",
    len(kp2)
)

print(
    "Tempo deteção + descrição:",
    round(
        feature_time * 1000,
        2
    ),
    "ms"
)

print(
    "Descriptor dimension:",
    descriptor_dimension
)

print(
    "Descriptor memory:",
    round(
        descriptor_memory,
        2
    ),
    "KB"
)


# ============================================================
# 4. GUARDAR KEYPOINTS
# ============================================================

keypoints_img1 = draw_keypoints(
    img1,
    kp1
)

keypoints_img2 = draw_keypoints(
    img2,
    kp2
)


save_image(
    keypoints_img1,
    results_folder
    / "01_keypoints_img1.jpg"
)

save_image(
    keypoints_img2,
    results_folder
    / "02_keypoints_img2.jpg"
)


# ============================================================
# 5. MATCHING
# ============================================================

start_time = time.perf_counter()


matches, good_matches = match_kaze(
    des1,
    des2,
    LOWE_RATIO
)


matching_time = (
    time.perf_counter()
    -
    start_time
)


print(
    "\n--- MATCHING ---"
)

print(
    "Matches iniciais:",
    len(matches)
)

print(
    "Good matches:",
    len(good_matches)
)

print(
    "Tempo matching:",
    round(
        matching_time * 1000,
        2
    ),
    "ms"
)


# Segurança
if len(good_matches) < 4:

    print(
        "ERRO: não existem matches suficientes "
        "para calcular a homografia."
    )

    exit()


# Visualizar good matches
good_matches_image = draw_matches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches
)


save_image(
    good_matches_image,
    results_folder
    / "03_good_matches.jpg"
)


# ============================================================
# 6. AVALIAÇÃO COM GROUND TRUTH
# ============================================================

H_gt = load_ground_truth_homography(
    ground_truth_path
)


# Garantir matches one-to-one
evaluation_matches = make_matches_one_to_one(
    good_matches
)


# Comparar matches com H1to2p.txt
gt_results = evaluate_matches_ground_truth(
    kp1,
    kp2,
    evaluation_matches,
    H_gt,
    img2.shape,
    GROUND_TRUTH_THRESHOLD
)


print(
    "\n--- GROUND TRUTH EVALUATION ---"
)

print(
    "Putative matches (one-to-one):",
    gt_results["putative_matches"]
)

print(
    "Correct matches:",
    gt_results["correct_matches"]
)

print(
    "PMR:",
    round(
        gt_results["pmr"] * 100,
        2
    ),
    "%"
)

print(
    "Precision:",
    round(
        gt_results["precision"] * 100,
        2
    ),
    "%"
)

print(
    "Matching Score:",
    round(
        gt_results["matching_score"] * 100,
        2
    ),
    "%"
)

print(
    "Recall:",
    round(
        gt_results["recall"] * 100,
        2
    ),
    "%"
)


# ============================================================
# 7. VISUALIZAR MATCHES CORRETOS
# ============================================================

correct_matches_image = draw_matches(
    img1,
    kp1,
    img2,
    kp2,
    evaluation_matches,
    gt_results["correct_mask"]
)


save_image(
    correct_matches_image,
    results_folder
    / "07_ground_truth_correct.jpg"
)


# ============================================================
# 8. VISUALIZAR MATCHES INCORRETOS
# ============================================================

incorrect_mask = (
    1
    -
    gt_results["correct_mask"]
)


incorrect_matches_image = draw_matches(
    img1,
    kp1,
    img2,
    kp2,
    evaluation_matches,
    incorrect_mask
)


save_image(
    incorrect_matches_image,
    results_folder
    / "08_ground_truth_incorrect.jpg"
)


# ============================================================
# 9. HOMOGRAFIA + RANSAC
# ============================================================

H, mask = estimate_homography(
    kp1,
    kp2,
    good_matches,
    RANSAC_THRESHOLD
)


average_reprojection_error = calculate_reprojection_error(
    kp1,
    kp2,
    good_matches,
    H,
    mask
)


num_inliers = int(
    mask.sum()
)

num_matches = len(
    good_matches
)

num_outliers = (
    num_matches
    -
    num_inliers
)

inlier_ratio = (
    num_inliers
    /
    num_matches
)


print(
    "\n--- RANSAC ---"
)

print(
    "Good matches:",
    num_matches
)

print(
    "Inliers:",
    num_inliers
)

print(
    "Outliers:",
    num_outliers
)

print(
    "Inlier ratio:",
    round(
        inlier_ratio * 100,
        2
    ),
    "%"
)

print(
    "Average reprojection error:",
    round(
        average_reprojection_error,
        4
    ),
    "pixels"
)

print(
    "\nHomografia:"
)

print(
    H
)


# ============================================================
# 10. GUARDAR RANSAC INLIERS
# ============================================================

inliers_image = draw_matches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    mask
)


save_image(
    inliers_image,
    results_folder
    / "04_ransac_inliers.jpg"
)


# ============================================================
# 11. WARP
# ============================================================

warped_img1 = warp_to_reference(
    img1,
    img2,
    H
)


save_image(
    warped_img1,
    results_folder
    / "05_warped_img1.jpg"
)


# ============================================================
# 12. PANORAMA
# ============================================================

panorama = create_panorama(
    img1,
    img2,
    H
)


save_image(
    panorama,
    results_folder
    / "06_panorama.jpg"
)


print(
    "\n--- PANORAMA ---"
)

print(
    "Panorama guardado em:",
    results_folder
    / "06_panorama.jpg"
)


# ============================================================
# 13. GUARDAR MÉTRICAS
# ============================================================

metrics_path = (
    results_folder
    / "metrics.txt"
)


with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Metodo: KAZE\n"
    )

    file.write(
        f"Keypoints imagem 1: "
        f"{len(kp1)}\n"
    )

    file.write(
        f"Keypoints imagem 2: "
        f"{len(kp2)}\n"
    )

    file.write(
        f"Matches iniciais: "
        f"{len(matches)}\n"
    )

    file.write(
        f"Good matches: "
        f"{num_matches}\n"
    )

    file.write(
        f"Inliers: "
        f"{num_inliers}\n"
    )

    file.write(
        f"Outliers: "
        f"{num_outliers}\n"
    )

    file.write(
        f"Inlier ratio: "
        f"{inlier_ratio * 100:.2f}%\n"
    )

    file.write(
        f"Average reprojection error: "
        f"{average_reprojection_error:.4f} pixels\n"
    )

    file.write(
        f"Feature extraction time: "
        f"{feature_time * 1000:.2f} ms\n"
    )

    file.write(
        f"Matching time: "
        f"{matching_time * 1000:.2f} ms\n"
    )

    file.write(
        f"Descriptor dimension: "
        f"{descriptor_dimension}\n"
    )

    file.write(
        f"Descriptor memory: "
        f"{descriptor_memory:.2f} KB\n"
    )

    file.write(
        f"Lowe ratio: "
        f"{LOWE_RATIO}\n"
    )

    file.write(
        f"RANSAC threshold: "
        f"{RANSAC_THRESHOLD}\n"
    )

    file.write(
        "Matching distance: L2\n"
    )


    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    file.write(
        "\n--- Ground Truth Evaluation ---\n"
    )

    file.write(
        f"Ground truth threshold: "
        f"{GROUND_TRUTH_THRESHOLD} pixels\n"
    )

    file.write(
        f"Putative matches one-to-one: "
        f"{gt_results['putative_matches']}\n"
    )

    file.write(
        f"Correct matches: "
        f"{gt_results['correct_matches']}\n"
    )

    file.write(
        f"Possible correspondences: "
        f"{gt_results['possible_correspondences']}\n"
    )

    file.write(
        f"PMR: "
        f"{gt_results['pmr'] * 100:.2f}%\n"
    )

    file.write(
        f"Precision: "
        f"{gt_results['precision'] * 100:.2f}%\n"
    )

    file.write(
        f"Matching Score: "
        f"{gt_results['matching_score'] * 100:.2f}%\n"
    )

    file.write(
        f"Recall: "
        f"{gt_results['recall'] * 100:.2f}%\n"
    )


print(
    "Métricas guardadas em:",
    metrics_path
)


# ============================================================
# 14. MOSTRAR RESULTADOS
# ============================================================

show_image(
    "1 - KAZE Keypoints imagem 1",
    keypoints_img1
)

show_image(
    "2 - KAZE Good Matches",
    good_matches_image
)

show_image(
    "3 - KAZE RANSAC Inliers",
    inliers_image
)

show_image(
    "4 - KAZE Warp",
    warped_img1
)

show_image(
    "5 - KAZE Panorama",
    panorama
)


cv2.waitKey(0)

cv2.destroyAllWindows()