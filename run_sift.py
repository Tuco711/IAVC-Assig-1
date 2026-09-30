#deve ser este ficheiro a ser corrido; 
"""
results
└── sift
    ├── 01_keypoints_img1.jpg
    ├── 02_keypoints_img2.jpg
    ├── 03_good_matches.jpg
    ├── 04_ransac_inliers.jpg
    ├── 05_warped_img1.jpg
    ├── 06_panorama.jpg
    └── metrics.txt
"""
import cv2
from pathlib import Path

from pipeline import (
    load_image,
    to_gray,
    detect_sift,
    match_sift,
    estimate_homography,
    draw_keypoints,
    draw_matches,
    warp_to_reference,
    create_panorama,
    save_image,
    show_image
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "data" / "graf"
results_folder = project_folder / "results" / "sift"

img1_path = data_folder / "img1.ppm"
img2_path = data_folder / "img2.ppm"

LOWE_RATIO = 0.8
RANSAC_THRESHOLD = 3.0


# ============================================================
# 1. CARREGAR IMAGENS
# ============================================================

img1 = load_image(img1_path)
img2 = load_image(img2_path)

print("Imagens carregadas com sucesso!")


# ============================================================
# 2. GRAYSCALE
# ============================================================

gray1 = to_gray(img1)
gray2 = to_gray(img2)


# ============================================================
# 3. SIFT
# ============================================================

kp1, des1 = detect_sift(gray1)
kp2, des2 = detect_sift(gray2)

print("\n--- SIFT ---")
print("Keypoints imagem 1:", len(kp1))
print("Keypoints imagem 2:", len(kp2))


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
    results_folder / "01_keypoints_img1.jpg"
)

save_image(
    keypoints_img2,
    results_folder / "02_keypoints_img2.jpg"
)


# ============================================================
# 5. MATCHING
# ============================================================

matches, good_matches = match_sift(
    des1,
    des2,
    LOWE_RATIO
)

print("\n--- MATCHING ---")
print("Matches iniciais:", len(matches))
print("Good matches:", len(good_matches))


good_matches_image = draw_matches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches
)

save_image(
    good_matches_image,
    results_folder / "03_good_matches.jpg"
)


# ============================================================
# 6. HOMOGRAFIA + RANSAC
# ============================================================

H, mask = estimate_homography(
    kp1,
    kp2,
    good_matches,
    RANSAC_THRESHOLD
)

num_inliers = int(mask.sum())
num_matches = len(good_matches)

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

print("\n--- RANSAC ---")
print("Good matches:", num_matches)
print("Inliers:", num_inliers)
print("Outliers:", num_outliers)
print(
    "Inlier ratio:",
    round(inlier_ratio * 100, 2),
    "%"
)

print("\nHomografia:")
print(H)


# ============================================================
# 7. MOSTRAR APENAS INLIERS
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
    results_folder / "04_ransac_inliers.jpg"
)


# ============================================================
# 8. WARP
# ============================================================

warped_img1 = warp_to_reference(
    img1,
    img2,
    H
)

save_image(
    warped_img1,
    results_folder / "05_warped_img1.jpg"
)


# ============================================================
# 9. PANORAMA
# ============================================================

panorama = create_panorama(
    img1,
    img2,
    H
)

save_image(
    panorama,
    results_folder / "06_panorama.jpg"
)

print("\n--- PANORAMA ---")
print(
    "Panorama guardado em:",
    results_folder / "06_panorama.jpg"
)


# ============================================================
# 10. GUARDAR MÉTRICAS
# ============================================================

metrics_path = (
    results_folder
    /
    "metrics.txt"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"Keypoints imagem 1: {len(kp1)}\n"
    )

    file.write(
        f"Keypoints imagem 2: {len(kp2)}\n"
    )

    file.write(
        f"Matches iniciais: {len(matches)}\n"
    )

    file.write(
        f"Good matches: {num_matches}\n"
    )

    file.write(
        f"Inliers: {num_inliers}\n"
    )

    file.write(
        f"Outliers: {num_outliers}\n"
    )

    file.write(
        f"Inlier ratio: {inlier_ratio * 100:.2f}%\n"
    )

    file.write(
        f"Lowe ratio: {LOWE_RATIO}\n"
    )

    file.write(
        f"RANSAC threshold: {RANSAC_THRESHOLD}\n"
    )


# ============================================================
# 11. MOSTRAR RESULTADOS
# ============================================================

show_image(
    "1 - Keypoints imagem 1",
    keypoints_img1
)

show_image(
    "2 - Good Matches",
    good_matches_image
)

show_image(
    "3 - RANSAC Inliers",
    inliers_image
)

show_image(
    "4 - Warp",
    warped_img1
)

show_image(
    "5 - Panorama",
    panorama
)

cv2.waitKey(0)
cv2.destroyAllWindows()