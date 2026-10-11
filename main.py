from pathlib import Path

from pipeline import (
    run_experiment,
    summarize_results,
    run_fast_threshold_sweep,
    run_superpoint_lightglue_experiment,
    detect_sift,
    match_sift,
    detect_orb,
    match_orb,
    detect_kaze,
    match_kaze,
    detect_brisk,
    detect_fast_freak,
    detect_fast_brief_uniform,
    detect_fast_brief_gaussian,
    detect_superpoint,
)


# ============================================================
# CONFIGURAÇÃO PARTILHADA
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
GRAF_DIR = PROJECT_DIR / "data" / "graf"
HPATCHES_DIR = PROJECT_DIR / "data" / "hpatches"
RESULTS_DIR = PROJECT_DIR / "results"

COMMON_PARAMS = dict(
    ratio=0.8,
    ransac_threshold=3.0,
    gt_threshold=3.0,
)


# ============================================================
# SCENES — pares de imagens + homografia ground truth a testar.
#
# Cada scene corre o grid EXPERIMENTS completo. Para acrescentar
# uma categoria nova (ex. iluminação via HPatches), basta
# adicionar uma entrada aqui com img1_path/img2_path/
# ground_truth_path.
# ============================================================

SCENES = {
    "graf_v_moderate": dict(
        img1_path=GRAF_DIR / "img1.ppm",
        img2_path=GRAF_DIR / "img2.ppm",
        ground_truth_path=GRAF_DIR / "H1to2p.txt",
    ),
    "graf_v_large": dict(
        img1_path=GRAF_DIR / "img1.ppm",
        img2_path=GRAF_DIR / "img4.ppm",
        ground_truth_path=GRAF_DIR / "H1to4p.txt",
    ),
    "hpatches_illumination_moderate": dict(
        img1_path=HPATCHES_DIR / "i_ajuntament" / "1.ppm",
        img2_path=HPATCHES_DIR / "i_ajuntament" / "3.ppm",
        ground_truth_path=HPATCHES_DIR / "i_ajuntament" / "H_1_3",
    ),
    "hpatches_illumination_extreme": dict(
        img1_path=HPATCHES_DIR / "i_ajuntament" / "1.ppm",
        img2_path=HPATCHES_DIR / "i_ajuntament" / "6.ppm",
        ground_truth_path=HPATCHES_DIR / "i_ajuntament" / "H_1_6",
    ),
}


# ============================================================
# GRID DE EXPERIÊNCIAS
#
# Para adicionar um novo método (FAST, BRIEF, BRISK, FREAK,
# SuperPoint, ...), basta acrescentar uma entrada aqui com o
# detect_fn/match_fn correspondente em pipeline.py.
# ============================================================

EXPERIMENTS = {
    "sift": dict(
        detect_fn=detect_sift,
        match_fn=match_sift,
        distance_label="L2",
    ),
    "orb": dict(
        detect_fn=detect_orb,
        match_fn=match_orb,
        distance_label="Hamming",
    ),
    "kaze": dict(
        detect_fn=detect_kaze,
        match_fn=match_kaze,
        distance_label="L2",
    ),
    "brisk": dict(
        detect_fn=detect_brisk,
        match_fn=match_orb,
        distance_label="Hamming",
    ),
    "fast_brief_uniform": dict(
        detect_fn=detect_fast_brief_uniform,
        match_fn=match_orb,
        distance_label="Hamming",
    ),
    "fast_brief_gaussian": dict(
        detect_fn=detect_fast_brief_gaussian,
        match_fn=match_orb,
        distance_label="Hamming",
    ),
    "fast_freak": dict(
        detect_fn=detect_fast_freak,
        match_fn=match_orb,
        distance_label="Hamming",
    ),
    "superpoint": dict(
        detect_fn=detect_superpoint,
        match_fn=match_sift,
        distance_label="L2",
    ),
}


def main():

    for scene_name, scene_config in SCENES.items():

        results_by_method = {}

        for name, overrides in EXPERIMENTS.items():

            config = {**COMMON_PARAMS, **scene_config, **overrides}

            print(f"\n===== {scene_name.upper()} / {name.upper()} =====")

            results_by_method[name] = run_experiment(
                name=name,
                results_folder=RESULTS_DIR / scene_name / name,
                show=False,
                **config,
            )

        print(f"\n===== {scene_name.upper()} / SUPERPOINT_LIGHTGLUE =====")

        results_by_method["superpoint_lightglue"] = run_superpoint_lightglue_experiment(
            img1_path=scene_config["img1_path"],
            img2_path=scene_config["img2_path"],
            ground_truth_path=scene_config["ground_truth_path"],
            results_folder=RESULTS_DIR / scene_name / "superpoint_lightglue",
            ransac_threshold=COMMON_PARAMS["ransac_threshold"],
            gt_threshold=COMMON_PARAMS["gt_threshold"],
            show=False,
        )

        summarize_results(
            results_by_method,
            RESULTS_DIR / scene_name / "comparison.csv"
        )

    # Varrimento de parâmetros do FAST (threshold x NMS) vs. SIFT,
    # sobre a scene de viewpoint moderado.
    run_fast_threshold_sweep(
        img1_path=SCENES["graf_v_moderate"]["img1_path"],
        img2_path=SCENES["graf_v_moderate"]["img2_path"],
        ground_truth_path=SCENES["graf_v_moderate"]["ground_truth_path"],
        csv_path=RESULTS_DIR / "fast_threshold_sweep.csv",
        gt_threshold=COMMON_PARAMS["gt_threshold"],
    )


if __name__ == "__main__":
    main()
