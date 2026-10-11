from pathlib import Path

from pipeline import (
    run_experiment,
    detect_sift,
    match_sift,
    detect_orb,
    match_orb,
    detect_kaze,
    match_kaze,
)


# ============================================================
# CONFIGURAÇÃO PARTILHADA
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "graf"
RESULTS_DIR = PROJECT_DIR / "results"

DEFAULT_CONFIG = dict(
    img1_path=DATA_DIR / "img1.ppm",
    img2_path=DATA_DIR / "img2.ppm",
    ground_truth_path=DATA_DIR / "H1to2p.txt",
    ratio=0.8,
    ransac_threshold=3.0,
    gt_threshold=3.0,
)


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
}


def main():

    for name, overrides in EXPERIMENTS.items():

        config = {**DEFAULT_CONFIG, **overrides}

        print(f"\n===== {name.upper()} =====")

        run_experiment(
            name=name,
            results_folder=RESULTS_DIR / name,
            show=False,
            **config,
        )


if __name__ == "__main__":
    main()
