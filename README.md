# Assignment #1 — Keypoint Detectors, Feature Descriptors and Applications

Pipeline de deteção de keypoints, descritores locais, matching, homografia
(RANSAC) e stitching de imagens, avaliado com o protocolo de ground truth do
enunciado (PMR, Precision, Matching Score, Recall).

## Ambiente

- Python 3.12 (ver `.python-version`).
- Gestor de pacotes: [uv](https://docs.astral.sh/uv/). Dependências e
  versões exatas em `pyproject.toml` / `uv.lock` (equivalente a
  `requirements.txt`).
- Dependências principais: `numpy==2.5.3`, `opencv-contrib-python==4.11.0.86`
  (build *contrib*, necessário para o FREAK via `cv2.xfeatures2d`; é um
  superset do `opencv-python`, não instalar os dois em simultâneo).
- Não é necessário GPU para os métodos atualmente implementados.

Criar o ambiente:

```bash
uv sync
```

(Alternativa sem `uv`: `pip install numpy==2.5.3 opencv-contrib-python==4.11.0.86`.)

## Correr as experiências

```bash
uv run main.py
```

Isto corre, para cada método configurado no dicionário `EXPERIMENTS` em
`main.py`, o pipeline completo (deteção → descrição → matching → avaliação
ground truth → RANSAC/homografia → warp → panorama) sobre o par de imagens
GRAF (`data/graf/img1.ppm` / `img2.ppm`, com `H1to2p.txt` como homografia
ground truth), e grava os resultados em `results/<método>/`.

Para adicionar ou remover um método da comparação, edita o dicionário
`EXPERIMENTS` em `main.py` — cada entrada só precisa de `detect_fn`,
`match_fn` e `distance_label` (as funções `detect_*`/`match_*` vivem em
`pipeline.py`).

## Estrutura

```
pipeline.py   # funções genéricas: load, detetores/descritores, matching,
              # RANSAC, homografia, ground truth, warp, panorama,
              # run_experiment() (orquestra tudo) e summarize_results()
main.py       # ponto de entrada; grid EXPERIMENTS com a configuração de
              # cada método a comparar
data/
  graf/       # dataset GRAF (viewpoint change), com homografias ground
              # truth H1to2p.txt (img1→img2) e H1to4p.txt (img1→img4)
  Panorama/   # tripleto keble_a/b/c.jpg para stitching de 3 imagens
  vehicle-seqs/  # sequências sem ground truth (uso apenas qualitativo)
results/<método>/
  01_keypoints_img1.jpg
  02_keypoints_img2.jpg
  03_good_matches.jpg
  04_ransac_inliers.jpg
  05_warped_img1.jpg
  06_panorama.jpg
  07_ground_truth_correct.jpg
  08_ground_truth_incorrect.jpg
  metrics.txt
results/comparison.csv   # tabela-resumo com uma linha por método
```

## Parâmetros usados

- Lowe ratio test: 0.8
- Threshold do RANSAC (`cv2.findHomography`): 3.0 px
- Threshold de correção ground truth: 3.0 px (match correto se o ponto
  transformado pela homografia ground truth cai a ≤3px do ponto emparelhado)
- Matches avaliados contra ground truth: one-to-one (cada keypoint só é
  usado uma vez, mantendo o melhor match por distância)

Nota: os números de inliers/outliers do RANSAC podem variar ligeiramente
entre execuções — `cv2.findHomography(..., cv2.RANSAC, ...)` depende da
amostragem aleatória interna do OpenCV. Keypoints, matches, PMR, Precision,
Matching Score e Recall são deterministicos para os mesmos parâmetros.

## Métodos avaliados (estado atual)

| Método | Detetor | Descritor | Distância |
|---|---|---|---|
| SIFT | SIFT | SIFT (float, 128-d) | L2 |
| ORB  | ORB  | ORB (binário, 32 bytes) | Hamming |
| KAZE | KAZE | KAZE (float, 64-d) | L2 |
| BRISK | BRISK | BRISK (binário, 64 bytes) | Hamming |
| FAST + BRIEF (uniforme) | FAST | BRIEF próprio, pares uniformes no patch | Hamming |
| FAST + BRIEF (gaussiana) | FAST | BRIEF próprio, pares ~Gaussiana isotrópica | Hamming |
| FAST + FREAK | FAST | FREAK (`cv2.xfeatures2d`, binário, 64 bytes) | Hamming |

A implementação própria do BRIEF está em `compute_brief()` em `pipeline.py`:
compara intensidades (numa versão desfocada da imagem) em pares de pontos
amostrados num patch de 31×31 à volta de cada keypoint, com padrão de
amostragem fixo (seed=42) reutilizado em todos os keypoints/imagens — tal
como no BRIEF original. Não tem informação de orientação, por isso não é
invariante a rotação (limitação conhecida do BRIEF "vanilla", ao contrário
do rBRIEF usado no ORB). A estratégia gaussiana supera consistentemente a
uniforme em qualidade de matching, coerente com o paper original do BRIEF.

Em falta face ao enunciado (ver plano de implementação): SuperPoint +
LightGlue, repeatability, segunda categoria de transformação (iluminação),
stitching de 3+ imagens e o relatório final.
