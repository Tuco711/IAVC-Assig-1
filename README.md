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
  superset do `opencv-python`, não instalar os dois em simultâneo), `torch`
  e `transformers` (SuperPoint + LightGlue).
- CPU é suficiente para todos os métodos, incluindo SuperPoint/LightGlue
  (duas imagens de cada vez) — não é necessário GPU.
- A primeira execução descarrega automaticamente os pesos pré-treinados do
  SuperPoint (`magic-leap-community/superpoint`) e do LightGlue
  (`ETH-CVG/lightglue_superpoint`) a partir do Hugging Face Hub — precisa
  de ligação à internet nessa primeira vez (ficam em cache local depois).

Criar o ambiente:

```bash
uv sync
```

(Alternativa sem `uv`: `pip install numpy==2.5.3 opencv-contrib-python==4.11.0.86 torch transformers pillow`.)

## Correr as experiências

```bash
uv run main.py
```

Isto corre, para cada combinação *scene* × método (dicionários `SCENES` e
`EXPERIMENTS` em `main.py`), o pipeline completo (deteção → descrição →
matching → avaliação ground truth → repeatability → RANSAC/homografia →
warp → panorama), e grava os resultados em `results/<scene>/<método>/`.
No fim corre também um varrimento de parâmetros do FAST (`threshold` ×
`nonmaxSuppression`, comparado com SIFT), gravado em
`results/fast_threshold_sweep.csv`.

Para adicionar ou remover um método da comparação, edita o dicionário
`EXPERIMENTS` em `main.py` — cada entrada só precisa de `detect_fn`,
`match_fn` e `distance_label` (as funções `detect_*`/`match_*` vivem em
`pipeline.py`). Para adicionar uma scene nova (outro par de imagens +
homografia ground truth), edita o dicionário `SCENES`.

### Scenes

| Scene | Categoria | Imagens | Ground truth |
|---|---|---|---|
| `graf_v_moderate` | viewpoint moderado | GRAF `img1`↔`img2` | `H1to2p.txt` |
| `graf_v_large` | viewpoint grande | GRAF `img1`↔`img4` | `H1to4p.txt` |
| `hpatches_illumination_moderate` | iluminação moderada | HPatches `i_ajuntament` `1.ppm`↔`3.ppm` | `H_1_3` (identidade — câmara estática) |
| `hpatches_illumination_extreme` | iluminação extrema (dia→noite) | HPatches `i_ajuntament` `1.ppm`↔`6.ppm` | `H_1_6` (identidade — câmara estática) |

A sequência `i_ajuntament` vem do
[HPatches sequences-release](https://huggingface.co/datasets/vbalnt/hpatches)
(Balntas, Lenc, Vedaldi, Mikolajczyk, "HPatches: A Benchmark and Evaluation
of Handcrafted and Learned Local Descriptors", CVPR 2017). Só esta sequência
está incluída no repositório (os restantes ~115 sequências do dataset
completo, ~1.2GB, não são necessárias e não foram incluídas). O par
`1.ppm`↔`6.ppm` é a maior diferença de iluminação da sequência — uma
transição dia/noite, teste de robustez deliberadamente exigente, bem mais
severo que uma simples variação de brilho/contraste. `1.ppm`↔`3.ppm` dá uma
variação de iluminação mais moderada da mesma cena, para comparação de
severidade (espelha `graf_v_moderate`/`graf_v_large`). Em todos os métodos,
repeatability e matching score pioram consistentemente de moderado para
extremo.

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
  hpatches/   # sequência HPatches i_ajuntament (iluminação), 1.ppm..6.ppm
              # + homografias H_1_2..H_1_6
  Panorama/   # tripleto keble_a/b/c.jpg para stitching de 3 imagens
  vehicle-seqs/  # sequências sem ground truth (uso apenas qualitativo)
results/<scene>/<método>/
  01_keypoints_img1.jpg
  02_keypoints_img2.jpg
  03_good_matches.jpg
  04_ransac_inliers.jpg
  05_warped_img1.jpg
  06_panorama.jpg
  07_ground_truth_correct.jpg
  08_ground_truth_incorrect.jpg
  metrics.txt
results/<scene>/comparison.csv   # tabela-resumo com uma linha por método
results/fast_threshold_sweep.csv # varrimento threshold x NMS do FAST vs. SIFT
```

## Parâmetros usados

- Lowe ratio test: 0.8
- Threshold do RANSAC (`cv2.findHomography`): 3.0 px
- Threshold de correção ground truth: 3.0 px (match correto se o ponto
  transformado pela homografia ground truth cai a ≤3px do ponto emparelhado)
- Matches avaliados contra ground truth: one-to-one (cada keypoint só é
  usado uma vez, mantendo o melhor match por distância)
- Repeatability: mesmo threshold (3.0 px) — fração de keypoints de img1,
  projetados para img2 via homografia ground truth e dentro dos limites de
  img2, que têm um keypoint de img2 a ≤3px de distância

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
| SuperPoint + BF | SuperPoint (learned) | SuperPoint (learned, float, 256-d) | L2 |
| SuperPoint + LightGlue | SuperPoint (learned) | — | matcher aprendido (LightGlue) |

SuperPoint e LightGlue usam checkpoints pré-treinados via Hugging Face
`transformers` (`magic-leap-community/superpoint`,
`ETH-CVG/lightglue_superpoint`) — não são treinados neste projeto. O
LightGlue deteta e empareilha num único passo conjunto (função dedicada
`run_superpoint_lightglue_experiment()` em `pipeline.py`, não encaixa no
par `detect_fn`/`match_fn` dos outros métodos); por isso os seus
`keypoints_img1`/`img2` em `comparison.csv` contam só os pontos já
emparelhados (acima do threshold do LightGlue), não a totalidade de
deteções do SuperPoint — essa contagem está na linha "superpoint"
separada. Seguindo a secção 4 do enunciado, SuperPoint e LightGlue são
tratados como configurações conceptualmente distintas (detetor/descritor
vs. matcher), nunca como um único "detetor".

A implementação própria do BRIEF está em `compute_brief()` em `pipeline.py`:
compara intensidades (numa versão desfocada da imagem) em pares de pontos
amostrados num patch de 31×31 à volta de cada keypoint, com padrão de
amostragem fixo (seed=42) reutilizado em todos os keypoints/imagens — tal
como no BRIEF original. Não tem informação de orientação, por isso não é
invariante a rotação (limitação conhecida do BRIEF "vanilla", ao contrário
do rBRIEF usado no ORB). A estratégia gaussiana supera consistentemente a
uniforme em qualidade de matching, coerente com o paper original do BRIEF.

Em falta face ao enunciado (ver plano de implementação): stitching de 3+
imagens e o relatório final.
