# Statistical Machine Learning — Lab II

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

Implementations of core statistical machine learning methods, written for
the SML-II lab (Integrated M.Sc. Quantitative Economics & Data Science,
BIT Mesra). Each notebook is self-contained: problem statement, method,
code, output, and a short interpretation of the result.

---

## Dataset

All questions use the **BSDS500** (Berkeley Segmentation Data Set):
natural photographs with hand-drawn segmentation boundaries, split into
`images/train`, `images/test` and `images/val`.

BSDS500 is an image-segmentation dataset, not a tabular one, so each
question adapts it to the method being demonstrated — for example by
flattening resized grayscale images into feature vectors, or by deriving
per-pixel colour and gradient features with an edge/non-edge label. The
adaptation used is documented at the top of each notebook.

> The dataset is **not** committed to this repo (it is large, and image
> corpora are better downloaded than duplicated). Download BSDS500 and set
> the path as described in [Setup](#setup).

---

## Contents

| # | Topic | File | Key ideas |
|---|---|---|---|
| 1 | k-Nearest Neighbours | [`Q1_KNN.ipynb`](Q1_KNN.ipynb) | Distance metrics, choice of *k*, bias–variance |
| 2 | Decision Trees | [`Q2_DECISION_TREE.ipynb`](Q2_DECISION_TREE.ipynb) | Gini vs entropy, depth, overfitting |
| 3 | Gaussian Naive Bayes | [`Q3_Gaussian_Naive_Bayes.ipynb`](Q3_Gaussian_Naive_Bayes.ipynb) | Conditional independence, class priors |
| 4 | Logistic Regression | [`Q4_logistic_regression.ipynb`](Q4_logistic_regression.ipynb) | Sigmoid, log-loss, gradient descent |
| 5 | PCA & t-SNE | [`Q5_PCA_tSNE.py`](Q5_PCA_tSNE.py) | Dimensionality reduction, explained variance, 2D cluster structure |
| 6 | Outlier Detection | [`Q6_outlier_detection.ipynb`](Q6_outlier_detection.ipynb) | z-score, IQR, effect on model fit |
| 7 | ROC & AUC | [`Q7_ROC_AUC.ipynb`](Q7_ROC_AUC.ipynb) | Threshold trade-offs, why accuracy misleads |
| 8 | Simulated Annealing | [`Q8_SA_Minimization.ipynb`](Q8_SA_Minimization.ipynb) | Stochastic optimisation, cooling schedule |

---

## Setup

```bash
# 1. clone
git clone https://github.com/aasthariya27/SML-II-LAB-CODES.git
cd SML-II-LAB-CODES

# 2. environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. dependencies
pip install -r requirements.txt

# 4. point the code at your BSDS500 folder
#    (set the BSDS_DIR environment variable, or edit BASE_DIR at the
#     top of the file you are running)
set BSDS_DIR=C:\path\to\BSDS500     # Windows
# export BSDS_DIR=/path/to/BSDS500  # macOS / Linux
```

Then open any notebook with `jupyter notebook`, or run a script directly:

```bash
python Q5_PCA_tSNE.py
```

---

## Repository layout

```
Q1_KNN.ipynb                  ... Q8_SA_Minimization.ipynb
requirements.txt              pinned dependencies
.gitignore                    keeps data, venvs and checkpoints out of git
README.md
```

---

## Author

**Aastha** — Integrated M.Sc., Quantitative Economics & Data Science,
BIT Mesra
[GitHub](https://github.com/aasthariya27)
