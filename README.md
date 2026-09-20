# Statistical Machine Learning II — Lab (ED408)

Classical machine learning algorithms implemented **from scratch** where the lab asks for it and with **scikit-learn** where it asks for library use — all applied to a single dataset derived from **BSDS500 (Berkeley Segmentation Data Set)** — followed by metaheuristic optimisation methods.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Dataset](https://img.shields.io/badge/dataset-BSDS500-4B8BBE?style=flat-square)
![Progress](https://img.shields.io/badge/progress-9%20of%2014%20labs-brightgreen?style=flat-square)

---

## Course

| | |
|---|---|
| **Course** | ED408 — Statistical Machine Learning II Lab |
| **Credit structure** | 0–0–3–1.5 |
| **Programme** | Integrated M.Sc. in Quantitative Economics and Data Science |
| **Institute** | Birla Institute of Technology, Mesra |
| **Language** | Python |

---

## The dataset

Every question in this repository is answered on the **same** dataset, derived from **BSDS500 — the Berkeley Segmentation Data Set and Benchmarks 500**.

BSDS500 is a computer-vision benchmark: 500 natural images, each hand-annotated by several human subjects who marked object boundaries, split into disjoint train / validation / test sets. It ships as JPEG images plus ground-truth annotation files — **not** as a tabular dataset.

**Why that matters.** The lab tasks assume tabular input (Iris, MNIST, a spam/ham corpus). BSDS500 has to be converted before any classifier can touch it. The conversion turns image segmentation into a genuine **binary classification problem**:

- **One row = one pixel**, sampled from the BSDS500 images.
- **Features** are computed per pixel — colour channels, grayscale intensity, gradient and texture measures, and normalised position within the image.
- **Label** = whether human annotators marked that pixel as a segment boundary (`edge` vs `non-edge`).

That single feature table is then reused, unchanged, across every question — including the ones whose original prompt named a different dataset — so results stay directly comparable from Q1 to Q7.

The prepared data lives in [`dataset/`](dataset/); generated figures and metrics are written to [`results/`](results/).

> **Source:** [BSDS500 — UC Berkeley Computer Vision Group](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
>
> Reference: Arbeláez, Maire, Fowlkes and Malik, *Contour Detection and Hierarchical Image Segmentation*, IEEE TPAMI 33(5), 2011.

<!--
EDIT THIS BLOCK to match your actual feature table, then delete these comment markers:

| Column | Meaning |
|---|---|
| `R`, `G`, `B`, `gray` | Colour and intensity at the pixel |
| `grad_mag`, `grad_dir` | Gradient magnitude and direction |
| `laplacian`, `local_std` | Second-derivative and local texture measures |
| `x_norm`, `y_norm` | Normalised pixel position |
| `is_edge` | Target — 1 if annotated as a boundary, else 0 |

Rows: __ · Class balance: __ / __
-->

---

## What this repository demonstrates

- Applying **classical ML to a computer-vision dataset** — feature engineering from raw images rather than using a ready-made tabular file.
- Core classifiers implemented **from first principles** (k-NN, logistic regression with gradient descent), not just library calls.
- Correct use of **scikit-learn** for tree models, probabilistic models and evaluation.
- **Dimensionality reduction and visualisation** (PCA, t-SNE) on the engineered feature space.
- **Data quality work** — outlier detection and treatment using z-score and IQR.
- **Model evaluation beyond accuracy** — precision, recall, F1 and ROC-AUC across competing models.
- **Metaheuristic optimisation** — simulated annealing on a non-convex objective function.

---

## Module I — Overview of Machine Learning Techniques

| # | Lab Task | Technique | Dataset | Dataset Type | Code |
|:--:|---|---|---|---|---|
| 1 | k-Nearest Neighbours classifier implemented from scratch | Instance-based / distance-based classification | BSDS500 pixel features | Image-derived tabular · binary | [`Q1_KNN.ipynb`](notebook/Q1_KNN.ipynb) |
| 2 | Decision Tree classifier built with scikit-learn, with tree visualisation | Tree-based classification | BSDS500 pixel features | Image-derived tabular · binary | [`Q2_DECISION_TREE.ipynb`](notebook/Q2_DECISION_TREE.ipynb) |
| 3 | Gaussian Naive Bayes classification | Probabilistic classification | BSDS500 pixel features | Image-derived tabular · binary | [`Q3_Gaussian_Naive_Bayes.ipynb`](notebook/Q3_Gaussian_Naive_Bayes.ipynb) |
| 4 | Logistic Regression implemented from scratch using gradient descent | Linear model · binary classification | BSDS500 pixel features | Image-derived tabular · binary | [`Q4_logistic_regression.ipynb`](notebook/Q4_logistic_regression.ipynb) |
| 5 | PCA and t-SNE to reduce the feature space to 2D and visualise clusters | Unsupervised dimensionality reduction | BSDS500 pixel features | Image-derived tabular · high-dimensional | [`Q5_PCA_tSNE.py`](notebook/Q5_PCA_tSNE.py) |
| 6 | Detection and treatment of outliers using z-score and IQR | Data preprocessing / data quality | BSDS500 pixel features | Image-derived tabular · continuous | [`Q6_outlier_detection.ipynb`](notebook/Q6_outlier_detection.ipynb) |
| 7 | Accuracy, precision, recall, F1-score and ROC-AUC compared for Logistic Regression vs SVM | Model evaluation & comparison | BSDS500 pixel features | Image-derived tabular · binary | [`Q7_ROC_AUC.ipynb`](notebook/Q7_ROC_AUC.ipynb) |

**On the lab sheet's named datasets.** Q1 and Q2 specify Iris, Q3 specifies a spam/ham corpus and Q5 specifies MNIST. Per the lab instruction to work on one assigned dataset throughout, the BSDS500-derived feature table replaces all of them. The algorithm being demonstrated is unchanged — only the input differs.

## Module II — Metaheuristic Optimisation

| # | Lab Task | Technique | Dataset | Dataset Type | Code |
|:--:|---|---|---|---|---|
| 8 | Simulated Annealing to minimise f(x) = x² + 10·sin(x) | Probabilistic global optimisation | None — analytic objective function | Synthetic · 1-D function | [`Q8_SA_Minimization.ipynb`](notebook/Q8_SA_Minimization%20%282%29.ipynb) |
| 9.1 | Genetic Algorithm (GA) — Part 1: maximise a fitness function | Evolutionary / population-based optimisation | None — synthetic search space | Synthetic · optimization | [`Q9_1_Genetic_Algorithm_Maximize.ipynb`](notebook/Q9_1_Genetic_Algorithm_Maximize.ipynb) |
| 9.2 | Genetic Algorithm (GA) — Part 2: minimise a fitness function (e.g. Traveling Salesman tour distance) | Evolutionary / population-based optimisation | None — synthetic search space | Synthetic · optimization | [`Q9_2_Genetic_Algorithm_Minimize.ipynb`](notebook/Q9_2_Genetic_Algorithm_Minimize.ipynb)  |

> 
---

## Repository structure

```
SML-II-LAB-CODES/
├── dataset/                            # BSDS500-derived feature table
├── notebook/
│   ├── Q1_KNN.ipynb                    # Q1 — k-NN from scratch
│   ├── Q2_DECISION_TREE.ipynb          # Q2 — Decision Tree (scikit-learn) + visualisation
│   ├── Q3_Gaussian_Naive_Bayes.ipynb   # Q3 — Gaussian Naive Bayes
│   ├── Q4_logistic_regression.ipynb    # Q4 — Logistic Regression from scratch
│   ├── Q5_PCA_tSNE.py                  # Q5 — PCA & t-SNE to 2D
│   ├── Q6_outlier_detection.ipynb      # Q6 — z-score & IQR outlier handling
│   ├── Q7_ROC_AUC.ipynb                # Q7 — LR vs SVM, full metric comparison
│   ├── Q8_SA_Minimization.ipynb        # Q8 — Simulated Annealing
│   ├── Q9_1_Genetic_Algorithm_Maximize.ipynb  # Q9 Part 1 — GA, maximisation
│   └── Q9_2_Genetic_Algorithm_Minimize.ipynb  # Q9 Part 2 — GA, minimisation
├── results/                            # Generated figures and metrics
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech stack

| Purpose | Libraries |
|---|---|
| Numerical computing | NumPy |
| Data handling | pandas |
| Machine learning | scikit-learn |
| Image / signal processing | SciPy, OpenCV *(as used for feature extraction)* |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Getting started

Clone the repository:

```bash
git clone https://github.com/aasthariya27/SML-II-LAB-CODES.git
cd SML-II-LAB-CODES
```

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebooks:

```bash
jupyter notebook notebook/
```

Or run the standalone script:

```bash
python notebook/Q5_PCA_tSNE.py
```

To rebuild the feature table from the original images, download BSDS500 from the [Berkeley Computer Vision Group](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).

---

## Roadmap

Remaining lab tasks from Module II, to be added as they are completed:

- [ ] Q10 — Particle Swarm Optimisation on the Rosenbrock function
- [ ] Q11 — L1, L2 and Elastic Net regularisation on a regression dataset
- [ ] Q12 — Kernel PCA on a non-linear dataset
- [ ] Q13 — Ant Colony Optimisation for shortest-path search
- [ ] Q14 — Gram matrix computation and visualisation across kernels

---

## Author

**Aastha** — Integrated M.Sc. in Quantitative Economics and Data Science, BIT Mesra

[![GitHub](https://img.shields.io/badge/GitHub-aasthariya27-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/aasthariya27)

<!-- Optional: paste your LinkedIn URL between the brackets below and delete these comment markers to show a LinkedIn badge.
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](PASTE_YOUR_LINKEDIN_URL_HERE)
-->

---

## Note on academic use

This repository documents coursework submitted for ED408. It is shared for reference and learning. Please do not submit this work as your own.

BSDS500 is the property of the UC Berkeley Computer Vision Group and is used here for academic purposes only.


Every question in this repository is answered on the **same** dataset, derived from **BSDS500 — the Berkeley Segmentation Data Set and Benchmarks 500**.

BSDS500 is a computer-vision benchmark: 500 natural images, each hand-annotated by several human subjects who marked object boundaries, split into disjoint train / validation / test sets. It ships as JPEG images plus ground-truth annotation files — **not** as a tabular dataset.

**Why that matters.** The lab tasks assume tabular input (Iris, MNIST, a spam/ham corpus). BSDS500 has to be converted before any classifier can touch it. The conversion turns image segmentation into a genuine **binary classification problem**:

- **One row = one pixel**, sampled from the BSDS500 images.
- **Features** are computed per pixel — colour channels, grayscale intensity, gradient and texture measures, and normalised position within the image.
- **Label** = whether human annotators marked that pixel as a segment boundary (`edge` vs `non-edge`).

That single feature table is then reused, unchanged, across every question — including the ones whose original prompt named a different dataset — so results stay directly comparable from Q1 to Q7.

The prepared data lives in [`dataset/`](dataset/); generated figures and metrics are written to [`results/`](results/).

> **Source:** [BSDS500 — UC Berkeley Computer Vision Group](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
>
> Reference: Arbeláez, Maire, Fowlkes and Malik, *Contour Detection and Hierarchical Image Segmentation*, IEEE TPAMI 33(5), 2011.

<!--
EDIT THIS BLOCK to match your actual feature table, then delete these comment markers:

| Column | Meaning |
|---|---|
| `R`, `G`, `B`, `gray` | Colour and intensity at the pixel |
| `grad_mag`, `grad_dir` | Gradient magnitude and direction |
| `laplacian`, `local_std` | Second-derivative and local texture measures |
| `x_norm`, `y_norm` | Normalised pixel position |
| `is_edge` | Target — 1 if annotated as a boundary, else 0 |

Rows: __ · Class balance: __ / __
-->

---

## What this repository demonstrates

- Applying **classical ML to a computer-vision dataset** — feature engineering from raw images rather than using a ready-made tabular file.
- Core classifiers implemented **from first principles** (k-NN, logistic regression with gradient descent), not just library calls.
- Correct use of **scikit-learn** for tree models, probabilistic models and evaluation.
- **Dimensionality reduction and visualisation** (PCA, t-SNE) on the engineered feature space.
- **Data quality work** — outlier detection and treatment using z-score and IQR.
- **Model evaluation beyond accuracy** — precision, recall, F1 and ROC-AUC across competing models.
- **Metaheuristic optimisation** — simulated annealing on a non-convex objective function.

---

## Module I — Overview of Machine Learning Techniques

| # | Lab Task | Technique | Dataset | Dataset Type | Code |
|:--:|---|---|---|---|---|
| 1 | k-Nearest Neighbours classifier implemented from scratch | Instance-based / distance-based classification | BSDS500 pixel features | Image-derived tabular · binary | [`Q1_KNN.ipynb`](notebook/Q1_KNN.ipynb) |
| 2 | Decision Tree classifier built with scikit-learn, with tree visualisation | Tree-based classification | BSDS500 pixel features | Image-derived tabular · binary | [`Q2_DECISION_TREE.ipynb`](notebook/Q2_DECISION_TREE.ipynb) |
| 3 | Gaussian Naive Bayes classification | Probabilistic classification | BSDS500 pixel features | Image-derived tabular · binary | [`Q3_Gaussian_Naive_Bayes.ipynb`](notebook/Q3_Gaussian_Naive_Bayes.ipynb) |
| 4 | Logistic Regression implemented from scratch using gradient descent | Linear model · binary classification | BSDS500 pixel features | Image-derived tabular · binary | [`Q4_logistic_regression.ipynb`](notebook/Q4_logistic_regression.ipynb) |
| 5 | PCA and t-SNE to reduce the feature space to 2D and visualise clusters | Unsupervised dimensionality reduction | BSDS500 pixel features | Image-derived tabular · high-dimensional | [`Q5_PCA_tSNE.py`](notebook/Q5_PCA_tSNE.py) |
| 6 | Detection and treatment of outliers using z-score and IQR | Data preprocessing / data quality | BSDS500 pixel features | Image-derived tabular · continuous | [`Q6_outlier_detection.ipynb`](notebook/Q6_outlier_detection.ipynb) |
| 7 | Accuracy, precision, recall, F1-score and ROC-AUC compared for Logistic Regression vs SVM | Model evaluation & comparison | BSDS500 pixel features | Image-derived tabular · binary | [`Q7_ROC_AUC.ipynb`](notebook/Q7_ROC_AUC.ipynb) |

**On the lab sheet's named datasets.** Q1 and Q2 specify Iris, Q3 specifies a spam/ham corpus and Q5 specifies MNIST. Per the lab instruction to work on one assigned dataset throughout, the BSDS500-derived feature table replaces all of them. The algorithm being demonstrated is unchanged — only the input differs.

## Module II — Metaheuristic Optimisation

| # | Lab Task | Technique | Dataset | Dataset Type | Code |
|:--:|---|---|---|---|---|
| 8 | Simulated Annealing to minimise f(x) = x² + 10·sin(x) | Probabilistic global optimisation | None — analytic objective function | Synthetic · 1-D function | [`Q8_SA_Minimization.ipynb`](notebook/Q8_SA_Minimization%20%282%29.ipynb) |

---

## Repository structure

```
SML-II-LAB-CODES/
├── dataset/                            # BSDS500-derived feature table
├── notebook/
│   ├── Q1_KNN.ipynb                    # Q1 — k-NN from scratch
│   ├── Q2_DECISION_TREE.ipynb          # Q2 — Decision Tree (scikit-learn) + visualisation
│   ├── Q3_Gaussian_Naive_Bayes.ipynb   # Q3 — Gaussian Naive Bayes
│   ├── Q4_logistic_regression.ipynb    # Q4 — Logistic Regression from scratch
│   ├── Q5_PCA_tSNE.py                  # Q5 — PCA & t-SNE to 2D
│   ├── Q6_outlier_detection.ipynb      # Q6 — z-score & IQR outlier handling
│   ├── Q7_ROC_AUC.ipynb                # Q7 — LR vs SVM, full metric comparison
│   └── Q8_SA_Minimization.ipynb        # Q8 — Simulated Annealing
├── results/                            # Generated figures and metrics
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech stack

| Purpose | Libraries |
|---|---|
| Numerical computing | NumPy |
| Data handling | pandas |
| Machine learning | scikit-learn |
| Image / signal processing | SciPy, OpenCV *(as used for feature extraction)* |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Getting started

Clone the repository:

```bash
git clone https://github.com/aasthariya27/SML-II-LAB-CODES.git
cd SML-II-LAB-CODES
```

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the notebooks:

```bash
jupyter notebook notebook/
```

Or run the standalone script:

```bash
python notebook/Q5_PCA_tSNE.py
```

To rebuild the feature table from the original images, download BSDS500 from the [Berkeley Computer Vision Group](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).

---

## Roadmap

Remaining lab tasks from Module II, to be added as they are completed:

- [ ] Q9 — Genetic Algorithm for the Traveling Salesman Problem
- [ ] Q10 — Particle Swarm Optimisation on the Rosenbrock function
- [ ] Q11 — L1, L2 and Elastic Net regularisation on a regression dataset
- [ ] Q12 — Kernel PCA on a non-linear dataset
- [ ] Q13 — Ant Colony Optimisation for shortest-path search
- [ ] Q14 — Gram matrix computation and visualisation across kernels

---

## Author

**Aastha** — Integrated M.Sc. in Quantitative Economics and Data Science, BIT Mesra

[![GitHub](https://img.shields.io/badge/GitHub-aasthariya27-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/aasthariya27)

<!-- Optional: paste your LinkedIn URL between the brackets below and delete these comment markers to show a LinkedIn badge.
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](PASTE_YOUR_LINKEDIN_URL_HERE)
-->

---

## Note on academic use

This repository documents coursework submitted for ED408. It is shared for reference and learning. Please do not submit this work as your own.

BSDS500 is the property of the UC Berkeley Computer Vision Group and is used here for academic purposes only.
