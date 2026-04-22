# ⚙️ Intelligent Predictive Maintenance in IIoT
**Solving the Accuracy Paradox in Highly Imbalanced Industrial Datasets**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-green.svg)
![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-Imbalance_Handling-red.svg)

## 📌 The Problem: The Accuracy Paradox
In industrial manufacturing, machines operate normally the vast majority of the time. In the **AI4I 2020 Predictive Maintenance Dataset** used for this project, the baseline failure rate is only **3.39%**. 

Training a standard classification model on this data results in the "Accuracy Paradox": a model that blindly guesses "Normal" every time will achieve 96.6% accuracy, yet fail to catch a single catastrophic machine breakdown. This project aims to engineer a mathematically rigorous ML pipeline that prioritizes **Recall** to predict actual failures before they happen.

## 🔬 Methodology & Engineering
1. **Physics-Driven EDA:** Rather than relying on basic histograms, I utilized Kernel Density Estimation (KDE) to map the physical trigger points of failure, proving that breakdowns occur at the extremes of Torque and Rotational Speed.
   *(Insert Image 2: KDE Plots here)*

2. **Multicollinearity Elimination:** Calculated a Pearson correlation matrix to identify redundant sensors. Found a 0.88 correlation between Air and Process temperatures and dropped the latter to reduce algorithmic noise.
   *(Insert Image 3: Heatmap here)*

3. **Algorithmic Rebalancing (SMOTE):** Utilized Synthetic Minority Over-sampling Technique (SMOTE) to mathematically synthesize failure data, teaching the AI the patterns of a breakdown without duplicating existing rows.

4. **Explainable AI:** Extracted feature importances from the Random Forest classifier to provide actionable business intelligence on which sensors indicate the highest risk.
   *(Insert Image 1: Feature Importance Bar Chart here)*

## 📊 Model Evaluation & Business Impact
Standard accuracy is a misleading metric in predictive maintenance. The model was evaluated based on the industrial cost of False Positives (wasted inspection time) versus False Negatives (destroyed machinery).

* **Prioritized Recall (76%):** The model successfully catches the vast majority of true machine failures.
* **AUC-ROC (0.95):** Demonstrates exceptional discriminative ability across different thresholds.

*(Insert Image 4 & 5: Confusion Matrix and ROC/PR Curves here)*

## 🚀 How to Run Locally
1. Clone the repository: `git clone https://github.com/yourusername/Predictive-Maintenance-IIoT.git`
2. Install dependencies: `pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn`
3. Run the Jupyter Notebook: `jupyter notebook Predictive_Maintenance.ipynb`

---
*Developed for INT375 - Continuous Assessment*
