# Intelligent Predictive Maintenance in IIoT
**Solving the Accuracy Paradox in Highly Imbalanced Industrial Datasets**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-green.svg)
![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-Imbalance_Handling-red.svg)

## The Problem: The Accuracy Paradox
In industrial manufacturing, machines operate normally the vast majority of the time. In the **AI4I 2020 Predictive Maintenance Dataset** used for this project, the baseline failure rate is only **3.39%**. 

Training a standard classification model on this data results in the "Accuracy Paradox": a model that blindly guesses "Normal" every time will achieve 96.6% accuracy, yet fail to catch a single catastrophic machine breakdown. This project aims to engineer a mathematically rigorous ML pipeline that prioritizes **Recall** to predict actual failures before they happen.

## Methodology & Engineering
1. **Physics-Driven EDA:** Rather than relying on basic histograms, I utilized Kernel Density Estimation (KDE) to map the physical trigger points of failure, proving that breakdowns occur at the extremes of Torque and Rotational Speed.
   <img width="830" height="307" alt="Screenshot 2026-04-22 124042" src="https://github.com/user-attachments/assets/54e02270-0ce2-4e17-9a50-e2ebc38aecdf" />


2. **Multicollinearity Elimination:** Calculated a Pearson correlation matrix to identify redundant sensors. Found a 0.88 correlation between Air and Process temperatures and dropped the latter to reduce algorithmic noise.
   <img width="503" height="347" alt="Screenshot 2026-04-22 124103" src="https://github.com/user-attachments/assets/fb28f1dc-b38f-4b16-9437-11de05d189ec" />


3. **Algorithmic Rebalancing (SMOTE):** Utilized Synthetic Minority Over-sampling Technique (SMOTE) to mathematically synthesize failure data, teaching the AI the patterns of a breakdown without duplicating existing rows.

4. **Explainable AI:** Extracted feature importances from the Random Forest classifier to provide actionable business intelligence on which sensors indicate the highest risk.
   <img width="460" height="304" alt="Screenshot 2026-04-22 124000" src="https://github.com/user-attachments/assets/72bc742b-acf4-46b8-8270-921b5991b423" />


## Model Evaluation & Business Impact
Standard accuracy is a misleading metric in predictive maintenance. The model was evaluated based on the industrial cost of False Positives (wasted inspection time) versus False Negatives (destroyed machinery).

* **Prioritized Recall (76%):** The model successfully catches the vast majority of true machine failures.
* **AUC-ROC (0.95):** Demonstrates exceptional discriminative ability across different thresholds.

<img width="384" height="313" alt="Screenshot 2026-04-22 124124" src="https://github.com/user-attachments/assets/8ce2ccb3-a70f-4a8e-8d3d-44dd1bffb373" />
<img width="823" height="311" alt="Screenshot 2026-04-22 123528" src="https://github.com/user-attachments/assets/9e027ed8-c6f6-4223-8394-4e2c870fc739" />


## How to Run Locally
1. Clone the repository: `git clone https://github.com/yourusername/Predictive-Maintenance-IIoT.git`
2. Install dependencies: `pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn`
3. Run the Jupyter Notebook: `jupyter notebook Predictive_Maintenance.ipynb`

