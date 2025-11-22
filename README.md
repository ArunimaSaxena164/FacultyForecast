# 🎓 FacultyForecast – Faculty Retention Prediction

FacultyForecast is a machine learning project that predicts whether a faculty member will be **retained** or **leave** an academic institution.  
The goal is to help institutions identify early attrition risks and make data-driven decisions.

---

## 📌 1. Problem Overview

Faculty attrition affects:

- 🏫 Department stability  
- 💸 Hiring and training costs  
- 🎓 Student learning quality  
- 🔬 Research continuity  

This project predicts the `left_institution` label (0 = stayed, 1 = left) using faculty characteristics, workload, performance, and institutional factors.

---

## 📊 2. Dataset Summary

**Total rows:** 15,200  
**Total columns:** 15  

### Feature Breakdown

**Categorical Features**
- academic_rank  
- tenure_status  
- institution_type  

**Numeric Features**
- years_at_institution  
- base_salary  
- teaching_load  
- research_funding  
- department_size  
- admin_support  
- work_life_balance  
- promotion_opportunities  
- publications_last_3_years  
- student_evaluation_avg  

**Target**
- left_institution  

**Class Distribution**
- Stayed: ~76%  
- Left: ~24%

---

## 📁 3. Project Structure

FACULTYFORECAST/
│
├── data/
│ └── Faculty_Atrrition_Dataset.csv
│
├── models/
│ ├── knn_svd.pkl
│ ├── log_reg.pkl
│ ├── random_forest.pkl
│ └── svm_rbf.pkl
│
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_LogisticRegression.ipynb
│ ├── 04_SVM.ipynb
│ ├── 05_knn.ipynb
│ ├── 06_randomForest.ipynb
│ └── 07_Ensemble_Models.ipynb
│
├── requirements.txt
└── README.md

---

## ⚙️ 4. Preprocessing Pipeline

- 🧹 Missing value handling  
- 🔡 One-Hot Encoding  
- 📏 Standard Scaling  
- 🚨 Outlier handling  
- ✂️ Train-Test Split (80/20)

---

## 🤖 5. Model Performance Summary

### Logistic Regression (Tuned)
- Accuracy: **0.796**
- AUC: **0.886**

### SVM (RBF Kernel)
- Accuracy: **0.874**
- AUC: **0.896**

### Random Forest (Tuned)
- Accuracy: **0.959**
- AUC: **0.912**
- ⭐ Best standalone model

### KNN (K = 11)
- Accuracy: **0.853**
- AUC: **0.860**

### KNN with SVD
- Accuracy: **0.820**
- AUC: **0.847**

---

## 🧠 6. Ensemble Models

### Soft Voting Ensemble
- Accuracy: **0.955**
- AUC: **0.910**

### Stacking Ensemble (Best Overall)
- Accuracy: **0.957**
- AUC: **0.909**
- 🏆 Most balanced model

---

## ▶️ 7. How to Run

### Install dependencies
pip install -r requirements.txt 
### Run notebooks in order
1. 01_EDA.ipynb  
2. 02_preprocessing.ipynb  
3. Individual model notebooks  
4. 07_Ensemble_Models.ipynb  

### Using saved models
Models are stored in the `models/` folder as `.pkl` files.

---

## 💡 8. Key Insights

- Work-life balance, admin support, and research funding strongly influence attrition.  
- Random Forest and Ensembles outperform linear models.  
- Class imbalance affects recall for class 1; ensembles handle this well.

---

