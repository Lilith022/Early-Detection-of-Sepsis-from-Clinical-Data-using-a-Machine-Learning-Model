# Early Detection of Sepsis from Clinical Data using a Machine Learning Model

The aim of this study is to develop a model for early sepsis detection by implementing a rigorous, clinically-grounded data preprocessing pipeline. This approach prioritizes careful patient selection, advanced handling of missing values, and temporal alignment to create an analysis-ready dataset. This foundation enables the subsequent selection of a machine learning model using a broad set of metrics, including the Utility Score.

The preprocessed, imputed, and standardized dataset is publicly available at the following [Drive link](https://drive.google.com/file/d/1L7ijycVetKs71aC7XezKcbnYG0s0QR9g/view?usp=sharing).

--- 

## Methodology
The methodology follows a **systematic and clinically-grounded pipeline**:
1. **Data Preprocessing**  
   - Patient-level temporal segmentation (21-hour observation windows).  
   - Sliding windows of 6 hours with 1-hour stride to capture temporal dynamics.  
   - Feature engineering (vital signs, labs, demographics, derived features).  

2. **Imputation**  
   - Multivariate imputation using **LightGBM**.  
   - Hyperparameter tuning with **HalvingRandomSearchCV**.  
   - Z-score normalization of variables.  

3. **Model Development**
   - Gradient Boosting family: **LightGBM, Gradient Boosting, CatBoost, XGBoost**.  
   - Hyperparameter optimization via **Optuna**.  
   - Threshold optimization.  

4. **Evaluation**  
   - Metrics: Accuracy, F1 Score, Precision, ROC-AUC, Sensitivity, Specificity, **Utiity Score**.  
   - 5-fold Stratified Group Cross-Validation.  

5. **Interpretability**  
   - **SHAP values** for feature importance (global and patient-level).  
   - Visualization of key predictors (ICULOS, BUN, respiratory patterns).  

---

## 📂 Repository Structure


```bash
├── Pre-processing/
│   └── data_preprocessing.py
│
├── Models/
│   └── Total_models.py
│
├── Optimized_Models/
│   ├── CatBoost.py
│   ├── Gradient_Boosting.py
│   ├── LightGBM.py
│   └── XGBoost.py
└── Requirements.txt
```
---

## Files and Directories

### Pre-processing
- **`data_preprocessing.py`**: Full preprocessing pipeline (forward-fill imputation, 21h temporal alignment, clinical feature engineering, 6h sliding windows, and variable filtering).  

### Models
- **`Total_models.py`**: Comparative analysis of 11 ML models with cross-validation, multiple metrics, and statistical confidence intervals.  

### Optimized_Models
- **`CatBoost.py`**: Final CatBoost pipeline with utility metrics, Optuna optimization, SHAP interpretability, and evaluation.  
- **`Gradient_Boosting.py`**: Final Gradient Boosting pipeline with cross-validation, hyperparameter tuning, SHAP, and metrics.  
- **`LightGBM.py`**: Final LightGBM pipeline including optimization, SHAP explanations, and evaluation.  
- **`XGBoost.py`**: Final XGBoost pipeline with patient-level utility, Optuna, SHAP, and full metrics.
### `Requirements.txt`: All required libraries

## Requirements & Reproducibility

To reproduce the experiments, you need:

- **Python 3.9+** installed.  
- All required libraries listed in `Requirements.txt`.  

### 🔧 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/sepsis-prediction.git](https://github.com/Lilith022/Early-Detection-of-Sepsis-from-Clinical-Data.git
cd sepsis-prediction
pip install -r Requirements.txt
```

## Authors

- Josman Rico
- Deisy Torres
- Camilo Santos
- Harold H. Rodríguez
- Carlos A. Fajardo

Department of Electrical, Electronics and Telecommunications – Universidad Industrial de Santander – Bucaramanga, Colombia
