# Breast Cancer ER Status Classification Using Gene Expression Data and Machine Learning

This project uses supervised machine learning models to classify Estrogen Receptor (ER) status (positive or negative) in breast cancer patients using gene expression (mRNA) microarray data. We implemented feature selection, normalization, classification using Random Forest and Support Vector Machine (SVM), and model evaluation using PCA, confusion matrices, and ROC curves.


## 🧬 Dataset

- **Source:** cBioPortal / METABRIC Breast Cancer Dataset
- **Files Used:**
  - `data_mrna_illumina_microarray.txt` – Gene expression data
  - `data_clinical_sample.txt` – Clinical information including ER status

> *Note:* Raw data is not included due to size and privacy; download directly from [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric)


## 🛠️ Workflow

1. **Data Preprocessing**
   - Transposed gene expression matrix
   - Merged with clinical data on `SAMPLE_ID`
   - Removed missing ER status

2. **Feature Engineering**
   - Label encoding of ER status
   - Standardized gene expression using `StandardScaler`
   - Selected top 100 features using mutual information

3. **Model Building**
   - Trained both Random Forest and SVM (with linear kernel)
   - Performed 80/20 train-test split with stratification

4. **Evaluation Metrics**
   - Classification report: precision, recall, F1-score
   - Confusion matrix
   - ROC-AUC and ROC curves
   - PCA visualization for dimensionality reduction


## 📊 Results Summary

| Model         | Accuracy | Precision (ER+) | Recall (ER+) | ROC AUC |
|---------------|----------|------------------|---------------|----------|
| Random Forest | 97%      | 98%              | 99%           | ~0.996   |
| SVM           | 97%      | 97%              | 99%           | ~0.995   |

- **PCA** clearly separated ER+ and ER− groups, indicating strong underlying patterns.
- Both models showed high classification performance with slightly better generalization in SVM (fewer false positives).



