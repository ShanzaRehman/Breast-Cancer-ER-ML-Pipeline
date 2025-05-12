import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.decomposition import PCA

# 1. Load gene expression data
expr = pd.read_csv("data_mrna_illumina_microarray.txt", sep="\t")
if 'Entrez_Gene_Id' in expr.columns:
    expr = expr.drop(columns=['Entrez_Gene_Id'])

expr = expr.set_index('Hugo_Symbol').T
expr.index.name = 'SAMPLE_ID'
expr.reset_index(inplace=True)

# 2. Load clinical data
clinical = pd.read_csv("data_clinical_sample.txt", sep="\t", comment='#')

# 3. Merge datasets using ER_STATUS
merged = pd.merge(expr, clinical[['SAMPLE_ID', 'ER_STATUS']], on='SAMPLE_ID')
merged = merged.dropna()

# 4. Feature matrix and target vector
X = merged.drop(columns=['SAMPLE_ID', 'ER_STATUS'])
y = merged['ER_STATUS']

# 5. Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 6. Normalize gene expression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Feature selection using mutual information
selector = SelectKBest(score_func=mutual_info_classif, k=100)
X_selected = selector.fit_transform(X_scaled, y_encoded)

# 8. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 9. Initialize and train classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='linear', probability=True, random_state=42)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# 10. Evaluate classifiers
print("Random Forest:\n", classification_report(y_test, rf.predict(X_test)))
print("SVM:\n", classification_report(y_test, svm.predict(X_test)))

# 11. Confusion matrix for Random Forest
sns.heatmap(confusion_matrix(y_test, rf.predict(X_test)), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 12. Confusion matrix for SVM
sns.heatmap(confusion_matrix(y_test, svm.predict(X_test)), annot=True, fmt='d', cmap='Oranges')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 13. PCA visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_selected)
pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
pca_df['ER_STATUS'] = y.values

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='ER_STATUS', data=pca_df, palette='Set1', s=80, alpha=0.7)
plt.title('PCA of Gene Expression Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='ER Status')
plt.show()

# 14. ROC AUC and Curve for Random Forest
y_probs_rf = rf.predict_proba(X_test)
roc_auc_rf = roc_auc_score(y_test, y_probs_rf[:, 1])
print("ROC AUC (Random Forest):", roc_auc_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 15. ROC AUC and Curve for SVM
y_probs_svm = svm.predict_proba(X_test)
roc_auc_svm = roc_auc_score(y_test, y_probs_svm[:, 1])
print("ROC AUC (SVM):", roc_auc_svm)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_probs_svm[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
