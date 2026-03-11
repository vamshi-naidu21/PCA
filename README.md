# Dimensionality Reduction with PCA for Machine Learning 📊

This project demonstrates how to use **Principal Component Analysis (PCA)** to reduce feature dimensionality and improve machine learning model performance. PCA helps transform high-dimensional data into a smaller set of informative components while preserving most of the variance.

## Project Objective

* Apply PCA to reduce the number of features in a dataset
* Train a machine learning model on the reduced feature space
* Evaluate model performance before and after dimensionality reduction

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

## Project Workflow

1. Data loading and preprocessing
2. Feature scaling using StandardScaler
3. Apply PCA for dimensionality reduction
4. Train machine learning model on PCA-transformed data
5. Generate predictions
6. Evaluate model performance using classification metrics

## Example Code

```python id="pca1"
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train model
model = LogisticRegression()
model.fit(X_pca, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
```

## Model Evaluation

The model is evaluated using common classification metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## Results

PCA reduces feature dimensions while maintaining most of the important information, helping simplify models and sometimes improving performance.

## Future Improvements

* Hyperparameter tuning
* Visualization of PCA components
* Testing with multiple ML algorithms
* Model deployment using API

---

Machine Learning Project – Python Implementation
