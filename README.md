# Machine Learning - Company bankruptcy prediction

## 📌 Problem Statement
The objective of this project is to build and evaluate multiple machine learning models to predict **company bankruptcy** based on financial indicators.  
A key challenge addressed in this project is the **severe class imbalance**, where bankrupt companies (minority class) represent only ~3% of the dataset. Techniques such as **SMOTE (Synthetic Minority Oversampling Technique)** are applied to improve predictive performance, especially for identifying high-risk companies.

---

## 📑 Table of Contents
- Problem Statement  
- Libraries and Functions  
- Dataset Import  
- Exploratory Data Analysis (EDA)  
  - Missing Data Analysis  
  - Correlation Analysis  
  - Target Class Imbalance  
- Model Training and Evaluation  
  - Logistic Regression  
  - K-Nearest Neighbors  
  - Decision Tree  
  - Neural Networks (MLP)  
  - Random Forest  
  - Gradient Boosting  
- Comparison of Results  
- Conclusion  
- References  

---

## 🛠️ Libraries and Functions
The following Python libraries were used:

```python
pandas
numpy
scikit-learn
imbalanced-learn
google.colab
os
shutil

