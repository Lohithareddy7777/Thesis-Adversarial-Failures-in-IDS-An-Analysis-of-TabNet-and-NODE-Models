# 📊 Machine Learning for Tabular Data  

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Overview  
This repository presents a complete implementation of a bachelor’s thesis focused on improving tabular data modeling using feature selection, dimensionality reduction, and deep learning architectures.  

The work investigates how preprocessing techniques interact with models like **TabNet** and **NODE** to enhance performance and interpretability.

---

## 🎯 Problem Statement  
Traditional machine learning models often struggle to balance performance and interpretability on tabular data. This project explores whether combining:  
- Feature selection (**Lasso**)  
- Dimensionality reduction (**PCA**)  
- Deep learning models (**TabNet, NODE**)  

can lead to better generalization and efficiency.

---

## ⚙️ Methodology  
A structured pipeline is followed throughout the experiments:  

- Data preprocessing and cleaning  
- Feature selection using Lasso  
- Dimensionality reduction using PCA  
- Model training (TabNet, NODE)  
- Evaluation using standard metrics  

---

## 🧪 Experiments  
- Conducted on benchmark tabular datasets  
- Compared multiple configurations:
  - With/without Lasso  
  - With/without PCA  
  - Across different models  
- Focus on performance, stability, and feature impact  

---

## 📈 Results & Analysis  
- Feature selection reduces noise and improves model focus  
- PCA helps in dimensionality reduction but may affect interpretability  
- Deep learning models show competitive performance when combined with proper preprocessing  
- Trade-offs identified between accuracy, complexity, and interpretability  

---

## 📂 Repository Structure  
├── data/ # Dataset files
├── preprocessing/ # Lasso & PCA scripts
├── models/ # TabNet, NODE implementations
├── experiments/ # Training scripts
├── results/ # Outputs & metrics
└── README.md
