# Cyberbullying Detection in Social Media

This repository contains all the necessary components to classify, analyze and predict instances of cyberbullying using Machine Learning and Deep Learning.

## 🚀 Project Overview

This project aims to detect and classify different types of cyberbullying in social media text. We have used various machine learning models including SVM, Random Forest, Naive Bayes and Boosting models like XGBoost, LightGBM and CatBoost and advanced deep learning models (BERT) to identify patterns and predict harmful content effectively. The project also includes a summarization process to extract key insights from the detected cyberbullying cases.

---

## 📂 Project Structure

### **Artifacts**
- `CatBoost.pkl` – Trained CatBoost model.  
- `CyberbullyingClassifier.keras` – Trained Keras model for cyberbullying classification.  
- `CyberbullyingSummarisation.py` – Script for summarizing cyberbullying cases.  
- `CyberbullyingSummary.txt` – Text file containing summarized cyberbullying instances.  
- `Flowchart.txt` – Explanation of the project workflow.  
- `helper_prabowo_ml.py` – Helper functions used in machine learning training and evaluation.  
- `LabelEncoder.pkl` – Pre-trained label encoder for encoding class labels.  
- `LightGBM.pkl` – Trained LightGBM model.  
- `NaiveBayes.pkl` – Trained Naive Bayes model.  
- `RandomForest.pkl` – Trained Random Forest model.  
- `SVM-OvO.pkl` – Trained Support Vector Machine (One-vs-One) model.  
- `SVM-OvR.pkl` – Trained Support Vector Machine (One-vs-Rest) model.  
- `TFIDFVectorizer.pkl` – Pre-trained TF-IDF vectorizer.  
- `TrainingHistory.pkl` – Training history of deep learning models.  
- `x_test.pkl`, `X_train.pkl` – Processed training and test sets.  
- `XGBoost.pkl` – Trained XGBoost model.  

---

### **Dataset**
- `OriginalDataset.csv` – Raw dataset containing social media text and labels.  
- `PreprocessedDataset.csv` – Cleaned and processed dataset ready for model training.  

---

### **Notebooks**
- `Analysis.ipynb` – Exploratory Data Analysis (EDA) and model performance analysis.  
- `CyberBullying.ipynb` – Main notebook for model training and evaluation.  
- `Fine-tuning BERT.ipynb` – Notebook for fine-tuning BERT-based models.  
- `Summarisation.ipynb` – Notebook for summarizing cyberbullying cases using NLP.  

---

### **Templates**
- `Barplot.png` – Barplot showing class distribution.  
- `Boxplot.png` – Boxplot showing data distribution.  
- `ClassDistribution.png` – Distribution of cyberbullying categories.  
- `ConfusionMatrix.png` – Confusion matrix showing model performance.  
- `Flowchart.png` – Visual representation of the project workflow.  
- `Histogram.png` – Histogram showing text length distribution.  
- `ModelComparison.png` – Model performance comparison.  
- `ROC-AUC.png` – ROC-AUC curves of different models.  
- `Transformer.png` – Architecture of the Transformer model.  
- `TransformerAccuracy.png` – Accuracy trend of the Transformer model.  
- `TransformerLoss.png` – Loss trend of the Transformer model.  
- `WordCloud.png` – Word cloud showing common terms in the dataset.  

---

### **Configuration Files**
- `.gitignore` – Files and folders to be ignored by Git.  
- `.env` – Environment configuration file.  
- `.gitattributes` – File to control Git behavior.  
- `LICENSE` – Project licensing information.  
- `README.md` – Project documentation.  
- `requirements.txt` – List of Python packages required to run the project.  

---

## 🛠️ Getting Started

### **Prerequisites**
Ensure you have Python installed. Clone this repository and install the required packages:

```bash
git clone https://github.com/your-repo/Cyberbullying-Detection.git
cd Cyberbullying-Detection
pip install -r requirements.txt
