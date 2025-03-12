# Cyberbullying Detection in Social Media

This repository contains all the necessary components to classify, analyze and predict instances of cyberbullying using Machine Learning and Deep Learning.

## 🚀 Project Overview

This project aims to detect and classify different types of cyberbullying in social media text. We have used various machine learning models including SVM, Random Forest, Naive Bayes and Boosting models like XGBoost, LightGBM and CatBoost and advanced deep learning models (BERT) to identify patterns and predict harmful content effectively. The project also includes a summarization process to extract key insights from the detected cyberbullying cases.

## 📂 Project Structure

### **Artifacts**
- `Flowchart.txt` – Explanation of the project workflow.
- `LabelEncoder.pkl` – Label encoder fitted on the training data for encoding class labels.  
- `TFIDFVectorizer.pkl` – TF-IDF vectorizer fitted on the training data for text transformation.
- `RandomForest.pkl` – Trained Random Forest model.  
- `NaiveBayes.pkl` – Trained Naive Bayes model.
- `SVM-OvO.pkl` – Trained Support Vector Machine (One-vs-One) model.  
- `SVM-OvR.pkl` – Trained Support Vector Machine (One-vs-Rest) model.  
- `XGBoost.pkl` – Trained XGBoost model.  
- `LightGBM.pkl` – Trained LightGBM model.  
- `CatBoost.pkl` – Trained CatBoost model.  

- `helper_prabowo_ml.py` – Helper functions used in transformer training and evaluation.  
- `CyberbullyingClassifier.keras` – Fine-tuned BERT model.
- `X_test.pkl`, `X_train.pkl` – Processed training and test sets.  
- `TrainingHistory.pkl` – Training history of the fine-tuned transformer model.
- `CyberbullyingSummarisation.py` – Script for summarizing cyberbullying cases.  
- `CyberbullyingSummary.txt` – Text file containing summarized cyberbullying instances.  

### **Dataset**
- `OriginalDataset.csv` – Raw dataset containing social media text and labels. [Source](https://www.kaggle.com/datasets/momo12341234/cyberbully-detection-dataset)
- `PreprocessedDataset.csv` – Cleaned and processed dataset ready for model training.  

### **Notebooks**
- `CyberBullying.ipynb` – Notebook for training and evaluating Machine Learning models.
- `Fine-tuning BERT.ipynb` – Notebook for fine-tuning BERT-based models (published on Kaggle: [Cyberbullying Classification - Fine-Tuning BERT](https://www.kaggle.com/code/deepramazumder/cyberbullying-classification-fine-tuning-bert)).
- `Summarisation.ipynb` – Notebook for summarizing cyberbullying cases using NLP.  
- `Analysis.ipynb` – Exploratory Data Analysis (EDA) and model performance analysis.  

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

### **Configuration Files**
- `.gitignore` – Files and folders to be ignored by Git.  
- `.gitattributes` – File to control Git behavior.  
- `LICENSE` – Project licensing information.  
- `requirements.txt` – List of Python packages required to run the project.  

## 🛠️ Getting Started

### **Prerequisites**
Ensure you have Python installed. Clone this repository and install the required packages:

```bash
git clone https://github.com/your-repo/Cyberbullying-Detection.git
cd Cyberbullying-Detection
pip install -r requirements.txt
