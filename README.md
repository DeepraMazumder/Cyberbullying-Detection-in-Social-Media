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
- `X_test.pkl`, `X_train.pkl` – Processed training and test sets.   
- `CyberbullyingClassifier.keras` – Fine-tuned BERT model.
- `TrainingHistory.pkl` – Training history of the fine-tuned transformer model.
- `CyberbullyingSummarisation.py` – Module for analyzing and summarizing cyberbullying cases.  
- `CyberbullyingSummary.txt` – Sample output of harmful content detection and suggestions.  

### **Dataset**
- `OriginalDataset.csv` – Raw dataset containing social media text and labels. [Available on Kaggle](https://www.kaggle.com/datasets/momo12341234/cyberbully-detection-dataset)
- `PreprocessedDataset.csv` – Cleaned and processed dataset ready for model training.  

### **Notebooks**
- `CyberBullying.ipynb` – Notebook for training and evaluating Machine Learning models.
- `Fine-tuning BERT.ipynb` – Notebook for fine-tuning BERT-based models. [Published on Kaggle](https://www.kaggle.com/code/deepramazumder/cyberbullying-classification-fine-tuning-bert)
- `Summarisation.ipynb` – Notebook for generating cyberbullying summaries using a Generative AI model. 
- `Analysis.ipynb` – Script to classify and analyze user input for cyberbullying using a Generative AI model.  

### **Templates**
- `Flowchart.png` – Visual representation of the project workflow.
- `ClassDistribution.png` – Class distribution visualization based on the dataset.  
- `Histogram.png` – Histogram showing the length of tweets distribution.  
- `Boxplot.png` – Boxplot showing tweet length distribution per class.  
- `WordCloud.png` – Word cloud showing the most common terms in the dataset.  
- `Barplot.png` – Barplot showing the most common words in each class.  
- `ModelComparison.png` – Model performance comparison in terms of accuracy.  
- `ConfusionMatrix.png` – Confusion matrix of the model with the highest accuracy.  
- `ROC-AUC.png` – ROC-AUC curve of the model with the highest accuracy.
- `Transformer.png` – Architecture of the Transformer model.  
- `TransformerAccuracy.png` – Accuracy graph of the Transformer model.  
- `TransformerLoss.png` – Loss graph of the Transformer model.  

### **Configuration Files**
- `.gitignore` – Files and folders to be ignored by Git.  
- `.gitattributes` – Configures Git Large File Storage (LFS) for managing large files in the repository.  
- `LICENSE` – Project licensing information.  
- `requirements.txt` – List of Python packages required to run the project.  

## 🛠️ Getting Started

### **Prerequisites**
Ensure you have Python installed. Clone this repository and install the required packages:

```bash
git clone https://github.com/your-repo/Cyberbullying-Detection.git
cd Cyberbullying-Detection
pip install -r requirements.txt

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
