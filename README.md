  Project Overview
  
This project implements an AI-based phishing email detection system designed to automatically distinguish between legitimate and malicious emails. It uses supervised machine learning to analyse URLs, sender information, and email text features to identify phishing attempts with high accuracy.

The goal of this project is to demonstrate how machine learning can enhance cyber security threat detection, particularly in email security — one of the most common attack vectors in real-world organisations.

  Problem Statement

Phishing attacks are one of the leading causes of data breaches, financial fraud, and credential theft. Traditional rule-based filters struggle to keep up with constantly evolving phishing techniques.
This project addresses this challenge by training machine learning models that:

Learn patterns from real phishing data

Generalise to unseen emails

Provide accurate and scalable detection

  Dataset

The system is trained and evaluated using a real-world phishing dataset containing labelled emails and URLs.
Features include:

URL structure (length, special characters, domain type)
Presence of IP addresses
HTTPS usage
Email content indicators
Sender metadata

The dataset is cleaned, encoded, and split into training and testing sets as part of the preprocessing pipeline.

  Technologies Used

Python
Scikit-learn
XGBoost
Pandas
NumPy
Matplotlib / Seaborn
Jupyter Notebook

  Machine Learning Models

Multiple models were trained and compared, including:
Model	Purpose
Logistic Regression	Baseline linear classifier
Decision Tree	Rule-based learning
Random Forest	Ensemble learning
XGBoost (Best Performer)	Gradient boosting for high accuracy
XGBoost achieved the highest performance due to its ability to model complex, non-linear relationships in phishing data.

  Results

The final XGBoost model achieved:
Accuracy: 99.17%
Precision & Recall: High for phishing detection
Low false positive rate, making it suitable for real-world deployment
These results show that machine learning can reliably identify phishing emails with minimal misclassification.

  Key Features of the System

End-to-end ML pipeline:
Data loading
Feature engineering
Model training
Hyperparameter tuning
Evaluation
Multiple classifiers for comparison
Confusion matrix and performance metrics
Reproducible experiments in Jupyter Notebook



Author

Sid Ali Bendris
MSc Advanced Cyber Security
Cardiff Metropolitan University
