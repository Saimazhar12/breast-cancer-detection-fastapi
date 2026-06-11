# 🎗️ Breast Cancer Detection using Machine Learning (XGBoost)

This project implements a machine learning-based system to detect breast cancer using the XGBoost classifier. The model is trained on a dataset of medical features and achieves high accuracy in predicting whether a tumor is benign or malignant.

# 🌐 Live Demo

[👉 Open Hugging Face App](https://huggingface.co/spaces/Saim1598/breast-cancer-detection)

# 📌 Project Overview

The goal of this project is to build a binary classification model that can accurately detect breast cancer. The project includes data preprocessing, exploratory data analysis (EDA), feature scaling, and model optimization.

# 📂 Dataset

- Total Samples: 569
- Features: 30 numerical features
- Target Classes: Benign (0), Malignant (1)
- No missing values in dataset

# ⚙️ Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (pairplots, heatmaps, correlation analysis)
- Feature scaling using StandardScaler
- Model training using XGBoost
- Hyperparameter tuning (Randomized Search & Grid Search)
- Cross-validation for model evaluation

# 🧠 Model & Training

- Model: XGBoost Classifier
- Train-Test Split: 80% / 20%
- Feature Scaling: StandardScaler
- Hyperparameter tuning applied

# 📊 Results

- Accuracy: ~98%
- Strong precision, recall, and F1-score performance
- Cross-validation mean accuracy: ~96%

# 🌍 Web Application Deployment (FastAPI)

The trained XGBoost model was deployed as a full-stack web application using FastAPI and Hugging Face Spaces.

- Developed a FastAPI backend to serve predictions in real-time
- Designed a custom HTML/CSS frontend (`index.html`) for user interaction
- Integrated the trained model using Pickle (`model.pkl`)
- Created a `/predict` API endpoint to handle form submissions
- Processed 30 input tumor features from user input
- Returned real-time prediction results on the web interface
- Deployed the complete application on Hugging Face Spaces

# 🧩 Project Structure

- `app.py` → FastAPI backend code
- `templates/index.html` → Frontend UI for prediction
- `model.pkl` → Trained XGBoost model
- `requirements.txt` → Dependencies
- `notebook.ipynb` → Training and EDA

# 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- XGBoost
- FastAPI
- HTML, CSS
- Jinja2 Templates
- Hugging Face Spaces

# 🚀 How to Run

1. Clone the repository  
2. Install dependencies: pip install -r requirements.txt
3. Run the FastAPI app: uvicorn app:app --reload
4. Open browser and go to: http://127.0.0.1:8000


# 📌 Conclusion

This project demonstrates the effectiveness of XGBoost in medical data classification, achieving high accuracy and reliability. The model is deployed using FastAPI on Hugging Face Spaces, allowing users to perform real-time breast cancer predictions through a web interface.
