# Banking Customer Churn Prediction App

A Streamlit web application that predicts whether bank customers are likely to leave (churn) based on their profile data and provides personalized retention strategies.

## 🔍 Overview

This application uses machine learning to predict customer churn in the banking sector. By analyzing customer demographics, account information, and engagement metrics, it identifies at-risk customers and suggests tailored retention strategies to help banks improve customer retention rates.

## ✨ Features

- **Interactive User Interface**: Clean, user-friendly interface for inputting customer data
- **Real-time Prediction**: Instant churn probability calculation with visual indicators
- **Risk Categorization**: Classification of customers into risk levels (Low, Medium, High, Critical)
- **Personalized Retention Strategies**: Customized recommendations based on customer profile and risk level
- **Model Insights**: Visual representation of feature importance and explanation of key churn factors
- **Responsive Design**: Works on desktop and mobile devices

## 💻 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/banking-churn-prediction.git
   cd banking-churn-prediction
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. Place your "Churn Modeling.csv" file in the same directory as the application (if you have one), or the app will generate synthetic data for demonstration.

2. Run the Streamlit app:
   ```bash
   streamlit run churn_prediction_app.py
   ```

3. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501).

4. Input customer information using the form fields and click "Predict Churn Probability".

5. View the prediction results and recommended retention strategies.

## 📁 Application Structure

```
banking-churn-prediction/
├── churn_prediction_app.py    # Main Streamlit application
├── Churn Modeling.csv         # Dataset (optional)
├── requirements.txt           # Dependencies
├── README.md                  # This documentation
└── LICENSE                    # License information
```

## 🧠 Model Information

The application uses a **Random Forest Classifier** trained on banking customer data with features including:

- **Customer Demographics**: Age, gender, geography
- **Banking Relationship**: Tenure, number of products
- **Financial Indicators**: Account balance, credit score, estimated salary
- **Engagement Metrics**: Active membership status, credit card ownership

Performance metrics based on test data:
- Accuracy: ~86%
- ROC-AUC Score: ~0.73




---

Created with ❤️ by [Your Name]
