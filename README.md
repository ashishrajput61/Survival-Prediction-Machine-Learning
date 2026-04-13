
A machine learning project that predicts Titanic passenger survival using the Seaborn Titanic dataset. Covers EDA, feature engineering, and classification models built with Python and scikit-learn.
markdown# 🚢 Titanic Survival Prediction — Machine Learning

A machine learning project that predicts whether a Titanic passenger survived 
or not, using the built-in Titanic dataset from the Seaborn library. This 
project demonstrates a complete ML pipeline from raw data to a evaluated 
classification model.

## 📌 Project Overview

The sinking of the RMS Titanic is one of the most infamous shipwrecks in 
history. This project uses passenger data (age, sex, class, fare, etc.) to 
build a model that predicts survival — a classic and beginner-friendly 
machine learning classification problem.

## 🔍 Key Steps

- **Data Loading** — Loaded the Titanic dataset directly using `seaborn.load_dataset('titanic')`
- **Exploratory Data Analysis (EDA)** — Survival rates by gender, class, age, 
  embarkation point, and more
- **Data Preprocessing** — Handling missing values, encoding categorical 
  features (sex, embarked), dropping irrelevant columns
- **Feature Engineering** — Selecting meaningful features like `pclass`, `sex`, 
  `age`, `sibsp`, `parch`, `fare`, `embarked`
- **Model Training** — Training classification models to predict survival
- **Model Evaluation** — Accuracy score, confusion matrix, classification report

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Jupyter Notebook | Development environment |
| Seaborn | Dataset source & visualization |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models & evaluation |
| Matplotlib | Plotting & charts |

## 🚀 Getting Started

1. Clone the repository:
```bash
   git clone https://github.com/ashishrajput61/Survival-Prediction-Machine-Learning.git
   cd Survival-Prediction-Machine-Learning
```

2. Install dependencies:
```bash
   pip install -r requirement.txt
```

3. Launch the notebook:
```bash
   jupyter notebook
```

## 📁 Project Structure
Survival-Prediction-Machine-Learning/
│
├── Survival-Prediction/       # Jupyter Notebook with full analysis
├── requirement.txt            # Python dependencies
└── README.md

## 📊 Features Used

| Feature | Description |
|--------|-------------|
| `pclass` | Passenger class (1st, 2nd, 3rd) |
| `sex` | Gender of passenger |
| `age` | Age of passenger |
| `sibsp` | No. of siblings/spouses aboard |
| `parch` | No. of parents/children aboard |
| `fare` | Ticket fare paid |
| `embarked` | Port of embarkation (C, Q, S) |

## 🎯 Target Variable

`survived` — 1 if the passenger survived, 0 if not.

## 📈 Results

The trained model successfully predicts passenger survival with solid accuracy. 
See the notebook for full evaluation metrics, confusion matrix, and 
visualizations.
