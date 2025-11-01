# 📊 DataVision [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DataVision** is a user-friendly, interactive web application built with **Streamlit** that helps users automatically clean, explore, and model their CSV data without writing a single line of code. It’s ideal for students, analysts, and professionals who want to perform Exploratory Data Analysis (EDA) and basic machine learning quickly.

---

## 🚀 Features

- **📤 CSV Upload**  
  Upload your dataset using the sidebar.

- **📊 Automatic Data Cleaning**
  - Removes missing values
  - Shows missing value summary
  - Row trimming and column removal features

- **🧭 Interactive EDA**
  - Dataset overview: rows, columns, cleaned rows
  - Missing value heatmap
  - Categorical value counts
  - Correlation heatmap, line plots, histograms
  - Scatter, box, KDE plots, pairplot sampling

- **🧠 Model Training**
  - Logistic Regression model is automatically trained if a `species` column is found
  - Shows classification report and confusion matrix

- **🔮 Make Predictions**
  - Interactive sliders allow input of features
  - Predicts the class of new data entries

- **✅ Responsive Layout**
  - Streamlit-based UI with dual-column views for side-by-side insights

---

## 📁 File Structure

DataVision---Clean-and-Analyze-Your-Data-Instantly/
│<br>
├── -app.py # Main Streamlit app<br>
├── -requirements.txt # Python package dependencies<br>
├── -README.md # Project documentation<br>
└── -.streamlit/<br>
└── -config.toml # UI customization settings<br>


---

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/amrishkharr/DataVision---Clean-and-Analyze-Your-Data-Instantly.git
cd DataVision

### 2. Install Requirements

pip install -r requirements.txt

### 3. Run the App

streamlit run app.py

```

👨‍💻 Author
Amrishkhar R
GitHub: @amrishkharr


