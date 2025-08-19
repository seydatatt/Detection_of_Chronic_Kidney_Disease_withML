
# 🩺 Chronic Kidney Disease Prediction using Decision Tree

This project predicts whether a patient has **Chronic Kidney Disease (CKD)** using clinical data.  
Dataset: [Kaggle CKD Dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease) 🗂️

---

## 🎯 Project Objective
Predict the presence of CKD in patients based on clinical attributes using a **Decision Tree classifier**.

---

## 📊 Dataset Overview

The dataset contains **25 attributes**:

- **Numerical Features:**  
  `age`, `blood_pressure`, `blood_glucose_random`, `blood_urea`, `serum_creatinine`, `sodium`, `potassium`, `hemoglobin`, `packed_cell_volume`, `white_blood_cell_count`, `red_blood_cell_count`  

- **Categorical Features:**  
  `specific_gravity`, `albumin`, `sugar`, `red_blood_cells`, `pus_cell`, `pus_cell_clumps`, `bacteria`, `hypertension`, `diabetes_mellitus`, `coronary_artery_disease`, `appetite`, `pedal_edema`, `anemia`  

- **Target Variable:** `class` (0 = CKD, 1 = not CKD) ✅

---

## 🧪 Project Workflow

### 1️⃣ Data Loading and Preprocessing
- 🔹 Load dataset and rename columns for clarity.  
- 🔹 Convert relevant columns to numeric.  
- 🔹 Handle missing values:
  - Numerical features → random sampling  
  - Categorical features → mode  
- 🔹 Encode categorical features using `LabelEncoder`  

### 2️⃣ Exploratory Data Analysis (EDA)
- 🔹 Visualize distributions of numerical features. 📈  
- 🔹 Correlation heatmap. 🔥  
- 🔹 KDE plots for selected features by class. 🎨  

### 3️⃣ Modeling
- 🔹 Split the data into training and test sets.  
- 🔹 Train `DecisionTreeClassifier` with `random_state=42`.  
- 🔹 Evaluate model: Accuracy, Confusion Matrix, Classification Report.  
- 🔹 Visualize the Decision Tree and feature importances. 🌳  

### 4️⃣ Results
- 🔹 **Top Features Identified:**  
  1. `specific_gravity` 💧  
  2. `serum_creatinine` 🧬  
  3. `red_blood_cell_count` 🩸  
  4. `hypertension` ❤️  
- 🔹 Feature importance visualized with bar plots 📊  


## 📝 Notes
- Missing values are handled using random sampling for numeric and mode for categorical columns.  
- Categorical features are encoded using `LabelEncoder`.  
- Decision Tree feature importance may slightly vary depending on preprocessing.  


## 📄 License
MIT License
