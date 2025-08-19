# -*- coding: utf-8 -*-
# Import Libraries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.tree import DecisionTreeClassifier, plot_tree

import warnings 
warnings.filterwarnings("ignore")

#Load Dataset

dataset = pd.read_csv("kidney_disease.csv")
dataset.drop("id", axis = 1, inplace = True)
dataset.columns = ["age","blood_pressure","specific_gravity","albumin","sugar","red_blood_cells",
                   "pus_cell","pus_cell_clumbs","bacteria","blood_glucose_random","blood_urea",
                   "serum_creatinine","sodium","potassium","hemoglobin","packed_cell_volume",
                   "white_blood_cell_count","red_blood_cell_count","hypertension","diabetes_mellitus",
                   "coronary_artery_disease","appetite","peda_edema","aanemia","class"]

dataset.info()
describe = dataset.describe()

dataset["packed_cell_volume"] = pd.to_numeric(dataset["packed_cell_volume"], errors = "coerce")
dataset["white_blood_cell_count"] = pd.to_numeric(dataset["white_blood_cell_count"], errors = "coerce")
dataset["red_blood_cell_count"] = pd.to_numeric(dataset["red_blood_cell_count"], errors = "coerce")

dataset.info()

#Exploratory Data Analysis (EDA) 
cat_cols = [col for col in dataset.columns if dataset[col].dtype == "object"] #categoric data 
num_cols = [col for col in dataset.columns if dataset[col].dtype != "object"] #float, int data -> numeric data 

for col in cat_cols:
    print(f"{col}: {dataset[col].unique()}")

dataset["diabetes_mellitus"].replace(to_replace = {"\tno": "no", "\tyes": "yes", " yes": "yes"}, inplace =True)
dataset["coronary_artery_disease"].replace(to_replace = {"\tno": "no"}, inplace =True)
dataset["class"].replace(to_replace = {"ckd\t": "ckd" }, inplace =True)

dataset["class"] = dataset["class"].map({"ckd": 0, "notckd": 1})

cat_cols = [col for col in dataset.columns if dataset[col].dtype == "object"]
num_cols = [col for col in dataset.columns if dataset[col].dtype != "object"]

plt.figure(figsize = (15,15))
plotnumber = 1
for col in num_cols:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.distplot(dataset[col])
        plt.xlabel(col)
    
    plotnumber +=1     

plt.tight_layout()
plt.show()

#Heatmap 
plt.figure(figsize = (12,8))
sns.heatmap(dataset[num_cols].corr(), annot = True, fmt = ".2f", linecolor="white", linewidths = 2)
plt.show()

def kde(col):
    grid = sns.FacetGrid(dataset, hue = "class", height = 6, aspect = 2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()

kde("hemoglobin")    
kde("white_blood_cell_count")
kde("packed_cell_volume")
#kde("albumin")
kde("specific_gravity")

#Pre-Processing: missing value problem
 
dataset.isna().sum().sort_values(ascending = False)

def solve_mv_random_value(feature):
    random_sample = dataset[feature].dropna().sample(dataset[feature].isna().sum())
    random_sample.index = dataset[dataset[feature].isnull()].index
    dataset.loc[dataset[feature].isnull(), feature] = random_sample
    
for col in num_cols:
    solve_mv_random_value(col)

dataset[num_cols].isnull().sum()

def solve_mv_mode(feature):
    mode = dataset[feature].mode()[0]
    dataset[feature] = dataset[feature].fillna(mode)

solve_mv_mode("red_blood_cells")
solve_mv_mode("pus_cell")
    
for col in cat_cols:
    solve_mv_mode(col)

dataset[cat_cols].isnull().sum()    

#Pre-Processing: feature encoding

for col in cat_cols:
    print(f"f{col}: {dataset[col].nunique}")
    
encoder = LabelEncoder()
for col in cat_cols:     
    dataset[col] = encoder.fit_transform(dataset[col])


#Model: train-test split
independent_col = [col for col in dataset.columns if col != "class"] #x 
dependent_col = "class"  #y

X = dataset[independent_col]
y = dataset[dependent_col]

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.3, random_state=42)

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

dtc_acc = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Confusion Matrix: \n", cm)
print("Classification_report: \n", cr)


#DT Visualization - feature importance 

class_names = ["ckd", "notckd"]
plt.figure(figsize = (20,10))
plot_tree(dtc, feature_names = independent_col, filled = True, rounded = True, fontsize=10)
plt.show()

feature_importance = pd.DataFrame({"Feature": independent_col, "Importance": dtc.feature_importances_})
print("Most important feature: ", feature_importance.sort_values(by="Importance", ascending=False).iloc[0])
plt.figure()
sns.barplot(x = "Importance", y= "Feature", data= feature_importance)
plt.title("Feature Importance")
plt.show()







