# 🫀 Heart Disease Prediction using Naive Bayes

This project implements a **Naive Bayes classification model** to predict the presence of heart disease using the **Heart Disease dataset**.

The model is implemented in Python using **Scikit-learn**, and its performance is evaluated using **Accuracy, Precision, and Recall**.

## 📌 Project Overview

The objective of this project is to build a machine learning model that can classify whether a patient is likely to have heart disease based on the available medical features.

The project uses the **Gaussian Naive Bayes (`GaussianNB`)** algorithm for classification.

## 🛠️ Technologies Used

* 🐍 Python
* 📊 Pandas
* 🤖 Scikit-learn
* 📓 Jupyter Notebook

## 📂 Dataset

The project uses a heart disease dataset stored as:

```text
heart.csv
```

The target column is:

```text
target
```

The remaining columns are used as input features for prediction.

## 🔄 Machine Learning Workflow

The project follows these steps:

1. Import the required Python libraries.
2. Load the heart disease dataset using Pandas.
3. Separate the dataset into:

   * **Features (`X`)**
   * **Target (`y`)**
4. Split the dataset into training and testing sets.
5. Create a **Gaussian Naive Bayes** model.
6. Train the model using the training data.
7. Make predictions on the test data.
8. Evaluate the model using:

   * Accuracy
   * Precision
   * Recall

## 🧠 Algorithm Used

### Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on **Bayes' theorem**.

It assumes that the features follow a Gaussian (normal) distribution and that the features are conditionally independent given the class.

In this project, the model is created using:

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

## 📊 Model Evaluation

The following evaluation metrics are calculated:

### Accuracy

Measures the proportion of correctly classified samples.

```python
accuracy = accuracy_score(y_test, y_predict)
```

### Precision

Measures how many of the samples predicted as positive are actually positive.

```python
precision = precision_score(y_test, y_predict)
```

### Recall

Measures how many of the actual positive samples were correctly identified.

```python
recall_score = recall_score(y_test, y_predict)
```

## 📁 Project Structure

```text
Heart-Disease-Naive-Bayes/
│
├── Naive_Bayed_implementaion.ipynb
├── heart.csv
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Heart-Disease-Naive-Bayes
```

### 3. Install the required libraries

```bash
pip install pandas scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Naive_Bayed_implementaion.ipynb
```

and run the cells.

## 📈 Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

## 📌 Results

The model performance is evaluated using:

| Metric    | Value                       |
| --------- | --------------------------- |
| Accuracy  | Calculated during execution |
| Precision | Calculated during execution |
| Recall    | Calculated during execution |

The exact values are generated when the notebook is executed.

## 🚀 Future Improvements

Possible improvements for this project include:

* Compare Naive Bayes with Logistic Regression, Decision Tree, Random Forest, and SVM.
* Add data preprocessing and feature scaling where appropriate.
* Perform exploratory data analysis (EDA).
* Add confusion matrix visualization.
* Perform hyperparameter tuning.
* Deploy the model as a web application.

## 👨‍💻 Author

**Kunal**

---

⭐ If you found this project useful, consider giving the repository a star!
