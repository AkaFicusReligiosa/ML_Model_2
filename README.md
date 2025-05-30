
# Diabetes Prediction System using Support Vector Classifier (SVC)

This project implements a **machine learning model** using a **Support Vector Classifier (SVC)** to predict whether a person is likely to have diabetes based on medical diagnostic data. The model is trained and tested on the popular **Pima Indians Diabetes Dataset**.

## 🎯 Objective

To build an accurate and interpretable classification model using **Support Vector Machine (SVM)** to assist in early detection of diabetes for effective medical intervention.

---

## 📁 Dataset Information

- **Source**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Total Instances**: 768
- **Features**: 8 numerical attributes:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target**: `0` (Non-diabetic), `1` (Diabetic)

---

## 🚀 Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook

---

## 🧠 Model Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Feature scaling (StandardScaler)
   - Train-test split (e.g., 80/20)

2. **Model Training**
   - Support Vector Classifier (SVC) with kernel tuning

3. **Evaluation**
   - Accuracy, Confusion Matrix, ROC-AUC
   - Cross-validation (optional)

4. **Prediction**
   - Accepts user input for medical parameters
   - Returns prediction: Diabetic / Non-diabetic

---

## 📊 Evaluation Metrics

- **Accuracy Score**
- **Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC-AUC Score**

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-svc.git
   cd diabetes-prediction-svc
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:

   ```bash
   jupyter notebook diabetes_prediction.ipynb
   ```

---

## 🔮 Sample Prediction Code

```python
sample_input = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  # Example input
prediction = svc_model.predict(sample_input)
print("Diabetes Prediction:", "Positive" if prediction[0] == 1 else "Negative")
```

---

## 📈 Results

* Achieved over **78–82% accuracy** on the test set.
* SVC demonstrated good performance with proper hyperparameter tuning and feature scaling.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* [Kaggle Dataset: Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* Scikit-learn Documentation
* Community Tutorials on SVMs

```

