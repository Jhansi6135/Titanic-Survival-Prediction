#  Titanic Survival Prediction

##  Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning techniques. It demonstrates the complete data science workflow including data preprocessing, visualization, model building, and evaluation.

---

##  Objective

To build a predictive model that determines passenger survival based on features like age, gender, passenger class, and fare.

---

##  Dataset

The dataset contains information about Titanic passengers such as:

* PassengerId
* Pclass (Ticket Class)
* Name
* Sex
* Age
* SibSp (Siblings/Spouses aboard)
* Parch (Parents/Children aboard)
* Ticket
* Fare
* Cabin
* Embarked
* Survived (Target Variable)

---

##  Technologies Used

* Python 
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn

---

##  Data Preprocessing

* Filled missing values in **Age** using mean
* Filled missing values in **Embarked** using mode
* Dropped unnecessary columns (**Cabin, Name, Ticket**)

---

##  Feature Engineering

* Created new feature: **FamilySize = SibSp + Parch + 1**

---

##  Data Visualization

* Count plot for survival distribution
* Bar plot for survival vs passenger class
* Heatmap for feature correlation

---

##  Model Building

* Algorithm Used: **Logistic Regression**
* Data split into training and testing sets (80:20)

---

##  Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

##  Results

* The model achieved good accuracy
* Gender and passenger class significantly affected survival
* Female passengers had higher survival rates

---

##  Conclusion

This project demonstrates how machine learning can be applied to real-world datasets. Proper data preprocessing and feature engineering play a crucial role in improving model performance.

---

##  How to Run the Project

1. Clone the repository
2. Install required libraries:

   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the script:

   ```
   python titanic_project.py
   ```

---

##  Author

**Pydi Jhansi**

---

##  Acknowledgment

This project is based on the famous Titanic dataset used for learning data science and machine learning concepts.
