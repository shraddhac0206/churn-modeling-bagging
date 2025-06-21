#  Customer Churn Prediction Using Bagging-Based Ensemble Methods

This project predicts customer churn using ensemble machine learning models: Random Forest and Extra Trees. It applies bagging (Bootstrap Aggregation) to build more robust and accurate classifiers for binary classification problems.

##  Objective

- Build a predictive model to classify whether a customer will churn (`Exited = 1`) or stay (`Exited = 0`)
- Use ensemble methods to improve accuracy and reduce overfitting
- Interpret model outputs using feature importance scores

##  Techniques Used

- Random Forest Classifier (with OOB score)
- Extra Trees Classifier
- One-Hot Encoding for categorical features
- Stratified Shuffle Split for balanced train-test sets
- Performance metrics: Accuracy, Precision, Recall, F1-score, AUC
- Visualization: Confusion Matrix, ROC Curve

##  Dataset

- `Churn_Modelling.csv` (Telecom customer data)
- Key features: Customer demographics, tenure, balance, number of products, credit score, etc.
- Target column: `Exited`

##  Steps Performed

1. **Data Loading and Initial Exploration**
   - Imported data, inspected shape and types
   - Generated statistical summaries and heatmaps

2. **Data Preprocessing**
   - Dropped irrelevant features (RowNumber, CustomerId, Surname)
   - Applied OneHotEncoding to categorical variables (e.g., Geography, Gender)
   - Ensured stratified sampling for train-test split

3. **Model Training**
   - Trained Random Forest with 100 trees and `oob_score=True`
   - Trained Extra Trees model with similar settings
   - Tuned basic hyperparameters for performance comparison

4. **Model Evaluation**
   - Evaluated models on test set using:
     - Confusion Matrix
     - Classification Report
     - ROC and Precision-Recall Curves
     - Feature Importance Rankings

5. **Interpretation**
   - Identified top features affecting churn (e.g., Age, Balance, Number of Products)
   - Compared model results to select the better-performing ensemble

##  Sample Output

- Accuracy: ~86–88%
- AUC Score: > 0.90 (approx.)
- Top features: Age, Balance, IsActiveMember, CreditScore

##  Technologies

- Python 3.x
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn

##  Learnings

- Ensemble methods significantly reduce overfitting compared to a single decision tree
- Out-of-Bag (OOB) score is a powerful internal validation method
- Feature importance can guide business decisions for churn reduction

---

Feel free to fork this repo, use the code, and contribute by submitting issues or pull requests!

