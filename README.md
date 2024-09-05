#### English Version | [Kaggle competition](https://www.kaggle.com/competitions/titanic)

# Titanic - Machine Learning from Disaster

### Competition Description

The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

### Goal

It is your job to predict if a passenger survived the sinking of the Titanic or not. For each in the test set, you must predict a 0 or 1 value for the variable.

### Approach

This project uses machine learning to predict passenger survival. The following steps were taken:

1. **Data exploration:**
   - The training and testing datasets were loaded and analyzed to understand the distribution of features such as "Age", "Pclass", and "Fare".
   - Visualizations such as histograms, bar charts, and correlation matrices were used to identify patterns in the data. For example, the relationship between survival rates and "Sex" or "Pclass" was explored.

2. **Feature Engineering:**
   - New features were introduced, including `Title`, which is extracted from passenger names, and `family_size`, which is calculated based on the number of siblings/spouses (`SibSp`) and parents/children (`Parch`) aboard.
   - Additionally, the `is_alone` feature has been introduced to indicate whether a passenger is traveling alone or with family members.

3. **Data Preprocessing:**
   - Missing values were handled by filling them with appropriate values, such as replacing missing `Embarked` values with the most common port and imputing missing `Age` values with the median age.
   - The dataset was normalized using `StandardScaler`, and categorical features like `Sex` and `Embarked` were encoded using One-Hot Encoding to prepare the data for machine learning algorithms.

4. **Model Selection and Training:**
   - `XGBoost` was chosen as the primary model due to its strong performance in classification tasks. 
   - Hyperparameter tuning was performed using `GridSearchCV` in order to optimize parameters such as `max_depth`, `learning_rate`, and `n_estimators`.

5. **Model Evaluation:**
   - The trained model was used to make predictions for the `Survived` variable among passengers in the test dataset. 
   - `Feature importance` was also plotted to identify which features had the most significant impact on survival predictions.

6. **Prediction and Submission:**
   - The trained model was used to predict the `Survived` variable for passengers in the test dataset.
   - These predictions were then saved in a submission file in the required format for submission to the Kaggle competition.
     
### Results

The developed model achieved a Kaggle score of **0.78708**, placing it in the **top 20%** of the leaderboard.

### Files

- `TTitanic - Machine Learning from Disaster.ipynb`: Jupyter Notebook containing the complete code and analysis.
- `train.csv`: Training dataset.
- `test.csv`: Testing dataset.
- `submission.csv`: File containing the predictions for the test dataset.
- `png`: Visualizations created during the project.

### Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

### Acknowledgements

This project was completed as part of a Kaggle competition. Thanks to Kaggle and the competition organizers for providing the dataset and platform.
