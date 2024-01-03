# Titanic-Survival-Prediction-Project
# Data Loading:

The code starts by loading the Titanic dataset into a Pandas DataFrame called df using the pd.read_csv function.
# Data Cleaning:

Missing values in the 'Embarked', 'Cabin', and 'Age' columns are handled by filling them with the mode (most frequent value) of each respective column.
Columns 'PassengerId' and 'Ticket' are dropped as they are considered unnecessary for the analysis.
# Feature Engineering:

The titles of passengers are extracted from the 'Name' column using regular expressions and stored in a new column called 'Title'.
# One-Hot Encoding:

Categorical variables 'Sex' and 'Embarked' are converted into numerical format using one-hot encoding. Dummy variables are created, and the original categorical columns are dropped from the DataFrame.
# More Feature Engineering and Data Preprocessing:

The 'Title' column is further one-hot encoded, and additional columns like 'Name', 'Cabin', and 'Fare' are dropped from the DataFrame.
# Train-Test Split:

The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The features are stored in X, and the target variable ('Survived') is stored in y.
# Feature Scaling:

Standard scaling is applied to the training and testing sets using StandardScaler to standardize the features.
# Support Vector Machine (SVM) Model:

A Support Vector Machine (SVM) model is instantiated with default hyperparameters, trained on the standardized training set, and used to make predictions on the test set. The accuracy of the model is printed.
# Polynomial SVM Model:

Another SVM model is created with a polynomial kernel and a specified value of C. It is trained on the standardized training set, and its accuracy on the test set is printed.
# Other Machine Learning Models:

Several other machine learning models, including Logistic Regression, Decision Tree, k-Nearest Neighbors, Random Forest, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost, are trained and their accuracies and precisions are printed.
# Voting Classifier:

A Voting Classifier is created, combining predictions from various models using a soft voting strategy. The combined model is trained on the training set, and its accuracy and precision are printed.
# Model Evaluation and Final Model:

The best-performing model (Gradient Boosting Classifier) from the Voting Classifier is selected. The accuracy, classification report, and confusion matrix are printed based on predictions on the test set.
# Test Submission:

The final model is used to predict on the test set, and the accuracy, classification report, and confusion matrix are printed for evaluation.
# Conclusion
the code encompasses data cleaning, feature engineering, model training, and evaluation using a variety of machine learning algorithms, ultimately selecting a Gradient Boosting Classifier as the final model.
