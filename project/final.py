import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score, f1_score
#load data

data = pd.read_csv('Churn_Modelling.csv')

data.head(30)

#preprocessing

#drop nulls

data = data.dropna()
data.head(30)

#fill missing values with mean, median, or mode

data['Age'].fillna(data['Age'].median(), inplace=True)

#Encoding Categorical Variables:

gender_mapping = {'Male': 1, 'Female': 0}
data['Gender'] = data['Gender'].map(gender_mapping)

label_encoder = LabelEncoder()
data['Geography'] = label_encoder.fit_transform(data['Geography'])

X = data.drop(['Exited','Surname'], axis=1)
y = data['Exited']

X.head(10)

#Feature Scaling:

#scaler = StandardScaler()
#numerical_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
#X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

#X.head(10)

#Splitting the dataset into train and test sets:

X = data.drop(['Exited','Surname'], axis=1)
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#decitision tree model 

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy1 = accuracy_score(y_test, y_pred)
print(f"DecisionTree Accuracy: {accuracy1:.2f}")

report = classification_report(y_test, y_pred)
print("DecisionTree Classification Report:")
print(report)

#SVM model 

from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Using an RBF kernel
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy2:.2f}")

from sklearn.metrics import classification_report

# Print classification report with zero_division parameter

report = classification_report(y_test, y_pred,zero_division=1)
print("SVM Classification Report:")
print(report)

#to improve accuracy of SVM

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}

svm = SVC()

grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_svm = SVC(**grid_search.best_params_)
best_svm.fit(X_train, y_train)

y_pred = best_svm.predict(X_test)

accuracy3 = accuracy_score(y_test, y_pred)
print(f"improve SVM Accuracy: {accuracy3:.2f}")

report = classification_report(y_test, y_pred,zero_division=1)
print("improve SVM Classification Report:")
print(report)

#Random Forest Classifier model  

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42) 
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy4 = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy4:.2f}")

report = classification_report(y_test, y_pred)
print("Random Forest Classification Report:")
print(report)

#XGBClassifier model

from xgboost import XGBClassifier


model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy5 = accuracy_score(y_test, y_pred)
print(f"XGB Accuracy: {accuracy5:.2f}")

report = classification_report(y_test, y_pred)
print("XGB Classification Report:")
print(report)

#K-Nearest Neighbours model 

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)  # Set the number of neighbors, you can adjust this parameter
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy6 = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy6:.2f}")

report = classification_report(y_test, y_pred)
print("KNN Classification Report:")
print(report)

#Stochastic Gradient Descent model 

from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log', max_iter=1000, random_state=42)  # Using 'log' for logistic regression, you can explore other loss functions
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy7 = accuracy_score(y_test, y_pred)
print(f"SGD Accuracy: {accuracy7:.2f}")

report = classification_report(y_test, y_pred,zero_division=1)
print("SGD Classification Report:")
print(report)

# =============================================================================
# #build model from scratch 
# 
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.utils import shuffle
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_classification
# 
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# 
# X_shuffled, y_shuffled = shuffle(X_train_scaled, y_train, random_state=42)
# 
# model = Sequential()
# 
# model.add(Dense(512, activation='relu', input_shape=(X_shuffled.shape[1],)))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# 
# model.summary()
# 
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
# 
# history = model.fit(X_shuffled, y_shuffled, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr])
# 
# test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# =============================================================================
