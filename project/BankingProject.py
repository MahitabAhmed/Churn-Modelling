import pandas as pd
import numpy as np
import statistics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
df = pd.read_csv('BankingData.csv')
print(df)
# check missing values
df.isna().sum()
# check redundancy
df.duplicated()
# check incosistency
print(df.dtypes)
# remove (unknown) values from default , housing and loan columns
df.drop(df[df['housing'] == 'unknown'].index, inplace=True)
df.drop(df[df['loan'] == 'unknown'].index, inplace=True)
df.drop(df[df['default'] == 'unknown'].index, inplace=True)
df.drop(df[df['marital'] == 'unknown'].index, inplace=True)
df.drop(df[df['job'] == 'unknown'].index, inplace=True)
df.drop(df[df['age'] == 'unknown'].index, inplace=True)
df.drop(df[df['education'] == 'unknown'].index, inplace=True)
df.drop(df[df['contact'] == 'unknown'].index, inplace=True)
df.drop(df[df['month'] == 'unknown'].index, inplace=True)
df.drop(df[df['day_of_week'] == 'unknown'].index, inplace=True)
df.drop(df[df['duration'] == 'unknown'].index, inplace=True)
df.drop(df[df['campaign'] == 'unknown'].index, inplace=True)
df.drop(df[df['pdays'] == 'unknown'].index, inplace=True)
df.drop(df[df['previous'] == 'unknown'].index, inplace=True)
df.drop(df[df['poutcome'] == 'unknown'].index, inplace=True)
df.drop(df[df['y'] == 'unknown'].index, inplace=True)

# Convert To Numerical Data
label_encoder = preprocessing.LabelEncoder()
# Encode labels into numbers (for some columns)
df['marital'] = label_encoder.fit_transform(df['marital'])
df['job'] = label_encoder.fit_transform(df['job'])
df['education'] = label_encoder.fit_transform(df['education'])
df['default'] = label_encoder.fit_transform(df['default'])
df['housing'] = label_encoder.fit_transform(df['housing'])
df['loan'] = label_encoder.fit_transform(df['loan'])
df['contact'] = label_encoder.fit_transform(df['contact'])
df['month'] = label_encoder.fit_transform(df['month'])
df['day_of_week'] = label_encoder.fit_transform(df['day_of_week'])
df['poutcome'] = label_encoder.fit_transform(df['poutcome'])
df['y'] = label_encoder.fit_transform(df['y'])
print(df.dtypes)
print(df)
#convert using mapping 
# =============================================================================
# df['marital'].value_counts()
# df['marital']=df['marital'].map({'married':2,'divorced':1,'single':0})
# =============================================================================
df.drop(['default'], axis=1, inplace=True)
print(df)
# ============================================================================================
warnings.filterwarnings('ignore')
# Shape of Data before remove oultliers
print('Shape of Data before remove oultliers :', df.shape)
print("////////////////////////////////////////////")
# Removing the outliers


def remove_outliers(df, column, multiplier=1.5):
    # Calculate the quartiles and IQR
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    # Define the upper and lower bounds for outliers
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Remove outliers from the DataFrame
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


for i in ['age', 'job', 'marital', 'education', 'housing', 'contact',
          'month', 'day_of_week', 'duration', 'campaign']:
    while True:
        initial_rows = df.shape[0]
        df = remove_outliers(df, i)
        current_rows = df.shape[0]
        if initial_rows == current_rows:
            break
# Shape of Data after remove oultliers
print('Shape of Data after remove oultliers :', df.shape)
# ===========================================================================================
#############################################################################################
# summarize the data
print('')
print('Summarize Data :')
print(df.describe())
# Calculate IQR for age , duration, education, day_of_week
print('')
print('IQR for age , duration, education,day_of_week ')
print('')
age_IQR = df.age.describe()['75%'] - df.age.describe()['25%']
print("Age IQR : ", age_IQR)

duration_IQR = df.duration.describe()['75%'] - df.duration.describe()['25%']
print("Duration IQR : ", duration_IQR)

education_IQR = df.education.describe()['75%'] - df.education.describe()['25%']
print("Education IQR : ", education_IQR)

dayofweek_IQR = df.day_of_week.describe(
)['75%'] - df.day_of_week.describe()['25%']
print("Day of week IQR : ", dayofweek_IQR)
# ========================================================================================
mean_duration = np.mean(df['duration'])
mode_duration = statistics.mode(df['duration'])
median_duration = np.median(df['duration'])
print('')
print('Duration Mean :', mean_duration)
print('Duration Mode :', mode_duration)
print('Duration Median :', median_duration)
print('')
var_age = np.var(df['age'])
std_age = np.std(df['age'])
print('Age variance :', var_age)
print('Age standard deviation :', std_age)
# =========================================================================================
# DATA EXPLORATION
# histogram for column education
plt.hist(df.education, color='gold')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title('Education Histogram')
# histogram for column age
plt.hist(df.age, color='gold')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title('Age Histogram')
# BoxPLot for duration
plt.boxplot(df.duration, notch=True, vert=False, patch_artist=True)
plt.title('Duration BoxPlot')
# BoxPlot for Education
plt.boxplot(df.education, notch=True, vert=False, patch_artist=True, boxprops=dict(facecolor='#89CFF0'))
plt.title('Education BoxPlot')
######################################VISUALIZATION##########################################
# Histogram between age and y (target)
sns.histplot(data=df, x='age', hue='y', kde=True)
plt.title('Subscription Status by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
# Density PLot
#sns.distplot(df['marital'], kde_kws={'shade': True}, color='red', hist=False)
####################################################################################
# PieChart for housing
housing_counts = df['housing'].value_counts()
plt.pie(housing_counts, labels=housing_counts.index,autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Housing Loan by Subscription Status')
plt.legend(title='Subscription Status', loc='best',bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()
##################################################################################
# PieChart for loan
loan_counts = df['loan'].value_counts()
plt.pie(loan_counts, labels=loan_counts.index,autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Personal Loan by Subscription Status')
plt.legend(title='Subscription Status', loc='best',bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()
####################################################################################
# HEATMAP for Y and campaign
# Create a crosstab of 'campaign' and 'y'
campaign_y_counts = pd.crosstab(df['campaign'], df['y'])
# Create the heatmap
sns.heatmap(campaign_y_counts, annot=True, fmt='d', cmap='Blues')
plt.title('Subscription Status by Number of Contacts')
plt.xlabel('Subscription Status')
###################################################################################
# boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='y', y='duration')
plt.title('Subscription Status by Call Duration')
plt.xlabel('Subscription Status')
plt.ylabel('Call Duration (seconds)')
plt.show()
#####################################################################################
# Create a stacked bar plot
# Create a DataFrame with 'contact' and 'y'
contact_y_df = df[['contact', 'y']]
# Create a crosstab of 'contact' and 'y'
contact_y_counts = pd.crosstab(contact_y_df['contact'], contact_y_df['y'])
# Create a stacked bar plot
contact_y_counts.plot(kind='bar', stacked=True)
plt.title('Subscription Status by Contact Type')
plt.xlabel('Contact Type')
plt.ylabel('Count')
plt.legend(title='Subscription Status', labels=['cellular', 'telephone'])
plt.show()
# ==============================================================================
# Spilt Data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    df.drop('y', axis=1), df['y'], test_size=0.3, random_state=42)
# Random Forest Classifier (TRUE)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
rfc_accuracy = accuracy_score(y_test, y_pred)
rfc_f1_score = f1_score(y_test, y_pred, average='weighted')
rfc_precision = precision_score(y_test, y_pred, average='weighted')
rfc_recall = recall_score(y_test, y_pred, average='weighted')
print('')
print('Random Forest Classifier accuracy :', rfc_accuracy)

# =============================================================================
# print('Random Forest Classifier F1 score :', rfc_f1_score)
# print('Random Forest Classifier precision :', rfc_precision)
# print('Random Forest Classifier recall :', rfc_recall)
# =============================================================================
# =============================================================================
# report=classification_report(y_test,y_pred)
# print(report)
# =============================================================================
# Desicion Tree Classifier
model = DecisionTreeClassifier(max_depth=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
dtc_accuracy = accuracy_score(y_test, y_pred)
dtc_f1_score = f1_score(y_test, y_pred,average='weighted')
dtc_precision = precision_score(y_test, y_pred,average='weighted')
dtc_recall = recall_score(y_test, y_pred,average='weighted')
print('')
print('Desicion Tree Classifier accuracy :', dtc_accuracy)

# =============================================================================
# print('Desicion Tree Classifier F1 score:', dtc_f1_score)
# print('Desicion Tree Classifier precision:', dtc_precision)
# print('Desicion Tree Classifier recall:', dtc_recall)
# print('Decision Tree Score : ', model.score(x_train, y_train))
# =============================================================================
# ==============================================================================
# ==============================================================================
# COFUSION MATRIX
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)
print('')
# ==============================================================================
# Perform data normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# LINEAR REGRESSION
# Create a LinearRegression model
linear_reg = LinearRegression()
# Train the model on the normalized training set
linear_reg.fit(X_train_scaled, y_train)
# Make predictions on the normalized testing set
y_pred = linear_reg.predict(X_test_scaled)
# Calculate the root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
# Calculate the coefficient of determination (R-squared)
r2 = r2_score(y_test, y_pred)
# Print the RMSE
print('')
print("Root Mean Squared Error (RMSE):", rmse)
#print('Coefficient of Determination (R-squared):', r2)
# ==============================================================================
# LOGISITIC REGRESSION
# Create a Logistic Regression Classifier
logreg = LogisticRegression()
# Train the model on the training set
logreg.fit(x_train, y_train)
# Make predictions on the testing set
y_pred = logreg.predict(x_test)
# Calculate evaluation metrics
log_accuracy = accuracy_score(y_test, y_pred)
log_precision = precision_score(y_test, y_pred,average='weighted')
log_recall = recall_score(y_test, y_pred,average='weighted')
log_f1 = f1_score(y_test, y_pred,average='weighted')
# Print the evaluation metrics
print('')
print("Logistic Regression Classifier Accuracy:", log_accuracy)
print('')
# =============================================================================
# print("Logistic Regression Classifier Precision:", log_precision)
# print("Logistic Regression Classifier Recall:", log_recall)
# print("Logistic Regression Classifier F1 Score:", log_f1)
# print('')
# =============================================================================


# =============================================================================
# =============================================================================
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(df)
# normalized_data = scaler.transform(df)
# print(normalized_data)
# =============================================================================
# =============================================================================
# =============================================================================
# #DECISION TREE REGESSOR
# from sklearn.tree import DecisionTreeRegressor
# dtr=DecisionTreeRegressor()
# dtr.fit(x_train, y_train)
# y_pred=dtr.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print('Decision Tree Regessor Accuracy:', accuracy)
# =============================================================================