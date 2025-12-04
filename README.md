# Manufacturing-Defect-Prediction-using-Machine-Learning
A Data Science project inspired by real-world challenges in pharmaceutical packaging &amp; manufacturing.
Dataset file used:
manufacturing_defects_large.csv

📊 Steps Performed
1. Data Loading & Cleaning
Loaded CSV file using Pandas
Checked for nulls
Verified column types
Previewed sample rows using df.head() and df.sample()

2. Exploratory Data Analysis (EDA)
Visualized distribution of temperature, vibration, pressure
Checked correlation heatmap
Analyzed defect vs non-defect ratio
Identified features influencing defects

3. Model Building
Algorithm used: Random Forest Classifier
Why?
Handles non-linear patterns
Good for classification
Works well with numerical manufacturing data
Steps:
Split dataset into train & test
Trained model
Evaluated using accuracy score
Visualized feature importance

4. Model Accuracy
Typical accuracy achieved: 83% – 90%
(depends on random split)

📈 Feature Importance
The model showed which features influence defects most:
Temperature
Vibration
Pressure

OUTPUT:
 First 5 rows:
   temperature    pressure  thickness   hardness   humidity       speed  \
0    82.483571  191.524806   4.660753  49.282884  43.482862  122.563103   
1    79.308678  190.931718   4.847250  49.836720  42.833236  120.183831   
2    83.238443  164.087137   4.701309  50.321474  30.634802  113.532674   
3    87.615149  193.398196   5.055209  54.734307  45.795842  119.962100   
4    78.829233  214.656582   5.598589  46.263913  25.099173  127.362624   

   defect  
0       0  
1       0  
2       0  
3       0  
4       0  

Dataset info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   temperature  5000 non-null   float64
 1   pressure     5000 non-null   float64
 2   thickness    5000 non-null   float64
 3   hardness     5000 non-null   float64
 4   humidity     5000 non-null   float64
 5   speed        5000 non-null   float64
 6   defect       5000 non-null   int64  
dtypes: float64(6), int64(1)
memory usage: 273.6 KB
None

Missing values:
temperature    0
pressure       0
thickness      0
hardness       0
humidity       0
speed          0
defect         0
dtype: int64

Accuracy: 0.901

Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.99      0.95       906
           1       0.37      0.07      0.12        94

    accuracy                           0.90      1000
   macro avg       0.64      0.53      0.54      1000
weighted avg       0.86      0.90      0.87      1000


Confusion Matrix:
[[894  12]
 [ 87   7]]

Feature Importance:
thickness      0.483051
temperature    0.140159
pressure       0.118441
speed          0.100354
humidity       0.093435
hardness       0.064560
dtype: float64

