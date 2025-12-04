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
