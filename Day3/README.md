# Day 3:

From today, I started my Machine Learning series where I got to know about the data wrangling process. 

# Stroke Prediction Project

This project aims to predict the likelihood of stroke in patients using a comprehensive dataset of demographic and health-related features. Below is a summary of the workflow and key learnings from today's work:

---

### What I Did

#### 1. **Business Problem Understanding**
- Researched the **global impact of stroke** and understood the importance of **early prediction** for improving patient outcomes.
- Reviewed **dataset attributes** and identified which features were relevant to predicting stroke risk (e.g., age, BMI, blood pressure).

#### 2. **Data Collection & Import**
- **Downloaded the dataset** directly from Google Drive using the `gdown` package.
- **Imported the dataset** into a **pandas DataFrame** for analysis and preprocessing.

#### 3. **Data Understanding**
- Explored the dataset's **shape** (number of rows and columns), **columns**, and **data types**.
- Displayed **sample rows** to get an initial sense of the data.
- Generated **descriptive statistics** to understand the distributions and central tendencies (e.g., mean, median) of features like age, BMI, and blood pressure.

#### 4. **Data Wrangling**
- **Checked for duplicate entries** and removed any redundancies to ensure data integrity.
- **Identified missing values** and visualized their patterns using **seaborn heatmaps**.
- Investigated specific columns with missing data, such as `weight_in_kg` and `bmi`, and used **logical imputation** for filling missing values:
  - For example, set `'ever_married'` to `'No'` for rows where children (likely with missing marital status) were present.

#### 5. **Data Cleaning & Preparation**
- **Filtered out rows** with critical missing values, resulting in a clean dataset.
- Discussed various strategies for handling missing data, such as:
  - **Imputation** (filling in missing values using logic or statistical methods).
  - **Dropping rows** that couldn’t be filled logically (if too many important features were missing).

#### 6. **Exploratory Data Analysis (EDA)**
- **Filtered the data** based on specific conditions (e.g., gender, age, glucose level) to understand relationships.
- Practiced **pandas indexing and selection techniques** to filter and manipulate the data.
- **Visualized missing data** patterns using **heatmaps** and explored **feature relationships** to understand the impact of each feature on stroke risk.

---

### What I Learned

- How to **import and inspect real-world health datasets** in Python.
- The importance of **understanding the data structure** before beginning analysis.
- Effective techniques for **identifying and handling duplicate and missing data**.
- Practical skills in using **pandas** for **data selection, filtering, and cleaning**.
- The value of **visualizing missing data** patterns using seaborn heatmaps.
- The significance of **data cleaning** in preparing a dataset for modeling.
- Improved skills in **documenting** and **explaining** each step of the data analysis process.

---

### Next Steps

- Moving forward, I will begin with **feature engineering** and **predictive modeling** (e.g., logistic regression, decision trees).
- Continue refining my skills in **data visualization** to better communicate insights from the data.
- Prepare the dataset for machine learning algorithms and fine-tune the model.

---

### Reflections

Today's work was a valuable hands-on experience in the **initial stages of a data science project**, emphasizing the importance of **data quality** and **understanding**. I look forward to diving deeper into the modeling phase and refining my skills in building and evaluating predictive models.

