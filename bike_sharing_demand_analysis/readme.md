# Bike Sharing Demand Analysis

## Overview
This project analyzes daily bike rental data to explore patterns in demand and predict rental counts using weather-related factors. It was developed as a personal project to demonstrate data science skills, including data cleaning, exploratory analysis, visualization, and basic machine learning.

## Dataset
- **Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (`day.csv`)
- **Description:** Contains daily bike rental counts from a bike-sharing system, along with weather (temperature, humidity, windspeed) and date information.
- **Key Columns Used:**
  - `bike_count` (target): Total bike rentals per day.
  - `temp`: Normalized temperature.
  - `hum`: Normalized humidity.
  - `windspeed`: Normalized windspeed.

## Objectives
- Clean and prepare the dataset for analysis.
- Explore how weather factors influence bike rental demand.
- Build a simple predictive model to estimate daily bike rentals.
- Visualize key findings.

## Tools and Libraries
- **Python 3**
- **Pandas**: Data loading and cleaning.
- **Matplotlib**: Data visualization.
- **scikit-learn**: Linear regression modeling.
- **Jupyter Notebook**: Interactive development environment.

## Project Structure
- `day.csv`: The raw dataset (not included in repo; download from UCI link above).
- `bike_sharing_analysis.ipynb`: Jupyter Notebook with the full analysis and code.
- `README.md`: This file.

## Steps
1. **Data Loading and Cleaning:**
   - Loaded `day.csv` using Pandas.
   - Dropped unnecessary columns (`instant`, `dteday`) and renamed `cnt` to `bike_count`.
   - Checked for missing values (none found).

2. **Exploratory Data Analysis (EDA):**
   - Calculated summary statistics (e.g., average bike count).
   - Plotted bike rentals vs. temperature to identify trends.

3. **Modeling:**
   - Used Linear Regression to predict `bike_count` based on `temp`, `hum`, and `windspeed`.
   - Split data into 80% training and 20% testing sets.
   - Evaluated model performance with R² score.

4. **Visualization:**
   - Scatter plot of bike rentals vs. temperature.
   - Scatter plot of actual vs. predicted bike counts.

## Results
- **Insights:**
  - Average daily bike rentals: ~4500 (varies by dataset).
  - Temperature positively correlates with bike rentals (visible in scatter plot).
- **Model Performance:** Achieved an R² score of approximately 0.4, indicating moderate predictive ability with room for improvement.

## How to Run
1. **Prerequisites:**
   - Install Python 3 and required libraries:
     ```bash
     pip install pandas matplotlib scikit-learn jupyter
     ```
   - Download `day.csv` from the [UCI dataset link](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) and place it in the project folder.

2. **Run the Notebook:**
   ```bash
   jupyter notebook