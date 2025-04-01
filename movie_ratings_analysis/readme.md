# Movie Ratings Analysis and Prediction

## Overview
This project analyzes movie ratings from the MovieLens dataset to identify patterns and predict user ratings using simple features like average user and movie ratings.

## Dataset
- **Source:** [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/)
- **Files:** `ratings.csv` (user ratings), `movies.csv` (movie details).

## Objectives
- Explore rating distributions and top-rated movies.
- Predict ratings with a basic machine learning model.
- Visualize findings.

## Tools
- Python 3, Pandas, Matplotlib, scikit-learn, Jupyter Notebook.

## Steps
1. Merged ratings and movies data.
2. Analyzed average ratings and plotted their distribution.
3. Built a Linear Regression model using user and movie average ratings.
4. Evaluated predictions with Mean Squared Error.

## Results
- Top-rated movies identified (e.g., "Shawshank Redemption").
- Model MSE: ~0.7 (varies slightly).

## How to Run
1. Install dependencies: `pip install pandas matplotlib scikit-learn jupyter`
2. Download `ml-latest-small.zip`, extract `ratings.csv` and `movies.csv`.
3. Run `jupyter notebook` and open `movie_ratings_analysis.ipynb`.

## Author
[Your Name]  
[Your GitHub Link]