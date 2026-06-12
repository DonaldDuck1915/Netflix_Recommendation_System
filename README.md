# Recommendation Systems for Personalized Content Discovery

## Collab link (anyone can view): [collab_link](https://colab.research.google.com/drive/1dGrokxdmdDJDFYx3v5iaLrM1AbMMKPHK?usp=sharing)

# Netflix Movie Recommendation System

This project implements a memory-efficient movie recommendation system using the Surprise library, focusing on handling large datasets and optimizing RAM usage. The system processes raw Netflix rating data, filters it, trains a Singular Value Decomposition (SVD) model, and evaluates its performance.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup](#setup)
3. [Data Processing](#data-processing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Recommendation and Analysis](#recommendation-and-analysis)
7. [Results Summary](#results-summary)

## Project Overview
The goal of this project is to build a movie recommendation system that can suggest personalized movie choices to users while being mindful of memory constraints often encountered with large datasets. It leverages the `scikit-surprise` library for collaborative filtering using SVD.

## Setup
The project requires several Python libraries, which are installed at the beginning of the notebook.

```python
!pip install scikit-surprise pandas numpy matplotlib seaborn tqdm
```

Custom utility functions (`free_ram` and `ram_status`) are included to monitor and manage RAM usage throughout the process.

## Data Processing

### Parse Raw Data
The initial step involves reading `combined_data_1.txt`, extracting `user_id`, `movie_id`, and `rating`, and converting them into a pandas DataFrame with optimized data types (`int32`, `int8`) to reduce memory footprint.

### Filter Data
To manage the dataset size and improve model relevance, aggressive filtering is applied:
- Users with fewer than 30 ratings are removed.
- Movies with fewer than 50 ratings are removed.

This significantly reduces the dataset while retaining active users and popular movies.

### Downsample and Optimize Data Types
The filtered data is then further optimized by converting `user_id` and `movie_id` columns to `category` dtype, which is more memory-efficient for categorical data.

### Load Movie Titles and Merge
Movie titles are loaded from `movie_titles.csv` and merged with the main DataFrame based on `movie_id`. Missing titles are filled with 'Unknown'. The processed DataFrame is then saved to `/content/netflix_lean.csv` and the original DataFrame is deleted from memory.

### Reload Optimized Data
The optimized data is reloaded from `/content/netflix_lean.csv` with specified dtypes to ensure consistent memory efficiency for subsequent steps.

## Exploratory Data Analysis (EDA)
Comprehensive EDA is performed with memory management in mind, including:
- **Rating Distribution**: Visualizing the frequency of each rating (1-5).
- **User Activity**: Analyzing the distribution of ratings per user.
- **Movie Popularity**: Examining the distribution of ratings per movie and identifying the top 10 most rated movies.
- **Sparsity Analysis**: Calculating the overall sparsity of the user-item interaction matrix and visualizing a heatmap of a small sample to show the sparsity pattern.
- **Average Rating per Movie**: Displaying the distribution of average ratings for movies.

## Model Training and Evaluation

### Prepare Surprise Data
The pandas DataFrame is converted into a Surprise-compatible dataset using a `Reader` object. An 80/20 train-test split is performed to prepare the data for model training and evaluation.

### Train SVD Model
A Singular Value Decomposition (SVD) model is trained using `scikit-surprise` with optimized parameters (`n_factors=20`, `n_epochs=15`, `lr_all=0.005`, `reg_all=0.02`) to ensure a lightweight and efficient model.

### Evaluate Model Performance
The model's performance is evaluated using:
- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the errors in rating predictions.
- **MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and actual ratings.
- **MAP@10 (Mean Average Precision at 10)**: A memory-efficient implementation to assess the ranking quality of the top 10 recommendations.

## Recommendation and Analysis

### Recommendation Function
A `get_top_k_svd` function is defined to generate top-K movie recommendations for a given user, excluding movies they have already rated, and merging movie titles for readability.

### Recommendation Demo
The system demonstrates recommendations for several active users, showing their existing top ratings and the SVD-generated top 10 recommendations.

### Success and Failure Cases
- **Success Case**: Identifies a user with a high number of positive ratings and displays recommendations, aiming to show relevant suggestions.
- **Failure Case**: Identifies a user with low MAP@10 and displays recommendations to highlight areas where the model might struggle.

### Similar Movies via Latent Factors
The SVD item latent factors (`qi`) are used to find and display movies similar to a given movie, providing an insight into the model's understanding of movie relationships.

### Catalog Coverage
Catalog coverage is calculated by sampling users and determining the percentage of unique movies that appear in the top recommendations, indicating the diversity of recommendations.

## Results Summary
A final summary table presents the key performance metrics and characteristics of the recommendation system:

| Metric           | Value         | Definition                                        |
|:-----------------+---------------|:--------------------------------------------------|
| RMSE             | 0.9673        | Average rating prediction error (lower is better) |
| MAE              | 0.7545        | Average absolute error                            |
| MAP@10           | 0.6171        | Ranking quality for top-10 recommendations        |
| Catalog Coverage | 29.71%        | Percentage of items that can be recommended       |
| Dataset Size     | ~5.7K Ratings | Subset after aggressive filtering                 |
| Model Type       | SVD           | Matrix Factorization (Latent Factors)             |

This project provides a robust framework for building and evaluating a memory-efficient movie recommendation system using collaborative filtering techniques.
