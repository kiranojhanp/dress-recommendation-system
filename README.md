# Dress Recommendation System

This repository contains an R implementation for a machine learning pipeline aimed at predicting dress recommendations based on attributes such as style, price, season, and more. The project analyzes the Dresses Attribute Sales Dataset from the UCI Machine Learning Repository and applies various classification techniques, including Decision Tree, Random Forest, and Naive Bayes models.

## Project Overview

### Dataset

- **Source:** UCI Machine Learning Repository
- **Size:** 500 observations with 13 features:
  - **Categorical:** Style, Price, Size, Season, NeckLine, SleeveLength, etc.
  - **Numerical:** Rating
  - **Target Variable:** Recommendation (0 = Not Recommended, 1 = Recommended)

### Objective

Develop predictive models to classify dresses as "Recommended" (1) or "Not Recommended" (0) based on their attributes.

## Methods

### 1. Data Cleaning and Preprocessing

- **Missing Value Treatment:**

  - Dropped attributes with >15% missing values (e.g., Decoration, Pattern Type).
  - Replaced null entries with NA and removed rows with missing values.

- **Standardization:**

  - Corrected spelling errors (e.g., "Automn" → "Autumn").
  - Standardized categorical values (e.g., "s" → "S").

- **Encoding:** Categorical variables converted to numerical representations.

### 2. Exploratory Data Analysis (EDA)

- **Key Findings:**

  - Most dresses fall under "Low" or "Average" price categories.
  - Summer and Winter seasons dominate the dataset.
  - Sizes "M" and "Free" are the most popular, while "XL" is underrepresented.

- **Visualizations:**

  - Price distribution by size and season.
  - Seasonal popularity of dress styles.

### 3. Models

#### **Decision Tree**

- **Implementation:** `rpart` package
- **Optimization:**

  - Pruned using complexity parameter (cp) via 10-fold cross-validation.
  - Split dataset into 70% training and 30% validation sets.

- **Performance:**

  - **Accuracy:** 72.3%
  - **AUC:** 0.36 (poor discrimination ability).

#### **Random Forest**

- **Implementation:** `randomForest` package
- **Features:**

  - 100 decision trees (ntree = 100).
  - Hyperparameter tuning via 10-fold cross-validation.

- **Evaluation:**
  - Outperformed Decision Tree with better sensitivity and specificity.

#### Naive Bayes

- **Implementation:** `e1071` package
- **Enhancements:**

  - Binned Rating into Low, Medium, and High categories.
  - Applied 10-fold cross-validation.

- **Performance:**
  - **Accuracy:** 65.54%
  - **AUC:** 0.638 (better than Decision Tree).

### 4. Key Insights

- **High Ratings (4.6+)** are strong predictors of positive recommendations.
- **Seasonality:** Dresses for Spring are more likely to be recommended compared to other seasons.
- **Styles & Necklines:** Casual and trendy styles, along with specific necklines (e.g., Sweetheart, Boat-neck), increase recommendation likelihood.

## Results

- **Best Performing Model:** Naive Bayes
  - Higher AUC (0.638) and ROC curve performance.
  - Moderate balance between sensitivity and specificity.
- Decision Tree showed significant bias toward predicting "No" recommendations.

## Installation and Usage

**1. Clone the Repository:**

```bash
git clone https://github.com/your-username/dress-recommendation-system.git
cd dress-recommendation-system
```

**2. Prerequisites:**

- R (≥ 4.0)
- Required packages: `rpart`, `randomForest`, `e1071`, `caret`, `ggplot2`.

**3. Run the Code:**

- Load the R script in R studio
- Execute functions for EDA, data cleaning, and model evaluation.

## Contributions

Feel free to contribute to this repository by:

- Enhancing models with additional features.
- Exploring alternative classification techniques (e.g., SVM, XGBoost).

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements

- Dataset sourced from the UCI Machine Learning [Repository](https://archive.ics.uci.edu/dataset/289/dresses+attribute+sales).
- Guided by Murdoch University ICT515 coursework.
