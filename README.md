# Pharmaceutical_store_sales

# Rossmann Store Sales - Exploratory Data Analysis (EDA)

This repository contains the exploratory data analysis (EDA) for the Rossmann Store Sales dataset. The primary goal of this analysis is to explore customer purchasing behavior and understand how different factors such as promotions, competitor proximity, and holidays affect sales across various stores.

## Project Overview

The Rossmann Stores dataset consists of historical sales data from over 1,000 Rossmann stores across several cities. The goal of the analysis is to extract insights into customer purchasing behavior by examining store features, sales data, and external factors like promotions and competitor presence.

### Objectives

The main objectives of the EDA are:
- Explore how sales vary across different stores and over time.
- Investigate the impact of promotions on sales and customer behavior.
- Analyze seasonal purchase behaviors (e.g., during holidays like Christmas, Easter).
- Determine the effect of competitor presence on store performance.
- Visualize trends in customer behavior, including store opening and closing times.
- Identify how assortment types, competitor distance, and store location influence sales.

### Key Questions Explored:
1. How are promotions distributed between the training and test sets?
2. What is the sales behavior before, during, and after holidays?
3. Are there any seasonal purchase behaviors (e.g., Christmas, Easter)?
4. What is the correlation between sales and the number of customers?
5. How do promotions affect sales? Are they attracting new customers or influencing existing ones?
6. Could the promos be deployed more effectively? Which stores would benefit most from promos?
7. What are the trends in customer behavior during store opening and closing times?
8. Which stores are open all weekdays, and how does that affect weekend sales?
9. How does the assortment type affect sales?
10. How does the distance to the nearest competitor affect sales? Does it matter in city centers?
11. What is the effect of opening or reopening of new competitors?

## Repository Structure

```bash
├── data/                      # Folder containing dataset files
│   ├── train.csv              # Training dataset
│   ├── test.csv               # Test dataset
│   ├── store.csv              # Store metadata
│   
├── notebooks/                 # Jupyter notebooks for EDA
│   ├── EDA.ipynb              # Main exploratory data analysis
│   
├── scripts/                   # Python scripts for modular function
│   ├── eda.py        # Functions for exploratory data 
├── README.md                  # Project README file
├── requirements.txt           # Python package requirements
└── .gitignore                 # Files to ignore in Git repository
