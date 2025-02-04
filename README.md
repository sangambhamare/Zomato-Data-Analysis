# Zomato Data Dashboard

This repository contains a single-file Streamlit dashboard for analyzing Zomato restaurant data. The dashboard loads the dataset directly from GitHub and provides multiple interactive views to explore and gain insights from the data.

## Live Demo

Check out the live demo of the dashboard here:  
[Zomato Data Dashboard](https://zomato-data-analysis-fndmnfcng9x3jhycdethno.streamlit.app/)

## Features

- **Overview:**  
  Displays a preview of the dataset with a brief introduction.

- **Data Summary:**  
  Shows the structure of the dataset (using `df.info()`) and descriptive statistics (using `df.describe()`).

- **Missing Values:**  
  Lists the number of missing values for each column.

- **Restaurant Type Distribution:**  
  Visualizes the distribution of restaurant types using a count plot.

- **Detailed Report:**  
  Provides in-depth analysis with:
  - A correlation heatmap of numerical features.
  - Scatter plots for Cost vs. Rating and Votes vs. Rating.
  - Bar charts comparing average ratings for restaurants offering online ordering and table booking.
  - Additional insights to help interpret the results.

## How to Use

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/your_repo.git
   cd your_repo
   ```

2. **Install Required Packages**

   Ensure you have Python installed, then run:

   ```bash
   pip install streamlit pandas matplotlib seaborn
   ```

3. **Run the Dashboard**

   Launch the dashboard with:

   ```bash
   streamlit run app.py
   ```

4. **Navigate the Dashboard**

   Use the sidebar in the dashboard to switch between:
   - Overview
   - Data Summary
   - Missing Values
   - Restaurant Type Distribution
   - Detailed Report

## Dataset

The dataset is loaded from this GitHub raw URL:

```
https://raw.githubusercontent.com/sangambhamare/Zomato-Data-Analysis/master/Zomato-data-.csv
```
