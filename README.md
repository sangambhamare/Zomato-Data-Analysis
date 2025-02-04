```markdown
# Zomato Data Dashboard

This repository contains a single-file Streamlit dashboard for analyzing Zomato restaurant data. The dashboard loads the dataset directly from GitHub and provides multiple interactive views to explore and gain insights from the data.

## Features

- **Overview:**  
  Display the first few rows of the dataset along with a brief introduction.

- **Data Summary:**  
  View the dataset structure using `df.info()` and descriptive statistics via `df.describe()` in a standard format.

- **Missing Values:**  
  Check the number of missing values per column.

- **Restaurant Type Distribution:**  
  Visualize the distribution of restaurant types using a count plot.

- **Detailed Report:**  
  Get in-depth analysis and insights including:
  - A **Correlation Heatmap** of numerical features.
  - **Scatter Plots** for:
    - Cost for two vs. Rating.
    - Votes vs. Rating.
  - **Online Order & Table Booking Insights:**  
    Compare average ratings based on the availability of online ordering and table booking.
  - Additional textual insights to help interpret the results.

## How to Use

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/your_repo.git
   cd your_repo
   ```

2. **Install Required Packages:**

   Make sure you have Python installed. Then, install the necessary packages using pip:

   ```bash
   pip install streamlit pandas matplotlib seaborn
   ```

3. **Run the Dashboard:**

   Launch the dashboard with Streamlit:

   ```bash
   streamlit run app.py
   ```

4. **Navigate the Dashboard:**

   Use the sidebar to switch between different views:
   - **Overview**
   - **Data Summary**
   - **Missing Values**
   - **Restaurant Type Distribution**
   - **Detailed Report**

## Dataset

The dataset is loaded from the following GitHub raw URL:

```
https://raw.githubusercontent.com/sangambhamare/Zomato-Data-Analysis/master/Zomato-data-.csv
```

## Code Overview

- **Data Loading & Preprocessing:**  
  The code reads the CSV file, converts the `rate` column (formatted as "4.1/5") into a float, and prepares other columns for analysis.

- **Interactive Views:**  
  The dashboard uses Streamlit’s sidebar to provide different views for data exploration and visualization.

- **Visualizations:**  
  Includes a correlation heatmap, scatter plots for relationships between variables, and bar plots for online order and table booking insights.
