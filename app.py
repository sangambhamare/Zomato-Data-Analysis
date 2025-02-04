# Step 1: Import necessary Python libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Configure matplotlib for inline plots (useful if running in a notebook)
# %matplotlib inline

# ----------------------------------------------------------------

# Step 2: Create the data frame.
# (You can find the dataset link at the end of the article.)
dataframe = pd.read_csv("Zomato-data-.csv")
print("DataFrame Head (Before processing):")
print(dataframe.head())

# Expected Output:
#                     name online_order book_table   rate  votes  \
# 0                  Jalsa          Yes        Yes  4.1/5    775   
# 1         Spice Elephant          Yes         No  4.1/5    787   
# 2        San Churro Cafe          Yes         No  3.8/5    918   
# 3  Addhuri Udupi Bhojana           No         No  3.7/5     88   
# 4          Grand Village           No         No  3.8/5    166   
#
#    approx_cost(for two people) listed_in(type)
# 0                          800          Buffet
# 1                          800          Buffet
# 2                          800          Buffet
# 3                          300          Buffet
# 4                          600          Buffet

# ----------------------------------------------------------------

# Step 3: Convert the data type of the "rate" column to float and remove the denominator.
def handleRate(value):
    value = str(value).split('/')  # Split the string by '/'
    value = value[0]                # Take the first part (before the '/')
    return float(value)

dataframe['rate'] = dataframe['rate'].apply(handleRate)
print("\nDataFrame Head (After processing 'rate'):")
print(dataframe.head())

# Expected Output:
#                     name online_order book_table  rate  votes  \
# 0                  Jalsa          Yes        Yes   4.1    775   
# 1         Spice Elephant          Yes         No   4.1    787   
# 2        San Churro Cafe          Yes         No   3.8    918   
# 3  Addhuri Udupi Bhojana           No         No   3.7     88   
# 4          Grand Village           No         No   3.8    166   
#
#    approx_cost(for two people) listed_in(type)
# 0                          800          Buffet
# 1                          800          Buffet
# 2                          800          Buffet
# 3                          300          Buffet
# 4                          600          Buffet

# ----------------------------------------------------------------

# Step 4: Obtain a summary of the data frame.
print("\nDataFrame Info:")
dataframe.info()

# Expected Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 148 entries, 0 to 147
# Data columns (total 7 columns):
#  #   Column                       Non-Null Count  Dtype  
# ---  ------                       --------------  -----  
#  0   name                         148 non-null    object 
#  1   online_order                 148 non-null    object 
#  2   book_table                   148 non-null    object 
#  3   rate                         148 non-null    float64
#  4   votes                        148 non-null    int64  
#  5   approx_cost(for two people)  148 non-null    int64  
#  6   listed_in(type)              148 non-null    object 
# dtypes: float64(1), int64(2), object(4)
# memory usage: 8.2+ KB

# ----------------------------------------------------------------

# Step 5: Check for NULL values in the data frame.
print("\nMissing values in each column:")
print(dataframe.isnull().sum())

# Expected Output (if there are no NULL values):
# name                         0
# online_order                 0
# book_table                   0
# rate                         0
# votes                        0
# approx_cost(for two people)  0
# listed_in(type)              0
# dtype: int64

# ----------------------------------------------------------------

# Step 6: Explore the 'listed_in(type)' column.
# This plot shows the count of restaurants by their type.
sns.countplot(x=dataframe['listed_in(type)'])
plt.xlabel("Type of restaurant")
plt.title("Count of Restaurants by Type")
plt.show()
