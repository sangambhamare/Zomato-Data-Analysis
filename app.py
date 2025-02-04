import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# -------------------------
# Step 1: Import necessary Python libraries.
# -------------------------
st.title("Zomato Data Dashboard")
st.subheader("Step 1: Import necessary Python libraries")
st.code(
    '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    ''',
    language='python'
)

# -------------------------
# Step 2: Create the data frame.
# -------------------------
st.subheader("Step 2: Create the data frame")

# Use the raw GitHub URL for the CSV file.
data_url = "https://raw.githubusercontent.com/sangambhamare/Zomato-Data-Analysis/master/Zomato-data-.csv"
dataframe = pd.read_csv(data_url)

st.write("First few rows of the DataFrame:")
st.dataframe(dataframe.head())

st.code(
    '''
dataframe = pd.read_csv("https://raw.githubusercontent.com/sangambhamare/Zomato-Data-Analysis/master/Zomato-data-.csv")
print(dataframe.head())
    ''',
    language='python'
)

# -------------------------
# Step 3: Convert the data type of the "rate" column to float and remove the denominator.
# -------------------------
st.subheader("Step 3: Convert the 'rate' column to float and remove the denominator")
st.write("Before conversion:")
st.dataframe(dataframe[['rate']].head())

st.code(
    '''
def handleRate(value):
    value = str(value).split('/')
    value = value[0]
    return float(value)

dataframe['rate'] = dataframe['rate'].apply(handleRate)
print(dataframe.head())
    ''',
    language='python'
)

def handleRate(value):
    value = str(value).split('/')
    value = value[0]
    return float(value)

dataframe['rate'] = dataframe['rate'].apply(handleRate)

st.write("After conversion:")
st.dataframe(dataframe[['rate']].head())

# -------------------------
# Step 4: Obtain a summary of the data frame.
# -------------------------
st.subheader("Step 4: DataFrame Summary using dataframe.info()")
st.write("The summary of the DataFrame:")

# Capture the output of dataframe.info() using a buffer
buffer = io.StringIO()
dataframe.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.code(
    '''
dataframe.info()
    ''',
    language='python'
)

# -------------------------
# Step 5: Check for NULL values.
# -------------------------
st.subheader("Step 5: Check for NULL values")
st.write("Count of NULL values in each column:")
null_counts = dataframe.isnull().sum()
st.write(null_counts)

st.code(
    '''
print(dataframe.isnull().sum())
    ''',
    language='python'
)

# -------------------------
# Step 6: Explore the 'listed_in(type)' column.
# -------------------------
st.subheader("Step 6: Explore the 'listed_in(type)' column")
st.write("Count plot for the 'listed_in(type)' column:")

st.code(
    '''
sns.countplot(x=dataframe['listed_in(type)'])
plt.xlabel("Type of restaurant")
plt.show()
    ''',
    language='python'
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=dataframe['listed_in(type)'], ax=ax)
ax.set_xlabel("Type of Restaurant")
ax.set_ylabel("Count")
ax.set_title("Distribution of Restaurant Types")
plt.xticks(rotation=45)
st.pyplot(fig)
