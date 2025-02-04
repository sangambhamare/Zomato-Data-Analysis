import streamlit as st
import pandas as pd
import io

# Set the page configuration
st.set_page_config(page_title="Data Summary", layout="wide")

# Load the dataset from GitHub
data_url = "https://raw.githubusercontent.com/sangambhamare/Zomato-Data-Analysis/master/Zomato-data-.csv"
df = pd.read_csv(data_url)

# Preprocess: Convert the 'rate' column from a string like "4.1/5" to a float
def handleRate(value):
    try:
        return float(str(value).split('/')[0])
    except Exception:
        return None

df['rate'] = df['rate'].apply(handleRate)

# ---------------------------
# Data Summary in Standard Format
# ---------------------------
st.title("Data Summary")

# Capture the output of df.info() in a string buffer
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

st.subheader("Dataset Structure (df.info() output):")
st.text(info_str)

st.subheader("Descriptive Statistics (df.describe()):")
st.dataframe(df.describe())
