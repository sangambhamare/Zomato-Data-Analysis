import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set up the Streamlit page configuration
st.set_page_config(page_title="Zomato Data Dashboard", layout="wide")

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    
    # Convert the 'rate' column from a string like "4.1/5" to a float
    def handleRate(val):
        try:
            return float(str(val).split('/')[0])
        except Exception:
            return None
    df['rate'] = df['rate'].apply(handleRate)
    
    return df

# Use the raw GitHub URL for the dataset.
data_url = "https://raw.githubusercontent.com/sangambhamare/Zomato-Data-Analysis/master/Zomato-data-.csv"
df = load_data(data_url)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Dashboard Navigation")
view = st.sidebar.radio("Select a view:",
                        ("Overview", "Data Summary", "Missing Values", "Restaurant Type Distribution"))

# ---------------------------
# Dashboard Pages
# ---------------------------
if view == "Overview":
    st.title("Zomato Data Dashboard")
    st.markdown("### Overview of the Data")
    st.dataframe(df.head(10), height=400)

elif view == "Data Summary":
    st.title("Data Summary")
    st.markdown("Below is a standard formatted summary of the dataset:")

    # Capture the output of df.info() using a StringIO buffer
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    st.subheader("Dataset Structure (df.info()):")
    st.text(info_str)

    st.subheader("Descriptive Statistics (df.describe()):")
    st.dataframe(df.describe())

elif view == "Missing Values":
    st.title("Missing Values")
    st.markdown("Count of NULL values in each column:")
    st.write(df.isnull().sum())

elif view == "Restaurant Type Distribution":
    st.title("Restaurant Type Distribution")
    st.markdown("Count plot for the `listed_in(type)` column:")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=df['listed_in(type)'], ax=ax, palette="viridis")
    ax.set_xlabel("Type of Restaurant")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Restaurant Types")
    plt.xticks(rotation=45)
    st.pyplot(fig)
