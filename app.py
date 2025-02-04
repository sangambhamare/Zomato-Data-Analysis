import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the Streamlit page
st.set_page_config(page_title="Zomato Data Dashboard", layout="wide")

# Function to load data from GitHub with caching
@st.cache_data
def load_data(url):
    data = pd.read_csv(url)
    # Remove any extra whitespace in the column names
    data.columns = data.columns.str.strip()
    return data

# URL for the dataset on GitHub (adjust with your username/repo details)
data_url = "https://github.com/sangambhamare/Zomato-Data-Analysis/blob/master/Zomato-data-.csv"

# Load the dataset
data = load_data(data_url)

# Diagnostic: Show column names in the dataset
st.write("Columns in the dataset:", data.columns.tolist())

# --- Data Cleaning and Preprocessing ---

# Update the column name here if needed.
# For example, if the column is actually named "Rate" with an uppercase R:
rate_column = "rate"  # Change this if the printed columns indicate a different name, e.g., "Rate"

# Extract numeric rating from the 'rate' column which is in the form "4.1/5"
def extract_rating(rate_str):
    try:
        return float(rate_str.split('/')[0])
    except Exception:
        return None

# Check if the column exists
if rate_column in data.columns:
    data['Rating'] = data[rate_column].apply(extract_rating)
else:
    st.error(f"Column '{rate_column}' not found in the dataset. Please check the CSV file for the correct column name.")
    st.stop()  # Stop execution if the column is missing

# Convert the approximate cost column to numeric.
# Adjust the column name if necessary.
cost_column = "approx_cost(for two people)"
if cost_column in data.columns:
    data['cost_for_two'] = pd.to_numeric(data[cost_column], errors='coerce')
else:
    st.error(f"Column '{cost_column}' not found in the dataset.")
    st.stop()

# Rename the 'listed_in(type)' column to 'type' for convenience.
type_column = "listed_in(type)"
if type_column in data.columns:
    data.rename(columns={type_column: 'type'}, inplace=True)
else:
    st.error(f"Column '{type_column}' not found in the dataset.")
    st.stop()

# --- Sidebar Navigation ---

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                        ["Overview", "Ratings Distribution", "Cost vs Rating", 
                         "Online Order & Table Booking", "Restaurant Type Distribution"])

# --- Dashboard Pages ---

if page == "Overview":
    st.title("Zomato Data Dashboard")
    st.markdown("This dashboard provides insights from the Zomato dataset.")
    st.subheader("Data Preview")
    st.write(data.head())
    st.subheader("Dataset Statistics")
    st.write(data.describe())

elif page == "Ratings Distribution":
    st.title("Ratings Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Rating'].dropna(), bins=20, kde=True, ax=ax, color='skyblue')
    ax.set_title("Distribution of Restaurant Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

elif page == "Cost vs Rating":
    st.title("Cost for Two vs. Rating")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='cost_for_two', y='Rating', data=data, ax=ax, alpha=0.6)
    ax.set_title("Cost for Two vs. Rating")
    ax.set_xlabel("Cost for Two")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

elif page == "Online Order & Table Booking":
    st.title("Online Order & Table Booking")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Online Order Availability")
        online_counts = data['online_order'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=online_counts.index, y=online_counts.values, palette='pastel', ax=ax1)
        ax1.set_xlabel("Online Order")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Table Booking Availability")
        booking_counts = data['book_table'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=booking_counts.index, y=booking_counts.values, palette='pastel', ax=ax2)
        ax2.set_xlabel("Book Table")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

elif page == "Restaurant Type Distribution":
    st.title("Restaurant Type Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    order = data['type'].value_counts().index
    sns.countplot(y='type', data=data, order=order, palette='viridis', ax=ax)
    ax.set_title("Distribution of Restaurant Types")
    ax.set_xlabel("Count")
    ax.set_ylabel("Type")
    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
