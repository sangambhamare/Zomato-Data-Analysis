import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration
st.set_page_config(page_title="Zomato Data Dashboard", layout="wide")

# Function to load data from a GitHub CSV file
@st.cache_data
def load_data(url):
    data = pd.read_csv(url, sep='\t')  # Adjust separator if needed (e.g., comma for CSV)
    return data

# URL for your CSV file on GitHub (use the raw link)
data_url = "https://raw.githubusercontent.com/your_username/your_repo/main/zomato_data.csv"

# Load the data
data = load_data(data_url)

# Data cleaning: convert rating string (e.g., "4.1/5") to numeric
def extract_rating(rate_str):
    try:
        return float(rate_str.split('/')[0])
    except Exception:
        return None

data['Rating'] = data['rate'].apply(extract_rating)

# Convert cost column to numeric (if necessary)
data['cost_for_two'] = pd.to_numeric(data['approx_cost(for two people)'], errors='coerce')

# Rename columns for easier access (optional)
data.rename(columns={'listed_in(type)': 'type'}, inplace=True)

# Sidebar for navigation
st.sidebar.title("Dashboard Navigation")
option = st.sidebar.radio("Go to", 
                          ["Overview", "Ratings Distribution", "Cost vs Rating", 
                           "Online Order & Table Booking", "Restaurant Type Distribution"])

# Overview Section
if option == "Overview":
    st.title("Zomato Data Dashboard")
    st.markdown("This dashboard provides insights from the Zomato dataset.")
    st.subheader("Data Preview")
    st.write(data.head())
    st.subheader("Dataset Statistics")
    st.write(data.describe())

# Ratings Distribution Section
elif option == "Ratings Distribution":
    st.title("Ratings Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Rating'].dropna(), bins=20, kde=True, ax=ax, color='skyblue')
    ax.set_title("Distribution of Restaurant Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Cost vs Rating Section
elif option == "Cost vs Rating":
    st.title("Cost for Two vs. Rating")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='cost_for_two', y='Rating', data=data, ax=ax, alpha=0.6)
    ax.set_title("Cost for Two vs. Rating")
    ax.set_xlabel("Cost for Two")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

# Online Order & Table Booking Section
elif option == "Online Order & Table Booking":
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

# Restaurant Type Distribution Section
elif option == "Restaurant Type Distribution":
    st.title("Restaurant Type Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot count of restaurants by their type (listed_in(type) column)
    order = data['type'].value_counts().index
    sns.countplot(y='type', data=data, order=order, palette='viridis', ax=ax)
    ax.set_title("Distribution of Restaurant Types")
    ax.set_xlabel("Count")
    ax.set_ylabel("Type")
    st.pyplot(fig)

# Add an extra section for additional analysis if desired
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
