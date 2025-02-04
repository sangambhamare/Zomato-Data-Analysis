import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Zomato Data Dashboard", layout="wide")

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error("Error loading data: " + str(e))
        return pd.DataFrame()

    # Preprocess the rate column
    def handle_rate(val):
        try:
            return float(str(val).split('/')[0])
        except Exception:
            return None
    df['rate'] = df['rate'].apply(handle_rate)

    # Convert cost column to numeric and rename
    cost_col = "approx_cost(for two people)"
    if cost_col in df.columns:
        df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
        df.rename(columns={cost_col: 'cost_for_two'}, inplace=True)
    return df

def get_data_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def plot_countplot(column, title, xlabel, rotation=45):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=column, ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    plt.xticks(rotation=rotation)
    return fig

def run_classification(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
    return results, X_test, y_test, models

def run_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "SVM Regressor": SVR(),
        "KNN Regressor": KNeighborsRegressor(),
        "Random Forest Regressor": RandomForestRegressor()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = (mse, r2)
    return results

def run_clustering(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
    ax.set_title("PCA Projection with KMeans Clusters")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    return fig

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
DATA_URL = "https://raw.githubusercontent.com/sangambhamare/Zomato-Data-Analysis/master/Zomato-data-.csv"
df = load_data(DATA_URL)

# ---------------------------
# Horizontal Navigation using Tabs
# ---------------------------
tabs = st.tabs([
    "Overview", 
    "Data Summary", 
    "Missing Values", 
    "Restaurant Type Distribution", 
    "Detailed Report",
    "ML Implementation"
])

# ---------------------------
# Tab 1: Overview
# ---------------------------
with tabs[0]:
    st.title("Zomato Data Dashboard - Overview")
    st.markdown("### Overview of the Data")
    st.dataframe(df.head(10), height=400)
    st.markdown("This dashboard provides insights into the Zomato dataset.")

# ---------------------------
# Tab 2: Data Summary
# ---------------------------
with tabs[1]:
    st.title("Data Summary")
    st.markdown("Descriptive Statistics")
    st.dataframe(df.describe())

# ---------------------------
# Tab 3: Missing Values
# ---------------------------
with tabs[2]:
    st.title("Missing Values")
    st.markdown("Count of NULL values in each column:")
    st.write(df.isnull().sum())

# ---------------------------
# Tab 4: Restaurant Type Distribution
# ---------------------------
with tabs[3]:
    st.title("Restaurant Type Distribution")
    st.markdown("Count plot for the `listed_in(type)` column:")
    if 'listed_in(type)' in df.columns:
        fig = plot_countplot(df['listed_in(type)'], "Distribution of Restaurant Types", "Type of Restaurant")
        st.pyplot(fig)
    else:
        st.error("Column 'listed_in(type)' is missing.")
    
    st.markdown("### Additional Analysis:")
    
    # 1. Line plot: Sum of votes per restaurant type
    grouped_data = df.groupby('listed_in(type)')['votes'].sum()
    result = pd.DataFrame({'votes': grouped_data})
    fig_votes_line, ax_votes_line = plt.subplots()
    ax_votes_line.plot(result, c='green', marker='o')
    ax_votes_line.set_xlabel('Type of restaurant', color='red', size=20)
    ax_votes_line.set_ylabel('Votes', color='red', size=20)
    st.pyplot(fig_votes_line)
    
    # 2. Display the restaurant(s) with the maximum votes
    max_votes = df['votes'].max()
    restaurant_with_max_votes = df.loc[df['votes'] == max_votes, 'name']
    st.markdown("#### Restaurant(s) with the maximum votes:")
    st.write(restaurant_with_max_votes)
    
    # 3. Countplot for online_order
    st.markdown("#### Online Order Count")
    fig_online_order, ax_online_order = plt.subplots()
    sns.countplot(x=df['online_order'], ax=ax_online_order)
    st.pyplot(fig_online_order)
    
    # 4. Histogram for ratings distribution
    st.markdown("#### Ratings Distribution")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(df['rate'], bins=5)
    ax_hist.set_title('Ratings Distribution')
    st.pyplot(fig_hist)
    
    # 5. Countplot for cost for two
    st.markdown("#### Cost for Two Distribution")
    fig_couple, ax_couple = plt.subplots()
    sns.countplot(x=df['cost_for_two'], ax=ax_couple)
    st.pyplot(fig_couple)
    
    # 6. Boxplot: Online Order vs. Rating
    st.markdown("#### Boxplot: Online Order vs. Rating")
    fig_box, ax_box = plt.subplots(figsize=(6,6))
    sns.boxplot(x='online_order', y='rate', data=df, ax=ax_box)
    st.pyplot(fig_box)
    
    # 7. Heatmap: Pivot table of restaurant type vs. online order
    st.markdown("#### Heatmap: Restaurant Type vs. Online Order")
    pivot_table = df.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
    fig_heat, ax_heat = plt.subplots()
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='d', ax=ax_heat)
    ax_heat.set_title('Heatmap')
    ax_heat.set_xlabel('Online Order')
    ax_heat.set_ylabel('Listed In (Type)')
    st.pyplot(fig_heat)

# ---------------------------
# Tab 5: Detailed Report & Insights
# ---------------------------
with tabs[4]:
    st.title("Detailed Report & Insights")
    st.markdown("""
    This section provides further insights into the Zomato dataset through detailed analysis and various visualizations.
    
    **Key Analyses:**
    - Correlation Analysis among numerical features.
    - Scatter plots for Cost vs. Rating and Votes vs. Rating.
    - Analysis of Online Order & Table Booking impact on ratings.
    """)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
        ax_corr.set_title("Correlation Heatmap")
        st.pyplot(fig_corr)
    else:
        st.write("No numeric columns available for correlation analysis.")

    # Scatter Plot: Cost vs. Rating
    st.subheader("Cost for Two vs. Rating")
    if 'cost_for_two' in df.columns and 'rate' in df.columns:
        fig_cost, ax_cost = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='cost_for_two', y='rate', data=df, ax=ax_cost, alpha=0.7)
        ax_cost.set_xlabel("Cost for Two")
        ax_cost.set_ylabel("Rating")
        ax_cost.set_title("Relationship between Cost for Two and Rating")
        st.pyplot(fig_cost)
    else:
        st.write("Required columns for this analysis are missing.")

    # Scatter Plot: Votes vs. Rating
    st.subheader("Votes vs. Rating")
    if 'votes' in df.columns and 'rate' in df.columns:
        fig_votes, ax_votes = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='votes', y='rate', data=df, ax=ax_votes, color='green', alpha=0.7)
        ax_votes.set_xlabel("Votes")
        ax_votes.set_ylabel("Rating")
        ax_votes.set_title("Relationship between Votes and Rating")
        st.pyplot(fig_votes)
    else:
        st.write("Required columns for this analysis are missing.")

    # Online Order & Table Booking Insights
    st.subheader("Online Order & Table Booking Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Online Order Availability**")
        if 'online_order' in df.columns:
            online_order_avg = df.groupby('online_order')['rate'].mean().reset_index()
            fig_online, ax_online = plt.subplots(figsize=(6, 4))
            sns.barplot(x='online_order', y='rate', data=online_order_avg, palette="magma", ax=ax_online)
            ax_online.set_xlabel("Online Order")
            ax_online.set_ylabel("Average Rating")
            ax_online.set_title("Average Rating by Online Order")
            st.pyplot(fig_online)
        else:
            st.write("Column 'online_order' is missing.")
    with col2:
        st.markdown("**Table Booking Availability**")
        if 'book_table' in df.columns:
            table_booking_avg = df.groupby('book_table')['rate'].mean().reset_index()
            fig_booking, ax_booking = plt.subplots(figsize=(6, 4))
            sns.barplot(x='book_table', y='rate', data=table_booking_avg, palette="cool", ax=ax_booking)
            ax_booking.set_xlabel("Table Booking")
            ax_booking.set_ylabel("Average Rating")
            ax_booking.set_title("Average Rating by Table Booking")
            st.pyplot(fig_booking)
        else:
            st.write("Column 'book_table' is missing.")

    st.subheader("Additional Insights")
    st.markdown("""
    - The correlation heatmap highlights relationships among numerical features.
    - Scatter plots reveal trends between cost, votes, and ratings.
    - Service options (online order and table booking) may influence average ratings.
    """)

# ---------------------------
# Tab 6: ML Implementation
# ---------------------------
with tabs[5]:
    st.title("ML Implementation")
    st.markdown("This section demonstrates various machine learning experiments on the Zomato dataset.")

    # Prepare data for ML experiments
    st.markdown("#### Data Preparation for ML")
    df_ml = df.copy()
    # Convert categorical Yes/No to numeric
    for col in ['online_order', 'book_table']:
        df_ml[col] = df_ml[col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    # Create a binary target: HighRating (1 if rate >= 4.0 else 0)
    df_ml['HighRating'] = df_ml['rate'].apply(lambda x: 1 if x >= 4.0 else 0)
    features = ['cost_for_two', 'votes', 'online_order', 'book_table']
    X = df_ml[features].fillna(0)
    y_class = df_ml['HighRating']
    y_reg = df_ml['rate']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.markdown("Features used: **Cost for Two**, **Votes**, **Online Order**, **Table Booking**")

    # --- Classification ---
    st.markdown("### Classification: Predicting High-Rated Restaurants")
    class_results, X_test_class, y_test_class, class_models = run_classification(X_scaled, y_class)
    st.markdown("**Model Accuracies:**")
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(class_results.keys()), y=list(class_results.values()), ax=ax_acc)
    ax_acc.set_ylim(0, 1)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Classification Accuracy")
    plt.xticks(rotation=45)
    st.pyplot(fig_acc)

    # Confusion matrix for the best classifier
    best_class_model_name = max(class_results, key=class_results.get)
    best_model = class_models[best_class_model_name]
    y_pred_best = best_model.predict(X_test_class)
    cm = confusion_matrix(y_test_class, y_pred_best)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix: {best_class_model_name}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)

    # --- Clustering & Dimensionality Reduction ---
    st.markdown("### Clustering & Dimensionality Reduction")
    n_clusters = st.slider("Select number of clusters for KMeans", min_value=2, max_value=10, value=2)
    fig_cluster = run_clustering(X_scaled, n_clusters)
    st.pyplot(fig_cluster)

    # --- Regression ---
    st.markdown("### Regression: Predicting Continuous Rating")
    reg_results = run_regression(X_scaled, y_reg)
    st.markdown("**Regression Results (MSE and R² Score):**")
    for name, (mse, r2) in reg_results.items():
        st.text(f"{name}: MSE = {mse:.2f}, R² = {r2:.2f}")

st.markdown("---")
st.markdown("Built with ❤️ by Sangam S Bhamare 2025")
