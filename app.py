import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Additional ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Classification models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------------------
# Set up the Streamlit page configuration
# ---------------------------
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
    
    # Convert cost column to numeric if not already and rename for convenience
    cost_col = "approx_cost(for two people)"
    if cost_col in df.columns:
        df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
    df.rename(columns={cost_col: 'cost_for_two'}, inplace=True)
    
    return df

# Use the raw GitHub URL for the dataset.
data_url = "https://raw.githubusercontent.com/sangambhamare/Zomato-Data-Analysis/master/Zomato-data-.csv"
df = load_data(data_url)

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
    st.markdown("Below is a standard formatted summary of the dataset:")

    # Capture the output of df.info() using a StringIO buffer
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    st.subheader("Descriptive Statistics")
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=df['listed_in(type)'], ax=ax, palette="viridis")
    ax.set_xlabel("Type of Restaurant")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Restaurant Types")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------------------------
# Tab 5: Detailed Report & Insights
# ---------------------------
with tabs[4]:
    st.title("Detailed Report & Insights")
    st.markdown("""
    This section provides further insights into the Zomato dataset through detailed analysis and various visualizations.
    
    **Key Analyses:**
    - **Correlation Analysis:** Explore the relationships among numerical features.
    - **Cost vs. Rating:** How does the cost for two relate to the restaurant rating?
    - **Votes vs. Rating:** Is there a relationship between the number of votes and the rating?
    - **Online Order & Table Booking Analysis:** How do these features affect the average ratings?
    """)
    
    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        corr = df[numeric_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
        ax_corr.set_title("Correlation Heatmap")
        st.pyplot(fig_corr)
    else:
        st.write("No numeric columns available for correlation analysis.")
    
    # --- Scatter Plot: Cost vs. Rating ---
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
    
    # --- Scatter Plot: Votes vs. Rating ---
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
    
    # --- Analysis of Online Order and Table Booking ---
    st.subheader("Online Order & Table Booking Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Online Order Availability**")
        if 'online_order' in df.columns:
            online_order_avg_rating = df.groupby('online_order')['rate'].mean().reset_index()
            fig_online, ax_online = plt.subplots(figsize=(6, 4))
            sns.barplot(x='online_order', y='rate', data=online_order_avg_rating, palette="magma", ax=ax_online)
            ax_online.set_xlabel("Online Order")
            ax_online.set_ylabel("Average Rating")
            ax_online.set_title("Average Rating by Online Order Availability")
            st.pyplot(fig_online)
        else:
            st.write("Column 'online_order' is missing.")
    
    with col2:
        st.markdown("**Table Booking Availability**")
        if 'book_table' in df.columns:
            table_booking_avg_rating = df.groupby('book_table')['rate'].mean().reset_index()
            fig_booking, ax_booking = plt.subplots(figsize=(6, 4))
            sns.barplot(x='book_table', y='rate', data=table_booking_avg_rating, palette="cool", ax=ax_booking)
            ax_booking.set_xlabel("Table Booking")
            ax_booking.set_ylabel("Average Rating")
            ax_booking.set_title("Average Rating by Table Booking Availability")
            st.pyplot(fig_booking)
        else:
            st.write("Column 'book_table' is missing.")
    
    st.subheader("Additional Insights")
    st.markdown("""
    - **Correlation Insights:** The heatmap above helps identify how strongly various numerical features are related.
    - **Cost Analysis:** The scatter plot shows if higher cost for two is associated with higher or lower ratings.
    - **Popularity Analysis:** The relationship between votes and rating can indicate if more popular restaurants (with more votes) are rated differently.
    - **Service Options:** The bar charts provide insights into how online ordering and table booking may impact average ratings.
    """)

# ---------------------------
# Tab 6: ML Implementation
# ---------------------------
with tabs[5]:
    st.title("ML Implementation")
    st.markdown("This section demonstrates several machine learning algorithms applied to the Zomato dataset.")
    
    # --- Data Preparation for ML ---
    st.markdown("### Data Preparation for ML Tasks")
    # Create a copy for ML experiments
    df_ml = df.copy()
    
    # Convert 'online_order' and 'book_table' to numeric (1 for Yes, 0 for No)
    df_ml['online_order'] = df_ml['online_order'].apply(lambda x: 1 if str(x).lower()=='yes' else 0)
    df_ml['book_table'] = df_ml['book_table'].apply(lambda x: 1 if str(x).lower()=='yes' else 0)
    
    # Create a binary classification target: HighRating (1 if rate >= 4.0, else 0)
    df_ml['HighRating'] = df_ml['rate'].apply(lambda x: 1 if x >= 4.0 else 0)
    
    # Select features for ML (classification & regression)
    features = ['cost_for_two', 'votes', 'online_order', 'book_table']
    X = df_ml[features].fillna(0)
    y_class = df_ml['HighRating']
    y_reg = df_ml['rate']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.markdown("Features used for ML experiments: **Cost for Two**, **Votes**, **Online Order**, **Table Booking**")
    
    # --- Classification ---
    st.markdown("### Classification: Predicting High-Rated Restaurants")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_class, test_size=0.3, random_state=42)
    
    models_classification = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    
    results = {}
    for name, model in models_classification.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    
    # Plot classification accuracies
    fig_acc, ax_acc = plt.subplots()
    sns.barplot(x=list(results.keys()), y=list(results.values()), ax=ax_acc)
    ax_acc.set_title("Classification Accuracy")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0,1)
    plt.xticks(rotation=45)
    st.pyplot(fig_acc)
    
    # Display confusion matrix for the best model (highest accuracy)
    best_model_name = max(results, key=results.get)
    best_model = models_classification[best_model_name]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig_cm, ax_cm = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix: {best_model_name}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)
    
    st.markdown("### Clustering & Dimensionality Reduction")
    # --- KMeans Clustering with PCA ---
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig_pca, ax_pca = plt.subplots(figsize=(8,6))
    scatter = ax_pca.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis", alpha=0.7)
    ax_pca.set_title("PCA Projection with KMeans Clusters")
    ax_pca.set_xlabel("Principal Component 1")
    ax_pca.set_ylabel("Principal Component 2")
    st.pyplot(fig_pca)
    
    st.markdown("### Regression: Predicting Continuous Rating")
    # --- Regression ---
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.3, random_state=42)
    
    models_regression = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "SVM Regressor": SVR(),
        "KNN Regressor": KNeighborsRegressor(),
        "Random Forest Regressor": RandomForestRegressor()
    }
    
    reg_results = {}
    for name, model in models_regression.items():
        model.fit(X_train_reg, y_train_reg)
        y_pred_reg = model.predict(X_test_reg)
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        reg_results[name] = (mse, r2)
    
    st.markdown("**Regression Results (Mean Squared Error and R² Score):**")
    for name, (mse, r2) in reg_results.items():
        st.text(f"{name}: MSE = {mse:.2f}, R² = {r2:.2f}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
