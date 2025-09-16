import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import os

# Step 1: Preliminary Analysis
def preliminary_analysis(df):
    print("=== Step 1: Preliminary Analysis ===")
    print("Columns:", df.columns.tolist())
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

# Step 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis(df):
    print("\n=== Step 2: Exploratory Data Analysis ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()

    if 'Revenue' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['Revenue'], bins=20, kde=True)
        plt.title('Revenue Distribution')
        plt.savefig('revenue_distribution.png')
        plt.close()

    if 'Products' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Products')
        plt.title('Sales Count by Product')
        plt.xticks(rotation=45)
        plt.savefig('sales_by_product.png')
        plt.close()

    print("EDA plots saved as 'correlation_matrix.png', 'revenue_distribution.png', and 'sales_by_product.png'")

# Step 3: Data Cleaning
def data_cleaning(df):
    print("\n=== Step 3: Data Cleaning ===")
    df = df.copy()
    # Ensure Revenue is consistent (recompute if needed)
    if 'Revenue' not in df.columns and 'Price' in df.columns and 'Units' in df.columns:
        df['Revenue'] = df['Price'] * df['Units']
        print("Created 'Revenue' column from Price * Units")
    
    # Fill missing values
    df['Price'] = df['Price'].fillna(df['Price'].median())
    df['Units'] = df['Units'].fillna(df['Units'].median())
    df['Products'] = df['Products'].fillna('Unknown') if 'Products' in df.columns else 'Unknown'
    df['Sales Agent'] = df['Sales Agent'].fillna('Unknown') if 'Sales Agent' in df.columns else 'Unknown'

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")

    # Convert Date to datetime (Excel serial format)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], origin='1899-12-30', unit='D', errors='coerce')
        # Update Year, Month, Week if they don't exist or need correction
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month_name()
        df['Week'] = df['Date'].dt.day_name()

    print("Missing values after cleaning:")
    print(df.isnull().sum())
    return df

# Step 4: Data Wrangling (Analysis)
def data_wrangling(df):
    print("\n=== Step 4: Data Wrangling ===")
    df = df.copy()
    # Ensure Revenue exists
    if 'Revenue' not in df.columns and 'Price' in df.columns and 'Units' in df.columns:
        df['Revenue'] = df['Price'] * df['Units']
    
    # Trends of sales by Year and Month
    if 'Year' in df.columns and 'Month' in df.columns and 'Revenue' in df.columns:
        trends_of_sales = df.groupby(["Year", "Month"])["Revenue"].sum().reset_index()
        print("\nTrends of Sales by Year and Month:")
        print(trends_of_sales)
    else:
        print("Error: Required columns (Year, Month, Revenue) not found for trends analysis.")

    # Sales summary by Product and Year
    sales_summary = df.groupby(['Products', 'Year'])['Revenue'].sum().reset_index() if 'Products' in df.columns else df.groupby(['Year'])['Revenue'].sum().reset_index()
    print("\nSales Summary by Product and Year:")
    print(sales_summary)
    
    return df, sales_summary

# Step 5: Data Preprocessing and Data Mining
def data_preprocessing(df):
    print("\n=== Step 5: Data Preprocessing ===")
    df = df.copy()
    if 'Products' in df.columns:
        df = pd.get_dummies(df, columns=['Products'], drop_first=True)
    
    scaler = StandardScaler()
    numeric_cols = [col for col in ['Price', 'Units', 'Revenue'] if col in df.columns]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print("Preprocessed DataFrame (first 5 rows):")
    print(df.head())
    return df, scaler

# Step 6: Building Models (Classification and Clustering)
def build_models(df):
    print("\n=== Step 6: Model Building ===")
    df = df.copy()
    if 'Revenue' in df.columns:
        df['Sales_Label'] = (df['Revenue'] > df['Revenue'].median()).astype(int)
    
    features = [col for col in df.columns if col in ['Price', 'Units'] or col.startswith('Products_')]
    if not features:
        print("Error: No valid features for modeling.")
        return None, None, df
    
    X = df[features]
    y = df['Sales_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[[col for col in ['Price', 'Units', 'Revenue'] if col in df.columns]])
    print("\nClustering Results (first 5 rows):")
    print(df[[col for col in ['Price', 'Units', 'Revenue', 'Cluster'] if col in df.columns]].head())
    
    return clf, kmeans, df

# Step 7: Reporting and Visualization
def reporting_visualization(df, sales_summary):
    print("\n=== Step 7: Reporting and Visualization ===")
    fig = px.line(sales_summary, x='Year', y='Revenue', color='Products' if 'Products' in sales_summary.columns else None, 
                  title='Total Sales by Product Over Time')
    fig.write('sales_trend.html')
    print("Interactive sales trend plot saved as 'sales_trend.html'")
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Price', y='Units', hue='Cluster', palette='deep')
    plt.title('Customer Segments (K-Means Clustering)')
    plt.savefig('cluster_plot.png')
    plt.close()
    print("Cluster plot saved as 'cluster_plot.png'")

# Step 8: Deployment (Simulated)
def deployment(clf, scaler):
    print("\n=== Step 8: Deployment ===")
    from joblib import dump
    dump(clf, 'random_forest_model.joblib')
    dump(scaler, 'scaler.joblib')
    print("Model and scaler saved as 'random_forest_model.joblib' and 'scaler.joblib'")
    print("For production, deploy using Flask/FastAPI or cloud services like AWS/GCP.")

# Step 9: Maintenance and Monitoring
def maintenance_monitoring():
    print("\n=== Step 9: Maintenance and Monitoring ===")
    print("Set up logging to monitor model performance (e.g., accuracy, drift).")
    print("Schedule periodic retraining with new data.")
    with open('model_log.txt', 'a') as f:
        f.write("Model performance logged at 2025-09-12 22:40:00\n")
    print("Logged performance to 'model_log.txt'")

# Step 10: Support and Troubleshooting
def support_troubleshooting():
    print("\n=== Step 10: Support and Troubleshooting ===")
    print("Common issues and solutions:")
    print("- KeyError: Check column names with df.columns and ensure 'Revenue', 'Year', 'Month', 'Units' exist.")
    print("- FileNotFoundError: Ensure 'sales_data_cleaned.xlsx' is in the correct path.")
    print("- ModuleNotFoundError: Run 'pip install pandas numpy openpyxl matplotlib seaborn plotly scikit-learn'.")
    print("For further issues, check logs in 'model_log.txt'.")

# Main execution
if __name__ == "__main__":
    file_path = 'sales_data_cleaned.xlsx'
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        exit()
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()
    
    preliminary_analysis(df)
    exploratory_data_analysis