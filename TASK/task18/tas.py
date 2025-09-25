# eda_possum.py
# Comprehensive EDA on Possum Dataset with Interpretations
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
 
# Set visualization style
sns.set(style="whitegrid")
 
# -----------------------------------
# Step 1: Preliminary Data Analysis
# -----------------------------------
print("=== Preliminary Data Analysis ===")
 
# Load the dataset (replace 'possum.csv' with your file path)
# If you don't have the dataset, you can use synthetic data (see below)
try:
    df = pd.read_csv('possum.csv')
except FileNotFoundError:
    # Generate synthetic possum-like data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'case': range(1, n+1),
        'site': np.random.choice([1, 2, 3, 4, 5, 6, 7], n),
        'Pop': np.random.choice(['Vic', 'other'], n),
        'sex': np.random.choice(['m', 'f'], n),
        'age': np.random.normal(4, 1.5, n).clip(1, 10),
        'hdlngth': np.random.normal(90, 5, n),
        'skullw': np.random.normal(60, 4, n),
        'totlngth': np.random.normal(85, 5, n),
        'taill': np.random.normal(40, 3, n),
        'footlgth': np.random.normal(65, 5, n),
        'earconch': np.random.normal(45, 4, n),
        'eye': np.random.normal(15, 1.5, n),
        'chest': np.random.normal(30, 3, n),
        'belly': np.random.normal(35, 4, n)
    })
    df.to_csv('possum.csv', index=False)
    print("Synthetic possum dataset generated and saved as 'possum.csv'")
 
# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())
 
# Display basic information
print("\nDataset Info:")
print(df.info())
 
# Display summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))
 
# Interpretation
print("\nPreliminary Analysis Interpretation:")
print("- The dataset has", len(df), "rows and", len(df.columns), "columns.")
print("- Columns include numerical (e.g., hdlngth, totlngth) and categorical (e.g., sex, Pop).")
print("- Check for missing values, data types, and potential errors in the next steps.")
 
# -----------------------------------
# Step 2: Data Cleaning
# -----------------------------------
print("\n=== Data Cleaning ===")
 
# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values)
 
# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
 
# Handle missing values
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        df[column].fillna(df[column].mode()[0], inplace=True)
 
# Check for duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Number of duplicate rows after removal:", df.duplicated().sum())
 
# Check for inconsistent categorical values
print("\nUnique values in categorical columns:")
for col in ['site', 'Pop', 'sex']:
    print(f"{col}: {df[col].unique()}")
 
# Convert 'site' to categorical if numerical
df['site'] = df['site'].astype('category')
 
# Interpretation
print("\nData Cleaning Interpretation:")
if missing_values.sum() > 0:
    print("- Missing values were found and imputed (numerical: mean, categorical: mode).")
else:
    print("- No missing values found.")
print("- Duplicates, if any, were removed.")
print("- Categorical columns were checked for consistency. 'site' converted to categorical.")
 
# -----------------------------------
# Step 3: Exploratory Data Analysis
# -----------------------------------
print("\n=== Exploratory Data Analysis ===")
 
# Select numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
 
# --- Measures of Central Tendency ---
print("\nMeasures of Central Tendency:")
for col in numerical_cols:
    mean = df[col].mean()
    median = df[col].median()
    mode = df[col].mode()[0]
    midrange = (df[col].min() + df[col].max()) / 2
    print(f"\n{col}:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Median: {median:.2f}")
    print(f"  Mode: {mode:.2f}")
    print(f"  Midrange: {midrange:.2f}")
    print(f"  Interpretation: The mean ({mean:.2f}) and median ({median:.2f}) indicate the central value of {col}. "
          f"If they differ significantly, the distribution may be skewed. "
          f"The mode ({mode:.2f}) shows the most frequent value, and midrange ({midrange:.2f}) gives the average of extremes.")
 
# --- Measures of Spread ---
print("\nMeasures of Spread:")
for col in numerical_cols:
    std = df[col].std()
    variance = df[col].var()
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    range_val = df[col].max() - df[col].min()
    print(f"\n{col}:")
    print(f"  Standard Deviation: {std:.2f}")
    print(f"  Variance: {variance:.2f}")
    print(f"  IQR: {iqr:.2f}")
    print(f"  Range: {range_val:.2f}")
    print(f"  Interpretation: The standard deviation ({std:.2f}) and variance ({variance:.2f}) indicate the spread of {col}. "
          f"A higher value suggests greater variability. The IQR ({iqr:.2f}) shows the middle 50% spread, and the range ({range_val:.2f}) shows the full spread.")
 
# --- Outlier Detection ---
print("\nOutlier Detection:")
for col in numerical_cols:
    # IQR method
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
 
    # Z-score method
    z_scores = np.abs(stats.zscore(df[col]))
    outliers_z = df[z_scores > 3][col]
 
    print(f"\n{col}:")
    print(f"  IQR Outliers: {len(outliers_iqr)} values outside [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Z-score Outliers: {len(outliers_z)} values with |Z-score| > 3")
    print(f"  Interpretation: IQR outliers indicate values beyond 1.5 * IQR from Q1 or Q3, suggesting potential anomalies. "
          f"Z-score outliers (beyond 3 standard deviations) indicate extreme values. Investigate these for errors or biological significance.")
 
    # Visualize outliers with box plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col} (Outlier Detection)')
    plt.show()
 
# --- Distribution Visualizations ---
print("\nVisualizing Distributions:")
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
    print(f"Interpretation: The histogram and KDE for {col} show the distribution shape. "
          f"Check for normality, skewness, or multimodality, which affect statistical assumptions.")
 
# --- Correlation Analysis ---
print("\nCorrelation Analysis:")
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
print("Interpretation: The heatmap shows pairwise correlations. Values close to 1 or -1 indicate strong positive or negative relationships. "
      "For example, high correlation between 'hdlngth' and 'skullw' suggests they may measure related traits.")
 
# --- Categorical vs. Numerical Relationships ---
print("\nCategorical vs. Numerical Relationships:")
for cat_col in categorical_cols:
    for num_col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f'{num_col} by {cat_col}')
        plt.xticks(rotation=45)
        plt.show()
        print(f"Interpretation: The box plot shows how {num_col} varies across {cat_col} categories. "
              f"Differences in medians or spread suggest potential relationships (e.g., sex affecting totlngth).")
 
# --- Save Cleaned Data ---
df.to_csv('cleaned_possum.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_possum.csv'")
 
# -----------------------------------
# Step 4: Summary of Findings
# -----------------------------------
print("\n=== Summary of Findings ===")
print("- Central Tendency: Mean, median, mode, and midrange provide different views of central values. Significant differences suggest skewness.")
print("- Spread: Standard deviation, variance, IQR, and range quantify variability. High values indicate diverse measurements.")
print("- Outliers: IQR and Z-score methods identified potential anomalies. Investigate for errors or biological significance (e.g., unusually large possums).")
print("- Distributions: Histograms and KDEs reveal the shape of numerical variables. Check for normality or skewness.")
print("- Correlations: Strong correlations between measurements (e.g., hdlngth and skullw) suggest related traits, useful for regression.")
print("- Categorical Relationships: Differences in numerical variables across categories (e.g., sex, Pop) suggest factors influencing possum traits.")
print("Next Steps: Consider feature engineering, statistical tests, or regression modeling to predict totlngth.")






# possum_groupby_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set display options for better visibility
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# Load and clean data
df = pd.read_csv('possum.csv')
df.dropna(axis=0, inplace=True)
df = df.set_index('case')
df_copy = df.copy()

# Select numerical columns for analysis
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Group by Population (Pop)
print("\nStatistical Summary by Population (Pop):")
pop_group = df.groupby('Pop')
pop_summary = pop_group[numerical_cols].agg(['mean', 'median', 'std', 'var', 'min', 'max'])
print(pop_summary.round(2))

# Group by Sex
print("\nStatistical Summary by Sex:")
sex_group = df.groupby('sex')
sex_summary = sex_group[numerical_cols].agg(['mean', 'median', 'std', 'var', 'min', 'max'])
print(sex_summary.round(2))

# Visualize differences using box plots
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Pop', y=col, data=df)
    plt.title(f'Box Plot of {col} by Population')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sex', y=col, data=df)
    plt.title(f'Box Plot of {col} by Sex')
    plt.show()

# Statistical test for significant differences (e.g., t-test)
from scipy.stats import ttest_ind

print("\nT-test Results for Differences by Population and Sex:")
for col in numerical_cols:
    # T-test by Population
    vic_data = df[df['Pop'] == 'Vic'][col]
    other_data = df[df['Pop'] == 'other'][col]
    t_stat_pop, p_val_pop = ttest_ind(vic_data, other_data, nan_policy='omit')
    print(f"{col} (Pop): t-statistic = {t_stat_pop:.2f}, p-value = {p_val_pop:.4f}")
    
    # T-test by Sex
    male_data = df[df['sex'] == 'm'][col]
    female_data = df[df['sex'] == 'f'][col]
    t_stat_sex, p_val_sex = ttest_ind(male_data, female_data, nan_policy='omit')
    print(f"{col} (Sex): t-statistic = {t_stat_sex:.2f}, p-value = {p_val_sex:.4f}")

# Save cleaned data
df.to_csv('cleaned_possum.csv', index=False)

print("\nAnalysis completed. Cleaned data saved to 'cleaned_possum.csv'.")