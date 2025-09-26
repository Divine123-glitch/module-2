# dda_possum_updated.py
# Comprehensive DDA on Possum Dataset with All Measures of Central Tendency, Spread, and Shape
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
 
# Set visualization style
sns.set(style="whitegrid")
 
# -----------------------------------
# Step 1: Preliminary Data Analysis
# -----------------------------------
print("=== Preliminary Data Analysis ===")
 
# Load the dataset
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
 
# Display first few rows, info, and summary
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include='all'))
 
print("\nPreliminary Interpretation: Dataset has {} rows and {} columns. Numerical vars (e.g., totlngth) will be analyzed for central tendency, spread, and shape.".format(len(df), len(df.columns)))
 
# -----------------------------------
# Step 2: Data Cleaning
# -----------------------------------
print("\n=== Data Cleaning ===")
 
# Missing values
missing = df.isnull().sum()
print("\nMissing Values:\n", missing)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
 
# Impute missing values
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)
 
# Duplicates
print("\nDuplicates:", df.duplicated().sum())
df = df.drop_duplicates()
 
# Categorical consistency
print("\nUnique Categorical Values:")
for col in ['site', 'Pop', 'sex']:
    print(f"{col}: {df[col].unique()}")
df['site'] = df['site'].astype('category')
 
print("\nCleaning Interpretation: Missing values imputed (mean/mode). Duplicates removed. Categorical vars checked for consistency.")
 
# -----------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# -----------------------------------
print("\n=== Exploratory Data Analysis (EDA) ===")
 
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
 
# Central Tendency, Spread, and Shape
print("\nMeasures of Central Tendency, Spread, and Shape:")
for col in numerical_cols:
    mean = df[col].mean()
    median = df[col].median()
    mode = df[col].mode()[0]
    midrange = (df[col].min() + df[col].max()) / 2
    min_val = df[col].min()
    max_val = df[col].max()
    std = df[col].std()
    q1 = df[col].quantile(0.25)
    q2 = df[col].quantile(0.50)  # Same as median
    q3 = df[col].quantile(0.75)
    kurt = stats.kurtosis(df[col], fisher=True)  # Fisher’s kurtosis (0 for normal)
 
    print(f"\n{col}:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Median (Q2): {median:.2f}")
    print(f"  Mode: {mode:.2f}")
    print(f"  Midrange: {midrange:.2f}")
    print(f"  Minimum: {min_val:.2f}")
    print(f"  Maximum: {max_val:.2f}")
    print(f"  Standard Deviation: {std:.2f}")
    print(f"  Q1 (25th Percentile): {q1:.2f}")
    print(f"  Q3 (75th Percentile): {q3:.2f}")
    print(f"  Kurtosis: {kurt:.2f}")
    print(f"  Interpretation: Mean ({mean:.2f}) vs. median ({median:.2f}) shows skewness if different. "
          f"Min ({min_val:.2f}) and max ({max_val:.2f}) define range. Std ({std:.2f}) indicates spread. "
          f"Q1 ({q1:.2f}), Q2 ({q2:.2f}), Q3 ({q3:.2f}) show quartiles. Kurtosis ({kurt:.2f}) > 0 suggests heavy tails, < 0 suggests light tails.")
 
    # Visualization: Histogram with Central Measures and Quartiles
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='-', label=f'Median (Q2): {median:.2f}')
    plt.axvline(mode, color='blue', linestyle=':', label=f'Mode: {mode:.2f}')
    plt.axvline(midrange, color='purple', linestyle='-.', label=f'Midrange: {midrange:.2f}')
    plt.axvline(q1, color='orange', linestyle='-', label=f'Q1: {q1:.2f}')
    plt.axvline(q3, color='brown', linestyle='-', label=f'Q3: {q3:.2f}')
    plt.title(f'{col} Distribution with Measures')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
 
# Outliers
print("\nOutlier Detection:")
for col in numerical_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers_iqr = df[(df[col] < lower) | (df[col] > upper)][col]
    z_outliers = df[np.abs(stats.zscore(df[col])) > 3][col]
    print(f"\n{col}: IQR Outliers={len(outliers_iqr)} [{lower:.2f}, {upper:.2f}], Z-Outliers={len(z_outliers)}")
    print("Interpretation: Outliers may indicate measurement errors or biological anomalies.")
 
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col])
    plt.title(f'{col} Box Plot (Outliers)')
    plt.show()
 
# Correlations
corr = df[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
print("EDA Interpretation: Strong correlations (e.g., hdlngth-totlngth) suggest related traits for DDA.")
 
# -----------------------------------
# Step 4: Diagnostic Data Analysis (DDA)
# -----------------------------------
print("\n=== Diagnostic Data Analysis (DDA) ===")
 
# Drill-Down: Why does totlngth vary?
print("\nDrill-Down: Why does 'totlngth' vary by 'sex' or 'Pop'?")
group_sex = df.groupby('sex')['totlngth'].agg(['mean', 'std', 'count']).round(2)
group_pop = df.groupby('Pop')['totlngth'].agg(['mean', 'std', 'count']).round(2)
print("By Sex:\n", group_sex)
print("By Pop:\n", group_pop)
print("Interpretation: Differences in mean totlngth by sex suggest sexual dimorphism. Pop differences may reflect environmental factors.")
 
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='sex', y='totlngth', data=df)
plt.title("Totlngth by Sex")
plt.subplot(1, 2, 2)
sns.boxplot(x='Pop', y='totlngth', data=df)
plt.title("Totlngth by Pop")
plt.tight_layout()
plt.show()
 
# T-test
t_stat, p_val = stats.ttest_ind(df[df['sex']=='m']['totlngth'], df[df['sex']=='f']['totlngth'])
print(f"T-test (Sex vs Totlngth): t={t_stat:.2f}, p={p_val:.4f}")
print("Interpretation: Low p-value (<0.05) confirms sex as a significant factor.")
 
# Data Mining: Root causes of correlations
print("\nData Mining: Root causes of totlngth correlations?")
high_corr = corr['totlngth'].abs().sort_values(ascending=False).head(3)
print("Top Correlations:\n", high_corr)
print("Interpretation: Strong correlations with hdlngth/chest suggest proportional body growth.")
 
# Anomaly Detection: Clustering
print("\nAnomaly Detection: Why outliers?")
features = df[['hdlngth', 'totlngth', 'chest']].values
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features)
outlier_cluster = df['cluster'].value_counts().idxmin()
print("Clusters (Outliers in Cluster {}):\n".format(outlier_cluster), df.groupby('cluster')['totlngth'].agg(['mean', 'count']))
 
plt.figure(figsize=(8, 6))
sns.scatterplot(x='hdlngth', y='totlngth', hue='cluster', data=df, palette='viridis')
plt.title('Clustering for Anomaly Diagnosis')
plt.show()
print("Interpretation: Small cluster may indicate errors or unique possums; check age/site.")
 
# -----------------------------------
# Step 5: Summary of Findings
# -----------------------------------
print("\n=== DDA Summary ===")
print("- Central Tendency: Mean, median, mode, midrange show central values; skewness if different.")
print("- Spread: Min, max, std, Q1-Q3 quantify range and variability.")
print("- Shape: Kurtosis indicates tail behavior; high values suggest outliers.")
print("- DDA: Sex and Pop drive totlngth variation; correlations reflect body proportions; outliers may be errors or rare cases.")
print("- Next Steps: Model totlngth or investigate outliers further.")
df.to_csv('diagnosed_possum.csv', index=False)
print("Diagnosed dataset saved as 'diagnosed_possum.csv'.")
 
# categorical_stats_possum.py
# Statistical Analysis Based on Categorical Columns in Possum Dataset
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score
 
# Set visualization style
sns.set(style="whitegrid")
 
# -----------------------------------
# Step 1: Preliminary Data Analysis
# -----------------------------------
print("=== Preliminary Data Analysis ===")
 
# Load the dataset
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
 
# Display first few rows, info, and summary
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include='all'))
 
print("\nPreliminary Interpretation: Dataset has {} rows and {} columns. Categorical vars (sex, Pop, site) will be analyzed for their impact on numerical vars.".format(len(df), len(df.columns)))
 
# -----------------------------------
# Step 2: Data Cleaning
# -----------------------------------
print("\n=== Data Cleaning ===")
 
# Missing values
missing = df.isnull().sum()
print("\nMissing Values:\n", missing)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
 
# Impute missing values
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)
 
# Duplicates
print("\nDuplicates:", df.duplicated().sum())
df = df.drop_duplicates()
 
# Categorical consistency
print("\nUnique Categorical Values:")
for col in ['site', 'Pop', 'sex']:
    print(f"{col}: {df[col].unique()}")
df['site'] = df['site'].astype('category')
 
print("\nCleaning Interpretation: Missing values imputed (mean/mode). Duplicates removed. Categorical vars ready for statistical analysis.")
 
# -----------------------------------
# Step 3: Statistical Analysis Based on Categorical Columns
# -----------------------------------
print("\n=== Statistical Analysis Based on Categorical Columns ===")
 
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
 
# 1. Group-wise Descriptive Statistics
print("\n1. Group-wise Descriptive Statistics:")
for cat_col in categorical_cols:
    print(f"\nGroup Statistics by {cat_col}:")
    for num_col in ['totlngth', 'hdlngth', 'chest']:  # Focus on key numerical vars
        group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'min', 'max', 'count']).round(2)
        print(f"\n{num_col} by {cat_col}:\n", group_stats)
        print(f"Interpretation: Differences in mean/median suggest {cat_col} influences {num_col}. High std indicates variability within groups.")
 
        # Visualization: Box Plot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f'{num_col} by {cat_col}')
        plt.xticks(rotation=45)
        plt.show()
 
# 2. T-tests for Binary Categorical Variables (sex, Pop)
print("\n2. T-tests for Binary Categorical Variables:")
for cat_col in ['sex', 'Pop']:
    print(f"\nT-tests for {cat_col}:")
    for num_col in ['totlngth', 'hdlngth', 'chest']:
        group1 = df[df[cat_col] == df[cat_col].unique()[0]][num_col]
        group2 = df[df[cat_col] == df[cat_col].unique()[1]][num_col]
        t_stat, p_val = stats.ttest_ind(group1, group2)
 
        # Cohen's d for effect size
        mean_diff = group1.mean() - group2.mean()
        pooled_std = np.sqrt(((len(group1) - 1) * group1.std()**2 + (len(group2) - 1) * group2.std()**2) / (len(group1) + len(group2) - 2))
        cohen_d = mean_diff / pooled_std if pooled_std != 0 else 0
 
        print(f"{num_col} by {cat_col}: t={t_stat:.2f}, p={p_val:.4f}, Cohen's d={cohen_d:.2f}")
        print(f"Interpretation: p<0.05 indicates significant difference. Cohen's d > 0.8 suggests large effect, diagnosing {cat_col} as a key factor.")
 
# 3. ANOVA for Multi-level Categorical Variable (site)
print("\n3. ANOVA for Site:")
for num_col in ['totlngth', 'hdlngth', 'chest']:
    groups = [df[df['site'] == s][num_col] for s in df['site'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
 
    # Eta-squared for effect size
    ss_total = np.var(df[num_col], ddof=1) * (len(df[num_col]) - 1)
    ss_between = sum(len(g) * (g.mean() - df[num_col].mean())**2 for g in groups)
    eta_squared = ss_between / ss_total if ss_total != 0 else 0
 
    print(f"{num_col} by site: F={f_stat:.2f}, p={p_val:.4f}, Eta-squared={eta_squared:.2f}")
    print(f"Interpretation: p<0.05 indicates significant differences across sites. Eta-squared > 0.14 suggests large effect, diagnosing site as influential.")
 
# 4. Chi-square Test for Categorical Relationships
print("\n4. Chi-square Test for Categorical Relationships:")
for cat_col1 in ['sex', 'Pop']:
    for cat_col2 in ['Pop', 'site']:
        if cat_col1 < cat_col2:  # Avoid duplicate pairs
            contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
            print(f"\n{cat_col1} vs {cat_col2}: Chi2={chi2:.2f}, p={p_val:.4f}, dof={dof}")
            print(f"Interpretation: p<0.05 suggests {cat_col1} and {cat_col2} are related, indicating possible confounding factors.")
 
            # Visualization: Heatmap of Contingency Table
            plt.figure(figsize=(8, 6))
            sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='d')
            plt.title(f'Contingency Table: {cat_col1} vs {cat_col2}')
            plt.show()
 
# 5. Comprehensive Measures for Numerical Vars (for context)
print("\n5. Comprehensive Measures for Numerical Variables:")
for col in numerical_cols:
    mean = df[col].mean()
    median = df[col].median()
    mode = df[col].mode()[0]
    midrange = (df[col].min() + df[col].max()) / 2
    min_val = df[col].min()
    max_val = df[col].max()
    std = df[col].std()
    q1 = df[col].quantile(0.25)
    q2 = df[col].quantile(0.50)
    q3 = df[col].quantile(0.75)
    kurt = stats.kurtosis(df[col], fisher=True)
 
    print(f"\n{col}:")
    print(f"  Mean: {mean:.2f}, Median (Q2): {median:.2f}, Mode: {mode:.2f}, Midrange: {midrange:.2f}")
    print(f"  Min: {min_val:.2f}, Max: {max_val:.2f}, Std: {std:.2f}")
    print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}, Kurtosis: {kurt:.2f}")
    print(f"  Interpretation: Skewness if mean ({mean:.2f}) ≠ median ({median:.2f}). High std ({std:.2f}) indicates variability. Kurtosis ({kurt:.2f}) > 0 suggests heavy tails.")
 
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='-', label=f'Median (Q2): {median:.2f}')
    plt.axvline(mode, color='blue', linestyle=':', label=f'Mode: {mode:.2f}')
    plt.axvline(midrange, color='purple', linestyle='-.', label=f'Midrange: {midrange:.2f}')
    plt.axvline(q1, color='orange', linestyle='-', label=f'Q1: {q1:.2f}')
    plt.axvline(q3, color='brown', linestyle='-', label=f'Q3: {q3:.2f}')
    plt.title(f'{col} Distribution with Measures')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
 
# -----------------------------------
# Step 4: Summary of Findings
# -----------------------------------
print("\n=== Summary of Findings ===")
print("- Group Stats: Differences in totlngth/hdlngth by sex/Pop/site suggest categorical influences.")
print("- T-tests: Significant p-values for sex/Pop indicate these factors drive numerical differences.")
print("- ANOVA: Site differences in measurements suggest environmental or location-specific factors.")
print("- Chi-square: Relationships between categorical vars (e.g., sex-Pop) may confound analyses.")
print("- Comprehensive Measures: Provide context for categorical impacts; high kurtosis may signal outliers.")
print("- Next Steps: Use findings for predictive modeling or further diagnose outliers by site.")
df.to_csv('stats_possum.csv', index=False)
print("Analyzed dataset saved as 'stats_possum.csv'.")