import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

def load_data(filepath):
    """Load and return the insurance dataset"""
    df = pd.read_csv(filepath)
    # Convert binary categories to 0/1 for easier analysis
    df['sex_code'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df['smoker_code'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
    return df

def basic_eda(df):
    """Perform comprehensive exploratory data analysis"""
    print("=== Dataset Overview ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values ===")
    print(df.isna().sum())
    
    print("\n=== Descriptive Statistics ===")
    print(df.describe(include='all'))
    
    print("\n=== Unique Values in Categorical Columns ===")
    for col in ['sex', 'smoker', 'region']:
        print(f"\n{col}:")
        print(df[col].value_counts())

def plot_distributions(df):
    """Plot distributions of all variables"""
    # Numerical variables
    num_cols = ['age', 'bmi', 'children', 'charges']
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.show()
        
        # Boxplot to check for outliers
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()
    
    # Categorical variables
    cat_cols = ['sex', 'smoker', 'region']
    for col in cat_cols:
        plt.figure()
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()

def analyze_relationships(df):
    """Analyze relationships between variables and charges"""
    # Correlation matrix
    print("\n=== Correlation Matrix ===")
    corr = df[['age', 'bmi', 'children', 'charges', 'sex_code', 'smoker_code']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()
    
    # Pairplot of numerical variables
    sns.pairplot(df[['age', 'bmi', 'children', 'charges', 'smoker']], 
                 hue='smoker', palette='viridis')
    plt.suptitle('Pairplot of Numerical Variables by Smoker Status', y=1.02)
    plt.show()
    
    # Charges by categorical factors
    for col in ['sex', 'smoker', 'region']:
        plt.figure()
        sns.boxplot(data=df, x=col, y='charges')
        plt.title(f'Charges Distribution by {col}')
        plt.show()
        
        # Statistical test
        groups = df.groupby(col)['charges'].apply(list)
        if len(groups) == 2:
            # T-test for 2 groups
            t_stat, p_val = stats.ttest_ind(*groups)
            print(f"\nT-test for charges by {col}: t-stat={t_stat:.2f}, p-value={p_val:.4f}")
        else:
            # ANOVA for >2 groups
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"\nANOVA for charges by {col}: F-stat={f_stat:.2f}, p-value={p_val:.4f}")

def analyze_age_bmi_effects(df):
    """Special analysis of age and BMI effects"""
    # Age vs Charges
    plt.figure()
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.7)
    plt.title('Age vs Charges by Smoker Status')
    plt.show()
    
    # BMI vs Charges
    plt.figure()
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=0.7)
    plt.title('BMI vs Charges by Smoker Status')
    plt.show()
    
    # Create BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], 
                               bins=[0, 18.5, 25, 30, 35, 40, 100],
                               labels=['Underweight', 'Normal', 'Overweight', 
                                      'Obese I', 'Obese II', 'Obese III'])
    
    plt.figure()
    sns.boxplot(data=df, x='bmi_category', y='charges', hue='smoker')
    plt.title('Charges by BMI Category and Smoker Status')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    # Load data
    data_path = "C:\\Users\\Rahel\\Desktop\\KAIM 5&6\\Week 3\\Insurance-risk-analytics\\data\\Statistical_modeling (1)\\SM\\data\\insurance.csv"  # Update this path
    df = load_data(data_path)
    
    # Perform EDA
    print("===================== BASIC EDA =====================")
    basic_eda(df)
    
    print("\n===================== DISTRIBUTIONS =====================")
    plot_distributions(df)
    
    print("\n===================== RELATIONSHIP ANALYSIS =====================")
    analyze_relationships(df)
    
    print("\n===================== AGE & BMI EFFECTS =====================")
    analyze_age_bmi_effects(df)
    