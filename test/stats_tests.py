import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load DVC-tracked data
try:
    df = pd.read_csv("C:\\Users\\Rahel\\Desktop\\KAIM 5-6\\Week 3\\Insurance-risk-analytics\\data\\Statistical_modeling (1)\\SM\\data\\insurance.csv")
    print("Data loaded successfully (rows:", len(df), ")")
except FileNotFoundError:
    print("Error: Run this from project root or check 'C:\\Users\\Rahel\\Desktop\\KAIM 5-6\\Week 3\\Insurance-risk-analytics\\data\\Statistical_modeling (1)\\SM\\data\\insurance.csv exists")
    exit(1)

# 1. Regional Cost Differences (ANOVA)
print("\n===== Region Analysis =====")
model = ols('charges ~ C(region)', data=df).fit()
anova_results = sm.stats.anova_lm(model)
print(anova_results)
if anova_results.iloc[0, 3] < 0.05:
    region_means = df.groupby('region')['charges'].mean()
    print("\nBusiness Insight: Highest cost region is", 
          region_means.idxmax(), f"(${region_means.max():,.0f})")

# 2. Gender Difference (Welch's t-test)
print("\n===== Gender Analysis =====")
male = df[df['sex'] == 'male']['charges']
female = df[df['sex'] == 'female']['charges']
t_stat, p_val = stats.ttest_ind(male, female, equal_var=False)
print(f"p-value: {p_val:.4f}")
if p_val < 0.05:
    gender_gap = (male.mean() - female.mean()) / female.mean() * 100
    print(f"Business Insight: Males pay {gender_gap:.1f}% higher premiums")

# 3. Smoker Impact (Mann-Whitney U)
print("\n===== Smoker Analysis =====")
smoker_charges = df[df['smoker'] == 'yes']['charges']
nonsmoker_charges = df[df['smoker'] == 'no']['charges']
u_stat, p_val = stats.mannwhitneyu(smoker_charges, nonsmoker_charges)
print(f"p-value: {p_val:.4e}")
if p_val < 0.001:
    cost_ratio = smoker_charges.median() / nonsmoker_charges.median()
    print(f"Business Insight: Smokers pay {cost_ratio:.1f}x more")