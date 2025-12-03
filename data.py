#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%[markdown]
# # Data Analysis for "Give Me Some Credit" Dataset

#%%[markdown]
# LOAD DATASET
# %%
df = pd.read_csv("GiveMeSomeCredit/cs-training.csv", index_col=0)
df.head()

#%%[markdown]
### UNDERSTAND DATASET
# %%
df.info()
# %%
df.describe()
# %%
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
#%%
df['SeriousDlqin2yrs'].value_counts()

#%%[markdown]
# TARGET VARIABLE ANALYSIS
# %%
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
df['SeriousDlqin2yrs'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Distribution of Serious Delinquency', fontsize=14, fontweight='bold')
plt.xlabel('Serious Delinquency (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# %%
plt.subplot(1, 2, 2)
target_pct = df['SeriousDlqin2yrs'].value_counts(normalize=True) * 100
plt.pie(target_pct, labels=['No Delinquency', 'Delinquency'], autopct='%1.1f%%', 
        colors=['#2ecc71', '#e74c3c'], startangle=90)
plt.title('Percentage of Serious Delinquency', fontsize=14, fontweight='bold')

plt.tight_layout()
# plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall age distribution
axes[0].hist(df['age'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
axes[0].set_title('Age Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['age'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df["age"].median():.0f}')
axes[0].legend()

df.boxplot(column='age', by='SeriousDlqin2yrs', ax=axes[1])
axes[1].set_title('Age Distribution by Delinquency Status', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Serious Delinquency (0=No, 1=Yes)')
axes[1].set_ylabel('Age')
plt.suptitle('')

plt.tight_layout()
# plt.savefig('age_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
#%%[markdown]
# MONTHLY INCOME ANALYSIS
#%%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Remove outliers for visualization
income_clean = df['MonthlyIncome'].dropna()
income_capped = income_clean[income_clean <= income_clean.quantile(0.95)]

axes[0].hist(income_capped, bins=50, color='#27ae60', edgecolor='black', alpha=0.7)
axes[0].set_title('Monthly Income Distribution (95th percentile)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Monthly Income')
axes[0].set_ylabel('Frequency')
axes[0].axvline(income_capped.median(), color='red', linestyle='--', linewidth=2, 
                label=f'Median: ${income_capped.median():.0f}')
axes[0].legend()

# Income by target (log scale for better visualization)
income_by_target = df[['MonthlyIncome', 'SeriousDlqin2yrs']].dropna()
income_by_target.boxplot(column='MonthlyIncome', by='SeriousDlqin2yrs', ax=axes[1])
axes[1].set_title('Monthly Income by Delinquency Status', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Serious Delinquency (0=No, 1=Yes)')
axes[1].set_ylabel('Monthly Income')
axes[1].set_ylim([0, income_clean.quantile(0.95)])

plt.suptitle('')
plt.tight_layout()
# plt.savefig('income_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

late_cols = ['NumberOfTime30-59DaysPastDueNotWorse', 
             'NumberOfTime60-89DaysPastDueNotWorse', 
             'NumberOfTimes90DaysLate']
titles = ['30-59 Days Late', '60-89 Days Late', '90+ Days Late']
colors = ['#f39c12', '#e67e22', '#c0392b']

for i, (col, title, color) in enumerate(zip(late_cols, titles, colors)):
    # Cap at reasonable value for visualization
    data_capped = df[col].clip(upper=5)
    value_counts = data_capped.value_counts().sort_index()
    
    axes[i].bar(value_counts.index, value_counts.values, color=color, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{title}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Number of Times')
    axes[i].set_ylabel('Count')
    axes[i].set_xticks(range(6))
    axes[i].set_xticklabels(['0', '1', '2', '3', '4', '5+'])

plt.tight_layout()
# plt.savefig('late_payments_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

#%%[markdown]
# FEATURE CORRELATION ANALYSIS
# %%
import numpy as np  
plt.figure(figsize=(12, 10))
correlation = df.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))

sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
# plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
df.isna().sum()
# %%
plt.figure(figsize=(10, 6))
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(df)) * 100

plt.barh(range(len(missing)), missing_pct, color='#e67e22')
plt.yticks(range(len(missing)), missing.index)
plt.xlabel('Percentage of Missing Values (%)', fontsize=12)
plt.title('Missing Values by Feature', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

for i, v in enumerate(missing_pct):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center')

plt.tight_layout()

#%%[markdown]
# HANDLE MISSING VALUES
#%%
df['age'].value_counts()
# %%
df['age'].describe()
# %%
df.drop(df[df['age'] == df['age'].min()].index, inplace=True)
# %%
df['MonthlyIncome'] = df.groupby('age')['MonthlyIncome'].transform(lambda x: x.fillna(x.median()))
# %%
df.isna().sum()
# %%
df[df['MonthlyIncome'].isna()]
# %%
df = df[df['MonthlyIncome'].notna()]
# %%
df.isna().sum()
# %%
df['NumberOfDependents'].value_counts(dropna=False)
# %%
df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
# %%
df.isna().sum()

# %%
#%%[markdown]
## handling outliers
# %%
df['RevolvingUtilizationOfUnsecuredLines'].describe()
# %%
# The column RevolvingUtilizationOfUnsecuredLines shows that how over limit the user is from their original credit limit. 
df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 1.5]  # Allow up to 150%
# %%
df['RevolvingUtilizationOfUnsecuredLines'].describe()
# %%
def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    return df_clean

# Usage
df = remove_outliers_iqr(df, ['age'])

#%%
df = remove_outliers_iqr(df, ['MonthlyIncome'])

#%%
df = remove_outliers_iqr(df, ['DebtRatio'])

#%%
df = remove_outliers_iqr(df, ['NumberOfDependents'])
# %%
def plot_boxplot(df, column_name, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.boxplot(df[column_name].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
    
    plt.ylabel('Value', fontsize=12)
    plt.title(f'Boxplot of {column_name}', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.show()

# %%
plot_boxplot(df, 'age')
# %%
plot_boxplot(df, 'MonthlyIncome')

#%%
plot_boxplot(df, 'DebtRatio')

#%%
plot_boxplot(df, 'RevolvingUtilizationOfUnsecuredLines')
# %%
plot_boxplot(df, 'NumberOfDependents')
# %%
df.describe()
# %%
df['SeriousDlqin2yrs'].value_counts()
# %%
df.to_csv("GiveMeSomeCredit/GiveMeSomeCredit-cleaned.csv", index=True)