# Comprehensive Exploratory Data Analysis (EDA) Guide

A complete, step-by-step guide covering all essential EDA concepts from absolute beginner to advanced techniques.

## Table of Contents - Beginner Level

- [Environment Setup](#environment-setup)
- [Data Import and Export](#data-import-and-export)
- [Initial Data Inspection](#initial-data-inspection)
- [Understanding Data Structure](#understanding-data-structure)
- [Data Types and Conversion](#data-types-and-conversion)
- [Basic Data Selection](#basic-data-selection)
- [Data Sampling](#data-sampling)
- [Column Operations](#column-operations)
- [Row Operations](#row-operations)
- [Basic Data Cleaning](#basic-data-cleaning)
- [Simple Statistics](#simple-statistics)
- [Value Counting and Frequency](#value-counting-and-frequency)
- [Basic Sorting](#basic-sorting)
- [Simple Filtering](#simple-filtering)
- [Basic Grouping](#basic-grouping)

## Environment Setup

### Essential Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Basic Configuration

```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)

plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")
```

## Data Import and Export

### Reading CSV Files

```python
df = pd.read_csv('data.csv')

df = pd.read_csv('data.csv', sep=';')
df = pd.read_csv('data.csv', sep='\t')

df = pd.read_csv('data.csv', header=0)
df = pd.read_csv('data.csv', header=None)
df = pd.read_csv('data.csv', names=['col1', 'col2', 'col3'])

df = pd.read_csv('data.csv', skiprows=1)
df = pd.read_csv('data.csv', skiprows=[0,2])

df = pd.read_csv('data.csv', usecols=['name', 'age', 'salary'])
df = pd.read_csv('data.csv', usecols=[0, 1, 3])

df = pd.read_csv('data.csv', nrows=1000)
```

### Reading Excel Files

```python
df = pd.read_excel('data.xlsx')

df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)

all_sheets = pd.read_excel('data.xlsx', sheet_name=None)
```

### Reading JSON Files

```python
df = pd.read_json('data.json')

df = pd.read_json('data.json', orient='records')
df = pd.read_json('data.json', orient='index')
```

### Saving Data

```python
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', sep=';', index=False)

df.to_excel('output.xlsx', index=False)
df.to_excel('output.xlsx', sheet_name='MyData', index=False)

df.to_json('output.json', orient='records')
```

## Initial Data Inspection

### Basic Dataset Information

```python
len(df)
len(df.columns)
df.shape

df.columns.tolist()
len(df.columns)

df.index
df.index.name
```

### First Look at Data

```python
df.head()
df.head(10)

df.tail()
df.tail(3)

df.sample()
df.sample(5)
df.sample(frac=0.1)
```

### Data Overview

```python
df.info()

df.memory_usage(deep=True)
df.memory_usage(deep=True).sum()

df.dtypes
```

## Understanding Data Structure

### Column Analysis

```python
columns = df.columns.tolist()

len(df.select_dtypes(include=[np.number]).columns)
len(df.select_dtypes(include=['object']).columns)
len(df.select_dtypes(include=['datetime']).columns)

duplicate_cols = df.columns[df.columns.duplicated()].tolist()
```

### Index Analysis

```python
type(df.index)
df.index.is_unique
df.index.duplicated().any()

df.index[:5].tolist()

df_reset = df.reset_index()
df_reset.head()
```

### Basic Content Check

```python
df.empty

null_cols = df.columns[df.isnull().all()].tolist()

null_rows = df.index[df.isnull().all(axis=1)].tolist()
```

## Data Types and Conversion

### Understanding Current Types

```python
for col in df.columns:
    print(f"{col}: {df[col].dtype}")

for dtype in df.dtypes.unique():
    cols = df.select_dtypes(include=[dtype]).columns.tolist()
    print(f"{dtype}: {cols}")
```

### Converting Data Types

```python
df['age'] = pd.to_numeric(df['age'])
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')

df['id'] = df['id'].astype(str)

df['count'] = df['count'].astype(int)

df['price'] = df['price'].astype(float)

df['grade'] = df['grade'].astype('category')

df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
```

### Checking Conversion Success

```python
df.dtypes

df['column'] = df['column'].astype('desired_type')
df.dtypes

def check_conversion_errors(series, target_type):
    if target_type == 'numeric':
        converted = pd.to_numeric(series, errors='coerce')
        errors = series[converted.isna() & series.notna()]
        return errors
    return None

errors = check_conversion_errors(df['column'], 'numeric')
errors.tolist()
```

## Basic Data Selection

### Selecting Columns

```python
name_column = df['name']
age_column = df.age

subset = df[['name', 'age', 'salary']]

all_except = df.drop(['unwanted_col'], axis=1)
all_except = df.drop(columns=['unwanted_col'])

first_three = df.iloc[:, :3]
last_two = df.iloc[:, -2:]
```

### Selecting Rows

```python
first_row = df.iloc[0]
first_five = df.iloc[:5]
last_row = df.iloc[-1]
specific_rows = df.iloc[[0, 2, 4]]

if 'id' in df.columns and df.set_index('id', inplace=False) is not None:
    df_indexed = df.set_index('id')
    row_by_id = df_indexed.loc[123]
```

### Selecting Both Rows and Columns

```python
cell_value = df.iloc[0, 1]
cell_value = df.at[0, 'name']

subset = df.iloc[0:5, 1:4]
subset = df.loc[0:4, ['name', 'age']]
```

## Data Sampling

### Simple Sampling

```python
sample_5 = df.sample(5)
sample_10pct = df.sample(frac=0.1)
sample_with_replacement = df.sample(10, replace=True)

sample_reproducible = df.sample(5, random_state=42)
```

### Systematic Sampling

```python
every_10th = df.iloc[::10]
every_5th = df.iloc[::5]

if 'category' in df.columns:
    sample_per_group = df.groupby('category').head(5)
```

### Stratified Sampling

```python
if 'category' in df.columns:
    stratified = df.groupby('category', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 10))
    )
```

## Column Operations

### Adding New Columns

```python
# Starting with a simple dataframe:
#    name  age
# 0  John   25
# 1  Jane   30
# 2   Bob   35

df['status'] = 'active'
# Output:
#    name  age  status
# 0  John   25  active
# 1  Jane   30  active
# 2   Bob   35  active

df['full_info'] = df['name'] + ' is ' + df['age'].astype(str) + ' years old'
# Output:
#    name  age  status              full_info
# 0  John   25  active   John is 25 years old
# 1  Jane   30  active   Jane is 30 years old
# 2   Bob   35  active    Bob is 35 years old

df['age_group'] = np.where(df['age'] >= 30, 'Senior', 'Junior')
# Output:
#    name  age  status              full_info age_group
# 0  John   25  active   John is 25 years old    Junior
# 1  Jane   30  active   Jane is 30 years old    Senior
# 2   Bob   35  active    Bob is 35 years old    Senior

df['birth_year'] = 2024 - df['age']
# Output:
#    name  age  status              full_info age_group  birth_year
# 0  John   25  active   John is 25 years old    Junior        1999
# 1  Jane   30  active   Jane is 30 years old    Senior        1994
# 2   Bob   35  active    Bob is 35 years old    Senior        1989
```

### Renaming Columns

```python
df = df.rename(columns={'old_name': 'new_name'})
df = df.rename(columns={'col1': 'column_1', 'col2': 'column_2'})

df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.strip()

df = df.add_prefix('prefix_')
df = df.add_suffix('_suffix')
```

### Reordering Columns

```python
new_order = ['name', 'age', 'salary', 'department']
df = df[new_order]

cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('important_column')))
df = df[cols]

df = df.reindex(sorted(df.columns), axis=1)
```

### Removing Columns

```python
df = df.drop('unwanted_column', axis=1)
df = df.drop(columns=['unwanted_column'])

df = df.drop(['col1', 'col2'], axis=1)

cols_to_drop = [col for col in df.columns if 'temp' in col.lower()]
df = df.drop(columns=cols_to_drop)

keep_cols = ['name', 'age', 'salary']
df = df[keep_cols]
```

## Row Operations

### Adding Rows

```python
new_row = {'name': 'John', 'age': 30, 'salary': 50000}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

new_rows = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 35],
    'salary': [45000, 60000]
})
df = pd.concat([df, new_rows], ignore_index=True)

df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
```

### Removing Rows

```python
df = df.drop(0)
df = df.drop([0, 1, 2])

df = df[df['age'] >= 18]
df = df[df['salary'] > 0]

df = df.drop_duplicates()
df = df.drop_duplicates(subset=['name'])
df = df.drop_duplicates(keep='first')
df = df.drop_duplicates(keep='last')
```

### Reordering Rows

```python
df = df.sort_index()

df = df.reset_index(drop=True)

df = df.iloc[::-1]

df = df.sample(frac=1).reset_index(drop=True)
```

## Basic Data Cleaning

### Handling Missing Values (Basic)

```python
# Example dataset with missing values:
#     name    age  salary
# 0   John   25.0   50000
# 1   Jane    NaN   60000  <- Missing age
# 2   Bob    35.0     NaN  <- Missing salary
# 3    NaN   28.0   45000  <- Missing name

missing_counts = df.isnull().sum()
# Output:
# name      1
# age       1
# salary    1

missing_percent = (df.isnull().sum() / len(df)) * 100
# Output:
# name      25.0%
# age       25.0%
# salary    25.0%

len(df)  # Output: Original rows: 4
df_no_missing = df.dropna()
len(df_no_missing)
# Output: After removing rows with missing values: 1

df_clean = df.dropna(subset=['salary'])
len(df_clean)
# Output: Rows after removing missing salaries: 3

df['age'] = df['age'].fillna(0)
df['name'] = df['name'].fillna('Unknown')

# Output:
#      name   age  salary
# 0    John  25.0   50000
# 1    Jane   0.0   60000  <- Was NaN, now 0
# 2     Bob  35.0     NaN
# 3  Unknown 28.0   45000  <- Was NaN, now 'Unknown'
```

### Basic Text Cleaning

```python
# Example: Let's say we have messy names like this:
# Before: ['  john doe  ', 'JANE SMITH', 'bob-johnson', '  ALICE  ']

df['name'].tolist()
# Output: ['  john doe  ', 'JANE SMITH', 'bob-johnson', '  ALICE  ']
df['name'] = df['name'].str.strip()
df['name'].tolist()
# Output: ['john doe', 'JANE SMITH', 'bob-johnson', 'ALICE']

df['name'] = df['name'].str.lower()
df['name'].tolist()
# Output: ['john doe', 'jane smith', 'bob-johnson', 'alice']

df['name'] = df['name'].str.title()
df['name'].tolist()
# Output: ['John Doe', 'Jane Smith', 'Bob-Johnson', 'Alice']

df['name'] = df['name'].str.replace('-', ' ')
df['name'].tolist()
# Output: ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice']

# Before: ['John  Doe', 'Jane    Smith', 'Bob     Johnson']
df['description'] = df['description'].str.replace(r'\s+', ' ', regex=True)
# After: ['John Doe', 'Jane Smith', 'Bob Johnson']
df['name'].tolist()
```

### Duplicate Handling

```python
# Example dataset with duplicates:
# Row 0: ['John', 25, 50000]
# Row 1: ['Jane', 30, 60000]
# Row 2: ['John', 25, 50000]  <- This is a duplicate of Row 0
# Row 3: ['Bob', 35, 70000]

duplicate_count = df.duplicated().sum()
# Output: Number of duplicate rows: 1

duplicates = df[df.duplicated(keep=False)]
# Output shows both Row 0 and Row 2 (both copies of the same data)

len(df)
# Output: Before removing duplicates: 4
df_clean = df.drop_duplicates()
len(df_clean)
# Output: After removing duplicates: 3

df_clean = df.drop_duplicates(subset=['name', 'email'])

df_clean = df.drop_duplicates(keep='last')
```

## Simple Statistics

### Basic Descriptive Statistics

```python
df.describe()

if 'age' in df.columns:
    df['age'].mean()
    df['age'].median()
    df['age'].mode().iloc[0] if not df['age'].mode().empty else 'No mode'
    df['age'].min()
    df['age'].max()
    df['age'].std()
    df['age'].max() - df['age'].min()
```

### Percentiles and Quartiles

```python
if 'salary' in df.columns:
    df['salary'].quantile(0.25)
    df['salary'].quantile(0.5)
    df['salary'].quantile(0.75)
    df['salary'].quantile(0.9)
    df['salary'].quantile(0.95)

percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
df['salary'].quantile(percentiles)
```

### Counting and Frequencies

```python
df.count()

if 'department' in df.columns:
    df['department'].value_counts()
    df['department'].value_counts(normalize=True) * 100

for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
```

## Value Counting and Frequency

### Simple Value Counts

```python
# Example: Let's say we have a 'department' column with these values:
# ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Marketing', 'IT']

counts = df['department'].value_counts()
# Output:
# IT          4
# HR          2
# Marketing   1
# Finance     1

percentages = df['department'].value_counts(normalize=True) * 100
# Output:
# IT          50.0%
# HR          25.0%
# Marketing   12.5%
# Finance     12.5%

counts_with_missing = df['department'].value_counts(dropna=False)
# Output:
# IT          4
# HR          2
# NaN         1  <- This shows missing values too
# Marketing   1
# Finance     1
```

### Frequency Analysis

```python
if 'product' in df.columns:
    most_common = df['product'].value_counts().head(5)
    least_common = df['product'].value_counts().tail(5)

if 'age' in df.columns:
    age_bins = pd.cut(df['age'], bins=5)
    age_bins.value_counts().sort_index()
```

### Cross-tabulation (Simple)

```python
if 'gender' in df.columns and 'department' in df.columns:
    crosstab = pd.crosstab(df['gender'], df['department'])

crosstab_pct = pd.crosstab(df['gender'], df['department'], normalize='all') * 100
```

## Basic Sorting

### Single Column Sorting

```python
df_sorted = df.sort_values('age')
df_sorted = df.sort_values('age', ascending=False)

df_sorted = df.sort_values('salary', na_position='first')
df_sorted = df.sort_values('salary', na_position='last')
```

### Multiple Column Sorting

```python
df_sorted = df.sort_values(['department', 'salary'])
df_sorted = df.sort_values(['department', 'salary'], ascending=[True, False])

df_sorted = df.sort_index()
df_sorted = df.sort_index(axis=1)
```

### Top/Bottom Values

```python
if 'salary' in df.columns:
    top_earners = df.nlargest(5, 'salary')
    bottom_earners = df.nsmallest(5, 'salary')

    top_earners[['name', 'salary']]
    bottom_earners[['name', 'salary']]
```

## Simple Filtering

### Basic Filtering

```python
adults = df[df['age'] >= 18]
high_earners = df[df['salary'] > 50000]
specific_dept = df[df['department'] == 'IT']

if 'name' in df.columns:
    names_with_a = df[df['name'].str.contains('a', case=False, na=False)]
    names_starting_j = df[df['name'].str.startswith('J')]
    names_ending_son = df[df['name'].str.endswith('son')]
```

### Multiple Conditions

```python
young_high_earners = df[(df['age'] < 30) & (df['salary'] > 60000)]

senior_or_high_earners = df[(df['age'] > 50) | (df['salary'] > 80000)]

not_it_dept = df[df['department'] != 'IT']
not_missing_salary = df[df['salary'].notna()]
```

### Using isin() for Multiple Values

```python
if 'department' in df.columns:
    tech_depts = df[df['department'].isin(['IT', 'Engineering', 'Data Science'])]

non_tech = df[~df['department'].isin(['IT', 'Engineering'])]

age_range = df[df['age'].between(25, 35)]
```

## Basic Grouping

### Simple Grouping

```python
if 'department' in df.columns:
    dept_groups = df.groupby('department')

    dept_groups.size()

    dept_groups['salary'].mean()
```

### Basic Aggregations

```python
if 'department' in df.columns and 'salary' in df.columns:
    dept_stats = df.groupby('department').agg({
        'salary': ['mean', 'sum', 'count', 'min', 'max']
    })

summary = df.groupby('department').agg({
    'salary': 'mean',
    'age': 'mean',
    'employee_id': 'count'
}).round(2)
```

### Group Analysis

```python
for name, group in df.groupby('department'):
    print(f"\nDepartment: {name}")
    print(f"Size: {len(group)}")
    print(f"Average salary: {group['salary'].mean():.2f}")
    group.head(2)

if 'department' in df.columns:
    it_group = df.groupby('department').get_group('IT')
    it_group.head()
```
