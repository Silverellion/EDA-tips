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
# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)

# Plot settings
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")
```

## Data Import and Export

### Reading CSV Files

```python
# Basic CSV reading
df = pd.read_csv('data.csv')

# With different separators
df = pd.read_csv('data.csv', sep=';')
df = pd.read_csv('data.csv', sep='\t')  # Tab separated

# Handling headers
df = pd.read_csv('data.csv', header=0)     # First row as header
df = pd.read_csv('data.csv', header=None)  # No header
df = pd.read_csv('data.csv', names=['col1', 'col2', 'col3'])  # Custom headers

# Skipping rows
df = pd.read_csv('data.csv', skiprows=1)      # Skip first row
df = pd.read_csv('data.csv', skiprows=[0,2])  # Skip specific rows

# Reading specific columns
df = pd.read_csv('data.csv', usecols=['name', 'age', 'salary'])
df = pd.read_csv('data.csv', usecols=[0, 1, 3])  # By position

# Limiting rows
df = pd.read_csv('data.csv', nrows=1000)  # Read only first 1000 rows
```

### Reading Excel Files

```python
# Basic Excel reading
df = pd.read_excel('data.xlsx')

# Specific sheet
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)  # By index

# Multiple sheets
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)
```

### Reading JSON Files

```python
# JSON files
df = pd.read_json('data.json')

# Different orientations
df = pd.read_json('data.json', orient='records')
df = pd.read_json('data.json', orient='index')
```

### Saving Data

```python
# Save to CSV
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', sep=';', index=False)

# Save to Excel
df.to_excel('output.xlsx', index=False)
df.to_excel('output.xlsx', sheet_name='MyData', index=False)

# Save to JSON
df.to_json('output.json', orient='records')
```

## Initial Data Inspection

### Basic Dataset Information

```python
# Dataset dimensions
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Shape:", df.shape)

# Column names
print("Column names:", df.columns.tolist())
print("Number of columns:", len(df.columns))

# Index information
print("Index:", df.index)
print("Index name:", df.index.name)
```

### First Look at Data

```python
# First few rows
print(df.head())      # Default 5 rows
print(df.head(10))    # First 10 rows

# Last few rows
print(df.tail())      # Default 5 rows
print(df.tail(3))     # Last 3 rows

# Random sample
print(df.sample())    # 1 random row
print(df.sample(5))   # 5 random rows
print(df.sample(frac=0.1))  # 10% of data
```

### Data Overview

```python
# General information
df.info()

# Memory usage
print("Memory usage:")
print(df.memory_usage(deep=True))
print("Total memory:", df.memory_usage(deep=True).sum(), "bytes")

# Data types
print("Data types:")
print(df.dtypes)
```

## Understanding Data Structure

### Column Analysis

```python
# Get all column names
columns = df.columns.tolist()
print("All columns:", columns)

# Count columns by type
print("Numeric columns:", len(df.select_dtypes(include=[np.number]).columns))
print("Text columns:", len(df.select_dtypes(include=['object']).columns))
print("Date columns:", len(df.select_dtypes(include=['datetime']).columns))

# Check for duplicate column names
duplicate_cols = df.columns[df.columns.duplicated()].tolist()
print("Duplicate columns:", duplicate_cols)
```

### Index Analysis

```python
# Index information
print("Index type:", type(df.index))
print("Index is unique:", df.index.is_unique)
print("Index has duplicates:", df.index.duplicated().any())

# Check if index is meaningful
print("Index values sample:", df.index[:5].tolist())

# Reset index if needed
df_reset = df.reset_index()
print("After reset:", df_reset.head())
```

### Basic Content Check

```python
# Check if dataframe is empty
print("Is empty:", df.empty)

# Check for completely null columns
null_cols = df.columns[df.isnull().all()].tolist()
print("Completely null columns:", null_cols)

# Check for completely null rows
null_rows = df.index[df.isnull().all(axis=1)].tolist()
print("Completely null rows:", null_rows)
```

## Data Types and Conversion

### Understanding Current Types

```python
# Detailed type information
for col in df.columns:
    print(f"{col}: {df[col].dtype}")

# Group by data type
print("\nColumns by type:")
for dtype in df.dtypes.unique():
    cols = df.select_dtypes(include=[dtype]).columns.tolist()
    print(f"{dtype}: {cols}")
```

### Converting Data Types

```python
# Convert to numeric
df['age'] = pd.to_numeric(df['age'])
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')  # NaN for invalid

# Convert to string
df['id'] = df['id'].astype(str)

# Convert to integer
df['count'] = df['count'].astype(int)

# Convert to float
df['price'] = df['price'].astype(float)

# Convert to category
df['grade'] = df['grade'].astype('category')

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
```

### Checking Conversion Success

```python
# Before and after comparison
print("Before conversion:")
print(df.dtypes)

# After conversion
df['column'] = df['column'].astype('desired_type')
print("After conversion:")
print(df.dtypes)

# Check for conversion errors
def check_conversion_errors(series, target_type):
    if target_type == 'numeric':
        converted = pd.to_numeric(series, errors='coerce')
        errors = series[converted.isna() & series.notna()]
        return errors
    return None

errors = check_conversion_errors(df['column'], 'numeric')
print("Conversion errors:", errors.tolist())
```

## Basic Data Selection

### Selecting Columns

```python
# Single column
name_column = df['name']
age_column = df.age  # Dot notation (if no spaces in name)

# Multiple columns
subset = df[['name', 'age', 'salary']]

# All columns except some
all_except = df.drop(['unwanted_col'], axis=1)
all_except = df.drop(columns=['unwanted_col'])

# Select by position
first_three = df.iloc[:, :3]  # First 3 columns
last_two = df.iloc[:, -2:]    # Last 2 columns
```

### Selecting Rows

```python
# By position
first_row = df.iloc[0]        # First row
first_five = df.iloc[:5]      # First 5 rows
last_row = df.iloc[-1]        # Last row
specific_rows = df.iloc[[0, 2, 4]]  # Specific positions

# By index value
if 'id' in df.columns and df.set_index('id', inplace=False) is not None:
    df_indexed = df.set_index('id')
    row_by_id = df_indexed.loc[123]  # Row with id=123
```

### Selecting Both Rows and Columns

```python
# Specific cell
cell_value = df.iloc[0, 1]    # Row 0, Column 1
cell_value = df.at[0, 'name'] # Row 0, 'name' column

# Subset of rows and columns
subset = df.iloc[0:5, 1:4]    # First 5 rows, columns 1-3
subset = df.loc[0:4, ['name', 'age']]  # Rows 0-4, specific columns
```

## Data Sampling

### Simple Sampling

```python
# Random sample
sample_5 = df.sample(5)              # 5 random rows
sample_10pct = df.sample(frac=0.1)   # 10% of data
sample_with_replacement = df.sample(10, replace=True)

# Set seed for reproducibility
sample_reproducible = df.sample(5, random_state=42)
```

### Systematic Sampling

```python
# Every nth row
every_10th = df.iloc[::10]  # Every 10th row
every_5th = df.iloc[::5]    # Every 5th row

# First n rows from each group
if 'category' in df.columns:
    sample_per_group = df.groupby('category').head(5)
```

### Stratified Sampling

```python
# Sample maintaining proportions
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

# 1. Add a column with the same value for everyone
df['status'] = 'active'
print("After adding status column:")
print(df)
# Output:
#    name  age  status
# 0  John   25  active
# 1  Jane   30  active
# 2   Bob   35  active

# 2. Create a column by combining other columns
df['full_info'] = df['name'] + ' is ' + df['age'].astype(str) + ' years old'
print("After adding full_info column:")
print(df)
# Output:
#    name  age  status              full_info
# 0  John   25  active   John is 25 years old
# 1  Jane   30  active   Jane is 30 years old
# 2   Bob   35  active    Bob is 35 years old

# 3. Add a column with conditions (if-else logic)
df['age_group'] = np.where(df['age'] >= 30, 'Senior', 'Junior')
print("After adding age_group column:")
print(df)
# Output:
#    name  age  status              full_info age_group
# 0  John   25  active   John is 25 years old    Junior
# 1  Jane   30  active   Jane is 30 years old    Senior
# 2   Bob   35  active    Bob is 35 years old    Senior

# 4. Add a column with math calculations
df['birth_year'] = 2024 - df['age']
print("After adding birth_year column:")
print(df)
# Output:
#    name  age  status              full_info age_group  birth_year
# 0  John   25  active   John is 25 years old    Junior        1999
# 1  Jane   30  active   Jane is 30 years old    Senior        1994
# 2   Bob   35  active    Bob is 35 years old    Senior        1989
```

### Renaming Columns

```python
# Rename specific columns
df = df.rename(columns={'old_name': 'new_name'})
df = df.rename(columns={'col1': 'column_1', 'col2': 'column_2'})

# Rename using a function
df.columns = df.columns.str.lower()              # Lowercase
df.columns = df.columns.str.replace(' ', '_')    # Replace spaces
df.columns = df.columns.str.strip()              # Remove whitespace

# Add prefix or suffix
df = df.add_prefix('prefix_')
df = df.add_suffix('_suffix')
```

### Reordering Columns

```python
# Specify new order
new_order = ['name', 'age', 'salary', 'department']
df = df[new_order]

# Move column to front
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index('important_column')))
df = df[cols]

# Sort columns alphabetically
df = df.reindex(sorted(df.columns), axis=1)
```

### Removing Columns

```python
# Drop single column
df = df.drop('unwanted_column', axis=1)
df = df.drop(columns=['unwanted_column'])

# Drop multiple columns
df = df.drop(['col1', 'col2'], axis=1)

# Drop by pattern
cols_to_drop = [col for col in df.columns if 'temp' in col.lower()]
df = df.drop(columns=cols_to_drop)

# Keep only specific columns
keep_cols = ['name', 'age', 'salary']
df = df[keep_cols]
```

## Row Operations

### Adding Rows

```python
# Add single row (dictionary)
new_row = {'name': 'John', 'age': 30, 'salary': 50000}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Add multiple rows
new_rows = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 35],
    'salary': [45000, 60000]
})
df = pd.concat([df, new_rows], ignore_index=True)

# Add row at specific position
# Insert at beginning
df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
```

### Removing Rows

```python
# Drop by index
df = df.drop(0)          # Drop first row
df = df.drop([0, 1, 2])  # Drop multiple rows

# Drop by condition
df = df[df['age'] >= 18]  # Keep only adults
df = df[df['salary'] > 0] # Remove zero salaries

# Drop duplicates
df = df.drop_duplicates()                    # All columns
df = df.drop_duplicates(subset=['name'])     # Based on specific columns
df = df.drop_duplicates(keep='first')        # Keep first occurrence
df = df.drop_duplicates(keep='last')         # Keep last occurrence
```

### Reordering Rows

```python
# Sort by index
df = df.sort_index()

# Reset index
df = df.reset_index(drop=True)

# Reverse order
df = df.iloc[::-1]

# Random shuffle
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

# 1. Check how many missing values in each column
missing_counts = df.isnull().sum()
print("Missing values per column:")
print(missing_counts)
# Output:
# name      1
# age       1
# salary    1

# 2. Check percentage of missing values
missing_percent = (df.isnull().sum() / len(df)) * 100
print("Missing value percentage:")
print(missing_percent)
# Output:
# name      25.0%
# age       25.0%
# salary    25.0%

# 3. Remove rows with ANY missing values
print("Original rows:", len(df))  # Output: Original rows: 4
df_no_missing = df.dropna()
print("After removing rows with missing values:", len(df_no_missing))
# Output: After removing rows with missing values: 1

# 4. Remove rows only if specific column is missing
df_clean = df.dropna(subset=['salary'])  # Only remove if salary is missing
print("Rows after removing missing salaries:", len(df_clean))
# Output: Rows after removing missing salaries: 3

# 5. Fill missing values with specific values
df['age'] = df['age'].fillna(0)           # Replace missing age with 0
df['name'] = df['name'].fillna('Unknown') # Replace missing name with 'Unknown'

print("After filling missing values:")
print(df)
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

# 1. Remove whitespace from beginning and end
print("Before strip:", df['name'].tolist())
# Output: ['  john doe  ', 'JANE SMITH', 'bob-johnson', '  ALICE  ']
df['name'] = df['name'].str.strip()
print("After strip:", df['name'].tolist())
# Output: ['john doe', 'JANE SMITH', 'bob-johnson', 'ALICE']

# 2. Make everything lowercase
df['name'] = df['name'].str.lower()
print("After lower:", df['name'].tolist())
# Output: ['john doe', 'jane smith', 'bob-johnson', 'alice']

# 3. Make first letter of each word uppercase (Title Case)
df['name'] = df['name'].str.title()
print("After title:", df['name'].tolist())
# Output: ['John Doe', 'Jane Smith', 'Bob-Johnson', 'Alice']

# 4. Replace dashes with spaces
df['name'] = df['name'].str.replace('-', ' ')
print("After replace dash:", df['name'].tolist())
# Output: ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice']

# 5. Remove extra spaces between words
# Before: ['John  Doe', 'Jane    Smith', 'Bob     Johnson']
df['description'] = df['description'].str.replace(r'\s+', ' ', regex=True)
# After: ['John Doe', 'Jane Smith', 'Bob Johnson']
print("Final result:", df['name'].tolist())
```

### Duplicate Handling

```python
# Example dataset with duplicates:
# Row 0: ['John', 25, 50000]
# Row 1: ['Jane', 30, 60000]
# Row 2: ['John', 25, 50000]  <- This is a duplicate of Row 0
# Row 3: ['Bob', 35, 70000]

# 1. Find how many duplicate rows exist
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")
# Output: Number of duplicate rows: 1

# 2. See which rows are duplicates (shows all copies)
duplicates = df[df.duplicated(keep=False)]
print("All duplicate rows:")
print(duplicates)
# Output shows both Row 0 and Row 2 (both copies of the same data)

# 3. Remove duplicate rows (keeps first occurrence)
print("Before removing duplicates:", len(df))
# Output: Before removing duplicates: 4
df_clean = df.drop_duplicates()
print("After removing duplicates:", len(df_clean))
# Output: After removing duplicates: 3

# 4. Remove duplicates based only on specific columns
# This removes rows where 'name' and 'email' are the same
df_clean = df.drop_duplicates(subset=['name', 'email'])

# 5. Keep last occurrence instead of first
df_clean = df.drop_duplicates(keep='last')
```

## Simple Statistics

### Basic Descriptive Statistics

```python
# For numeric columns
print("Basic statistics:")
print(df.describe())

# Individual statistics
if 'age' in df.columns:
    print("Age statistics:")
    print(f"Mean: {df['age'].mean():.2f}")
    print(f"Median: {df['age'].median():.2f}")
    print(f"Mode: {df['age'].mode().iloc[0] if not df['age'].mode().empty else 'No mode'}")
    print(f"Min: {df['age'].min()}")
    print(f"Max: {df['age'].max()}")
    print(f"Standard deviation: {df['age'].std():.2f}")
    print(f"Range: {df['age'].max() - df['age'].min()}")
```

### Percentiles and Quartiles

```python
# Percentiles
if 'salary' in df.columns:
    print("Salary percentiles:")
    print(f"25th percentile: {df['salary'].quantile(0.25):.2f}")
    print(f"50th percentile (median): {df['salary'].quantile(0.5):.2f}")
    print(f"75th percentile: {df['salary'].quantile(0.75):.2f}")
    print(f"90th percentile: {df['salary'].quantile(0.9):.2f}")
    print(f"95th percentile: {df['salary'].quantile(0.95):.2f}")

# Custom percentiles
percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
print("Custom percentiles:")
print(df['salary'].quantile(percentiles))
```

### Counting and Frequencies

```python
# Count non-null values
print("Non-null counts:")
print(df.count())

# Unique value counts
if 'department' in df.columns:
    print("Department counts:")
    print(df['department'].value_counts())
    print("\nDepartment percentages:")
    print(df['department'].value_counts(normalize=True) * 100)

# Number of unique values
print("Unique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
```

## Value Counting and Frequency

### Simple Value Counts

```python
# Example: Let's say we have a 'department' column with these values:
# ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Marketing', 'IT']

# 1. Count how many times each value appears
counts = df['department'].value_counts()
print("Department counts:")
print(counts)
# Output:
# IT          4
# HR          2
# Marketing   1
# Finance     1

# 2. Show as percentages instead of counts
percentages = df['department'].value_counts(normalize=True) * 100
print("Department percentages:")
print(percentages)
# Output:
# IT          50.0%
# HR          25.0%
# Marketing   12.5%
# Finance     12.5%

# 3. Include missing values in the count
# If some rows have NaN in department column
counts_with_missing = df['department'].value_counts(dropna=False)
print("Counts including missing values:")
print(counts_with_missing)
# Output:
# IT          4
# HR          2
# NaN         1  <- This shows missing values too
# Marketing   1
# Finance     1
```

### Frequency Analysis

```python
# Most and least common values
if 'product' in df.columns:
    most_common = df['product'].value_counts().head(5)
    least_common = df['product'].value_counts().tail(5)

    print("Top 5 products:")
    print(most_common)
    print("\nBottom 5 products:")
    print(least_common)

# Frequency ranges for numeric data
if 'age' in df.columns:
    age_bins = pd.cut(df['age'], bins=5)
    print("Age ranges:")
    print(age_bins.value_counts().sort_index())
```

### Cross-tabulation (Simple)

```python
# Two-way frequency table
if 'gender' in df.columns and 'department' in df.columns:
    crosstab = pd.crosstab(df['gender'], df['department'])
    print("Gender vs Department:")
    print(crosstab)

# With percentages
crosstab_pct = pd.crosstab(df['gender'], df['department'], normalize='all') * 100
print("Percentages:")
print(crosstab_pct)
```

## Basic Sorting

### Single Column Sorting

```python
# Sort by one column
df_sorted = df.sort_values('age')                    # Ascending
df_sorted = df.sort_values('age', ascending=False)   # Descending

# Sort with missing values
df_sorted = df.sort_values('salary', na_position='first')  # NaN first
df_sorted = df.sort_values('salary', na_position='last')   # NaN last
```

### Multiple Column Sorting

```python
# Sort by multiple columns
df_sorted = df.sort_values(['department', 'salary'])
df_sorted = df.sort_values(['department', 'salary'], ascending=[True, False])

# Sort by index
df_sorted = df.sort_index()                    # Sort by row index
df_sorted = df.sort_index(axis=1)              # Sort by column names
```

### Top/Bottom Values

```python
# Get top/bottom records
if 'salary' in df.columns:
    top_earners = df.nlargest(5, 'salary')     # Top 5 salaries
    bottom_earners = df.nsmallest(5, 'salary') # Bottom 5 salaries

    print("Top earners:")
    print(top_earners[['name', 'salary']])
    print("\nLowest earners:")
    print(bottom_earners[['name', 'salary']])
```

## Simple Filtering

### Basic Filtering

```python
# Simple conditions
adults = df[df['age'] >= 18]
high_earners = df[df['salary'] > 50000]
specific_dept = df[df['department'] == 'IT']

# String filtering
if 'name' in df.columns:
    names_with_a = df[df['name'].str.contains('a', case=False, na=False)]
    names_starting_j = df[df['name'].str.startswith('J')]
    names_ending_son = df[df['name'].str.endswith('son')]
```

### Multiple Conditions

```python
# AND conditions
young_high_earners = df[(df['age'] < 30) & (df['salary'] > 60000)]

# OR conditions
senior_or_high_earners = df[(df['age'] > 50) | (df['salary'] > 80000)]

# NOT condition
not_it_dept = df[df['department'] != 'IT']
not_missing_salary = df[df['salary'].notna()]
```

### Using isin() for Multiple Values

```python
# Filter for multiple values
if 'department' in df.columns:
    tech_depts = df[df['department'].isin(['IT', 'Engineering', 'Data Science'])]

# Exclude multiple values
non_tech = df[~df['department'].isin(['IT', 'Engineering'])]

# Numeric ranges
age_range = df[df['age'].between(25, 35)]  # Ages 25-35 inclusive
```

## Basic Grouping

### Simple Grouping

```python
# Group by single column
if 'department' in df.columns:
    dept_groups = df.groupby('department')

    # Group sizes
    print("Group sizes:")
    print(dept_groups.size())

    # Group means
    print("Average salary by department:")
    print(dept_groups['salary'].mean())
```

### Basic Aggregations

```python
# Common aggregations
if 'department' in df.columns and 'salary' in df.columns:
    dept_stats = df.groupby('department').agg({
        'salary': ['mean', 'sum', 'count', 'min', 'max']
    })
    print("Department salary statistics:")
    print(dept_stats)

# Multiple columns
summary = df.groupby('department').agg({
    'salary': 'mean',
    'age': 'mean',
    'employee_id': 'count'
}).round(2)
```

### Group Analysis

```python
# Iterate through groups
for name, group in df.groupby('department'):
    print(f"\nDepartment: {name}")
    print(f"Size: {len(group)}")
    print(f"Average salary: {group['salary'].mean():.2f}")
    print("Sample:")
    print(group.head(2))

# Get specific group
if 'department' in df.columns:
    it_group = df.groupby('department').get_group('IT')
    print("IT Department:")
    print(it_group.head())
```
