# Seaborn Plotting Guide: When to Use Which Plot

This guide will help you decide which type of Seaborn plot to use based on your data visualization needs and analysis goals.

## Table of Contents

- [Relational Plots](#relational-plots)
- [Categorical Plots](#categorical-plots)
- [Distribution Plots](#distribution-plots)
- [Regression Plots](#regression-plots)
- [Matrix Plots](#matrix-plots)
- [Multi-Plot Strategies](#multi-plot-strategies)

## Relational Plots

### Scatterplot (`sns.scatterplot`)

**When to use:**

- Visualize relationship between two continuous variables
- Detect patterns, correlations, clusters, or outliers
- Examine how a third variable affects the relationship (using hue)
- Analyze multi-dimensional relationships (using size, style)

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='horsepower',
    y='price',
    hue='fuel_type',      # Color by categorical variable
    size='engine_size',   # Point size by variable
    style='body_style',   # Different markers by category
    palette='viridis',    # Color palette
    alpha=0.7,            # Transparency
    sizes=(20, 200),      # Range of point sizes
    data=df
)
plt.title('Car Price vs Horsepower')
plt.tight_layout()
plt.show()
```

### Lineplot (`sns.lineplot`)

**When to use:**

- Show trends over time or ordered categories
- Compare multiple time series or trends
- Visualize aggregated data with confidence intervals
- Display changes in a continuous variable across ordered categories

```python
# Group by make to get mean prices
grouped_data = df.groupby(['make'])['price'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(
    x='make',
    y='price',
    markers=True,         # Add markers at data points
    dashes=False,         # Use solid lines
    palette='deep',       # Color palette
    err_style='band',     # Confidence interval style
    ci=95,                # Confidence interval level
    data=grouped_data.sort_values('price')
)
plt.title('Average Car Price by Make')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

## Categorical Plots

### Barplot (`sns.barplot`)

**When to use:**

- Compare quantities across categories
- Show summary statistics (mean, median, etc.) with error bars
- Visualize categorical data with a single numeric variable
- Compare groups or subgroups (using hue)

```python
plt.figure(figsize=(12, 6))
sns.barplot(
    x='make',
    y='price',
    hue='fuel_type',      # Separate bars by category
    palette='pastel',     # Color palette
    errorbar=('ci', 95),  # Error bars with 95% CI
    capsize=0.2,          # Size of error bar caps
    alpha=0.8,            # Transparency
    data=df
)
plt.title('Average Car Price by Make and Fuel Type')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

### Countplot (`sns.countplot`)

**When to use:**

- Show frequency of categories
- Compare distribution across categories
- Display categorical data count
- Analyze class distribution or balance

```python
plt.figure(figsize=(12, 6))
sns.countplot(
    x='make',
    hue='fuel_type',     # Separate counts by category
    palette='Set2',      # Color palette
    alpha=0.8,           # Transparency
    saturation=0.7,      # Color saturation
    dodge=True,          # Whether to dodge grouped bars
    data=df
)
plt.title('Count of Cars by Make and Fuel Type')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

### Boxplot (`sns.boxplot`)

**When to use:**

- Show distribution of numerical data across categories
- Identify outliers within categories
- Compare distributions between groups
- Visualize median, quartiles, and range

```python
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='fuel_type',
    y='price',
    hue='drive_wheels',   # Separate boxes by category
    palette='Set3',       # Color palette
    width=0.6,            # Box width
    fliersize=5,          # Size of outlier points
    linewidth=1,          # Width of box lines
    dodge=True,           # Whether to dodge grouped boxes
    data=df
)
plt.title('Distribution of Car Prices by Fuel Type and Drive Wheels')
plt.tight_layout()
plt.show()
```

### Violinplot (`sns.violinplot`)

**When to use:**

- Show full distribution shape across categories
- Compare distribution density across groups
- Visualize bimodal or complex distributions
- Show more detail than a boxplot when sample size is large

```python
plt.figure(figsize=(10, 6))
sns.violinplot(
    x='fuel_type',
    y='price',
    hue='drive_wheels',   # Separate violins by category
    split=True,           # Split violins for better comparison
    inner='quartile',     # Inner visualization
    palette='muted',      # Color palette
    cut=0,                # How far to extend density past observed data
    scale='area',         # Scale violins ('area', 'count', or 'width')
    data=df
)
plt.title('Distribution of Car Prices by Fuel Type and Drive Wheels')
plt.tight_layout()
plt.show()
```

### Stripplot (`sns.stripplot`)

**When to use:**

- Show individual data points within categories
- Visualize raw data distribution across categories
- Analyze small to medium-sized datasets
- See actual data points rather than just summaries

```python
plt.figure(figsize=(12, 6))
sns.stripplot(
    x='body_style',
    y='price',
    hue='fuel_type',      # Color by category
    palette='Paired',     # Color palette
    size=5,               # Size of markers
    jitter=True,          # Add jitter to avoid overlaps
    alpha=0.6,            # Transparency
    dodge=True,           # Whether to dodge points by hue
    data=df
)
plt.title('Individual Car Prices by Body Style and Fuel Type')
plt.tight_layout()
plt.show()
```

### Swarmplot (`sns.swarmplot`)

**When to use:**

- Show individual data points without overlap
- Display all data points for small to medium datasets
- Visualize distribution shape and density
- See exact values while avoiding overplotting

```python
plt.figure(figsize=(12, 6))
sns.swarmplot(
    x='body_style',
    y='price',
    hue='fuel_type',      # Color by category
    palette='Set1',       # Color palette
    size=5,               # Size of markers
    dodge=True,           # Whether to dodge points by hue
    alpha=0.7,            # Transparency
    data=df
)
plt.title('Individual Car Prices by Body Style (Non-overlapping)')
plt.tight_layout()
plt.show()
```

## Distribution Plots

### Histplot (`sns.histplot`)

**When to use:**

- Visualize distribution of a single variable
- Examine frequency or density across values
- Check for normality, skewness, or multiple modes
- Compare distributions across categories (using hue)

```python
plt.figure(figsize=(10, 6))
sns.histplot(
    x='price',
    hue='fuel_type',      # Separate histograms by category
    kde=True,             # Add density curve
    bins=15,              # Number of bins
    palette='rocket',     # Color palette
    alpha=0.6,            # Transparency
    multiple='layer',     # How multiple distributions are shown
    stat='density',       # Statistic to compute
    data=df
)
plt.title('Distribution of Car Prices')
plt.tight_layout()
plt.show()
```

### KDEPlot (`sns.kdeplot`)

**When to use:**

- Visualize smooth density estimation
- Focus on distribution shape rather than counts
- Compare multiple distributions
- Create cleaner visualizations without bin edge artifacts

```python
plt.figure(figsize=(10, 6))
sns.kdeplot(
    x='horsepower',
    hue='fuel_type',      # Separate densities by category
    palette='coolwarm',   # Color palette
    fill=True,            # Fill the density curves
    alpha=0.5,            # Transparency
    bw_adjust=1,          # Bandwidth adjustment
    common_norm=False,    # Whether to normalize densities together
    cumulative=False,     # Whether to plot cumulative distribution
    multiple='layer',     # How multiple distributions are shown
    data=df
)
plt.title('Density of Horsepower by Fuel Type')
plt.tight_layout()
plt.show()
```

### ECDFPlot (`sns.ecdfplot`)

**When to use:**

- Visualize cumulative distribution
- Compare distributions without binning artifacts
- Check for stochastic dominance between groups
- Examine percentiles of data

```python
plt.figure(figsize=(10, 6))
sns.ecdfplot(
    x='price',
    hue='fuel_type',      # Separate ECDFs by category
    palette='crest',      # Color palette
    stat='proportion',    # Statistic ('proportion' or 'count')
    complementary=False,  # Whether to plot the complementary CDF
    data=df
)
plt.title('Cumulative Distribution of Car Prices by Fuel Type')
plt.tight_layout()
plt.show()
```

## Regression Plots

### Regplot (`sns.regplot`)

**When to use:**

- Show relationship with regression line
- Visualize linear correlation between two variables
- Focus on a single relationship without categories
- Add confidence intervals to regression

```python
plt.figure(figsize=(10, 6))
sns.regplot(
    x='horsepower',
    y='price',
    data=df,
    scatter_kws={'alpha': 0.5, 'color': 'blue'},  # Customize scatter plot
    line_kws={'color': 'red', 'linewidth': 2},    # Customize regression line
    marker='o',           # Point marker style
    fit_reg=True,         # Whether to plot regression line
    ci=95,                # Confidence interval for regression
    order=1,              # Polynomial order for regression
    robust=False,         # Whether to use robust regression
    logx=False,           # Whether to log-transform x axis
)
plt.title('Regression of Price vs Horsepower')
plt.tight_layout()
plt.show()
```

### Lmplot (`sns.lmplot`)

**When to use:**

- Create facetted regression plots
- Compare relationships across different subgroups
- Visualize how a relationship varies across categories
- Create complex multi-panel regression visualizations

```python
sns.lmplot(
    x='horsepower',
    y='price',
    hue='fuel_type',      # Separate by category
    col='body_style',     # Create column facets by category
    palette='Spectral',   # Color palette
    height=4,             # Height of each facet
    aspect=1,             # Aspect ratio of each facet
    markers=['o', 's'],   # Different markers for categories
    scatter_kws={'alpha': 0.5},  # Customize scatter plot
    line_kws={'linewidth': 2},   # Customize regression line
    col_wrap=3,           # Number of facet columns
    data=df
)
plt.suptitle('Regression of Price vs Horsepower Across Categories', y=1.05)
plt.tight_layout()
plt.show()
```

## Matrix Plots

### Heatmap (`sns.heatmap`)

**When to use:**

- Visualize correlation matrices
- Show patterns in 2D data
- Display values in a color-encoded matrix
- Highlight relationships between many variables at once

```python
# Create correlation matrix
corr_matrix = df.select_dtypes(include=['number']).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,           # Show values in cells
    cmap='YlGnBu',        # Color map
    fmt='.2f',            # Format for annotations
    linewidths=0.5,       # Width of cell borders
    cbar=True,            # Whether to draw a colorbar
    cbar_kws={'label': 'Correlation Coefficient'},  # Colorbar customization
    center=0,             # Value at center of colormap
    square=True,          # Force cells to be square
)
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.show()
```

### Clustermap (`sns.clustermap`)

**When to use:**

- Perform hierarchical clustering on a matrix
- Discover patterns and similarities in complex data
- Group related variables or observations
- Create a reordered correlation matrix based on similarity

```python
# Create a pivot table for demonstration
pivot_data = pd.pivot_table(
    df,
    values='price',
    index=['body_style'],
    columns=['fuel_type'],
    aggfunc='mean'
)

# Fill any NaN values
pivot_data = pivot_data.fillna(0)

sns.clustermap(
    pivot_data,
    cmap='viridis',       # Color map
    standard_scale=0,     # Standardize data (0=rows, 1=columns, None=neither)
    col_cluster=True,     # Whether to cluster columns
    row_cluster=True,     # Whether to cluster rows
    figsize=(12, 10),     # Figure size
    annot=True,           # Show values in cells
    fmt='.0f',            # Format for annotations
    cbar_kws={'label': 'Average Price'},  # Colorbar customization
)
plt.suptitle('Clustered Heatmap of Car Prices by Body Style and Fuel Type', y=1.02)
plt.show()
```

## Multi-Plot Strategies

### Using FacetGrid

**When to use:**

- Create multi-panel plots with the same type of visualization
- Compare the same relationship across different subsets of data
- Control layout of subplots based on categorical variables
- Apply the same plotting function to different data subsets

```python
g = sns.FacetGrid(
    df,
    col="body_style",
    row="fuel_type",
    height=4,
    aspect=1.5,
    sharex=True,
    sharey=True
)
g.map(sns.scatterplot, "horsepower", "price", alpha=0.7)
g.add_legend()
g.fig.suptitle('Price vs Horsepower by Body Style and Fuel Type', y=1.02)
plt.tight_layout()
plt.show()
```

### Using subplots

**When to use:**

- Create a dashboard of different plot types
- Visualize multiple aspects of the data in one figure
- Combine different visualization types to tell a complete story
- Compare different variables or relationships side by side

```python
# Create a figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Boxplot
sns.boxplot(x='fuel_type', y='price', hue='drive_wheels', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Price Distribution by Fuel Type')
axes[0, 0].tick_params(axis='x', rotation=45)

# Scatterplot
sns.scatterplot(x='horsepower', y='price', hue='body_style', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Price vs Horsepower')

# Barplot
sns.barplot(x='body_style', y='highway_mpg', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Average Highway MPG by Body Style')
axes[1, 0].tick_params(axis='x', rotation=45)

# Histogram
sns.histplot(x='price', hue='fuel_type', kde=True, data=df, ax=axes[1, 1])
axes[1, 1].set_title('Price Distribution by Fuel Type')

plt.tight_layout()
plt.suptitle('Multi-faceted Analysis of Car Data', y=1.02, fontsize=16)
plt.show()
```

### PairPlot (`sns.pairplot`)

**When to use:**

- Explore relationships between multiple variables at once
- Create a matrix of scatterplots for all variable combinations
- See correlations and distributions across many variables
- Get a comprehensive overview of your dataset

```python
# Select a subset of numeric columns for clarity
cols = ['price', 'horsepower', 'city_mpg', 'highway_mpg', 'curb_weight']

sns.pairplot(
    df[cols + ['fuel_type']],
    hue='fuel_type',      # Color by category
    palette='tab10',      # Color palette
    diag_kind='kde',      # Type of plot on diagonal ('hist' or 'kde')
    markers=['o', 's'],   # Marker style by hue
    corner=True,          # Show only the lower triangle
    plot_kws={'alpha': 0.5}  # Kwargs for the underlying plot
)
plt.suptitle('Relationships Between Car Features', y=1.02)
plt.tight_layout()
plt.show()
```

## Quick Selection Guide

### For Relationships Between Variables

- Two continuous variables: **Scatterplot** or **Regplot**
- Time series or ordered categories: **Lineplot**
- Multiple pairs of variables: **Pairplot**

### For Categorical Comparisons

- Compare quantities: **Barplot**
- Compare frequencies: **Countplot**
- Compare distributions: **Boxplot** or **Violinplot**
- Show all data points: **Stripplot** or **Swarmplot**

### For Distributions

- Single variable distribution: **Histplot** or **KDEplot**
- Cumulative distribution: **ECDFplot**

### For Complex Patterns

- Correlation or matrix data: **Heatmap**
- Clustered data relationships: **Clustermap**

### For Multi-panel Analysis

- Same plot type across categories: **FacetGrid**
- Different plot types in one view: **Subplots**
