"""
Student Performance Prediction - Exploratory Data Analysis (EDA)
================================================================
This script performs EDA on the student performance dataset.

Visualizations Created:
1. Correlation Heatmap - Shows relationships between all numeric variables
2. HSC Percentage vs CGPA Scatter Plot - Shows key predictor relationship  
3. CGPA Distribution by Gender (Box Plot) - Shows demographic patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data(filepath):
    """
    Load the student performance dataset from Excel file.
    
    Parameters:
    -----------
    filepath : str
        Path to the Excel file
    
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    df = pd.read_excel(filepath)
    return df

def display_basic_info(df):
    """
    Display basic information about the dataset.
    
    This helps understand:
    - Dataset size (rows x columns)
    - Data types of each column
    - Missing values
    - Basic statistics
    """
    print("=" * 60)
    print("STUDENT PERFORMANCE DATASET - BASIC INFORMATION")
    print("=" * 60)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Column Information:")
    print("-" * 40)
    for col in df.columns:
        print(f"  • {col}: {df[col].dtype}")
    
    print("\n❓ Missing Values:")
    print("-" * 40)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✅ No missing values found!")
    else:
        print(missing[missing > 0])
    
    print("\n📈 Statistical Summary:")
    print("-" * 40)
    print(df.describe().round(2))
    
    print("\n🎯 Target Variable (CGPA) Distribution:")
    print("-" * 40)
    print(f"  • Mean CGPA: {df['CGPA'].mean():.2f}")
    print(f"  • Median CGPA: {df['CGPA'].median():.2f}")
    print(f"  • Min CGPA: {df['CGPA'].min():.2f}")
    print(f"  • Max CGPA: {df['CGPA'].max():.2f}")
    print(f"  • Std Dev: {df['CGPA'].std():.2f}")

def plot_correlation_heatmap(df, save_path=None):
    """
    VISUALIZATION 1: Correlation Heatmap
    """
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=True,           # Show correlation values
        cmap='RdYlBu_r',      # Red-Yellow-Blue color scheme
        center=0,             # Center the colormap at 0
        fmt='.2f',            # 2 decimal places
        square=True,          # Square cells
        linewidths=0.5,       # Line width between cells
        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'}
    )
    
    plt.title('Correlation Heatmap - Student Performance Variables', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Correlation heatmap saved to: {save_path}")
    
    plt.show()
    
    # Print key insights
    print("\n🔍 Key Correlation Insights:")
    print("-" * 40)
    cgpa_corr = correlation_matrix['CGPA'].drop('CGPA').sort_values(ascending=False)
    for feature, corr in cgpa_corr.items():
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        print(f"  • {feature} vs CGPA: {corr:.3f} ({strength})")

def plot_hsc_vs_cgpa(df, save_path=None):
    """
    VISUALIZATION 2: HSC Percentage vs CGPA Scatter Plot
    
    Purpose:
    --------
    Shows the relationship between HSC (Higher Secondary Certificate) 
    percentage and CGPA. This is typically the strongest predictor.
    
    """
    plt.figure(figsize=(10, 7))
    
    # Create scatter plot with regression line
    scatter = sns.regplot(
        data=df,
        x='HSC Percentage',
        y='CGPA',
        scatter_kws={'alpha': 0.6, 's': 50, 'color': '#3498db'},
        line_kws={'color': '#e74c3c', 'linewidth': 2}
    )
    
    # Calculate correlation for annotation
    correlation = df['HSC Percentage'].corr(df['CGPA'])
    
    # Add correlation text
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('HSC Percentage (%)', fontsize=12)
    plt.ylabel('CGPA', fontsize=12)
    plt.title('HSC Percentage vs CGPA - Key Predictor Relationship', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ HSC vs CGPA plot saved to: {save_path}")
    
    plt.show()
    
    print("\n🔍 Insight:")
    print(f"  • Correlation coefficient: {correlation:.3f}")
    print(f"  • This {'strong' if correlation > 0.5 else 'moderate'} positive correlation")
    print(f"    indicates HSC Percentage is a good predictor of CGPA.")

def plot_cgpa_by_gender(df, save_path=None):
    """
    VISUALIZATION 3: CGPA Distribution by Gender (Box Plot)
    
    Purpose:
    --------
    Shows how CGPA is distributed across different genders.
    Helps identify if there are any demographic patterns.
    
    """
    plt.figure(figsize=(10, 7))
    
    # Create box plot
    box = sns.boxplot(
        data=df,
        x='Gender',
        y='CGPA',
        palette=['#FF6B6B', '#4ECDC4'],  # Custom colors
        width=0.5
    )
    
    # Add individual data points
    sns.stripplot(
        data=df,
        x='Gender',
        y='CGPA',
        color='black',
        alpha=0.3,
        size=4
    )
    
    # Calculate and display statistics
    gender_stats = df.groupby('Gender')['CGPA'].agg(['mean', 'median', 'std', 'count'])
    
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('CGPA', fontsize=12)
    plt.title('CGPA Distribution by Gender', fontsize=14, fontweight='bold')
    
    # Add mean annotations
    for i, gender in enumerate(df['Gender'].unique()):
        mean_val = df[df['Gender'] == gender]['CGPA'].mean()
        plt.annotate(f'Mean: {mean_val:.2f}', 
                    xy=(i, mean_val), 
                    xytext=(i + 0.2, mean_val + 0.1),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gender boxplot saved to: {save_path}")
    
    plt.show()
    
    print("\n🔍 Gender-wise Statistics:")
    print("-" * 40)
    print(gender_stats.round(2))

def run_full_eda(filepath):
    """
    Run the complete EDA pipeline.
    
    This function:
    1. Loads the data
    2. Displays basic information
    3. Creates all 3 EDA visualizations
    """
    print("\n" + "=" * 60)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # Load data
    df = load_data(filepath)
    
    # Display basic info
    display_basic_info(df)
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    print("\n📊 Visualization 1: Correlation Heatmap")
    plot_correlation_heatmap(df, 'correlation_heatmap.png')
    
    print("\n📊 Visualization 2: HSC vs CGPA Scatter Plot")
    plot_hsc_vs_cgpa(df, 'hsc_vs_cgpa.png')
    
    print("\n📊 Visualization 3: CGPA by Gender Box Plot")
    plot_cgpa_by_gender(df, 'cgpa_by_gender.png')
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE!")
    print("=" * 60)
    
    return df

# Main execution
if __name__ == "__main__":
    # Path to dataset
    DATA_PATH = "student performance dataset.xlsx"
    
    # Run EDA
    df = run_full_eda(DATA_PATH)
