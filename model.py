"""
Student Performance Prediction - Model Training & Evaluation
============================================================
This script trains a Random Forest model to predict student CGPA
and evaluates its performance.

Visualizations Created:
4. Feature Importance Bar Chart (Model Training)
5. Actual vs Predicted CGPA Plot (Model Evaluation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset for machine learning.
    
    Steps:
    1. Load the Excel file
    2. Encode categorical variables (Gender)
    3. Select features and target
    
    For Viva:
    ---------
    Q: Why do we need to encode categorical variables?
    A: Machine learning algorithms work with numbers, not text.
       Label encoding converts categories like 'Male'/'Female' 
       into numbers like 0/1.
    """
    print("📂 Loading and preprocessing data...")
    
    # Load data
    df = pd.read_excel(filepath)
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Encode Gender (categorical to numerical)
    label_encoder = LabelEncoder()
    df_processed['Gender_Encoded'] = label_encoder.fit_transform(df_processed['Gender'])
    
    print(f"  • Gender encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Select features (X) and target (y)
    # We exclude 'ID' as it's just an identifier, not a predictor
    feature_columns = ['Gender_Encoded', 'Program Duration', 'Year of Student', 'HSC Percentage']
    
    X = df_processed[feature_columns]
    y = df_processed['CGPA']
    
    print(f"  • Features selected: {feature_columns}")
    print(f"  • Target variable: CGPA")
    print(f"  • Dataset size: {len(X)} samples")
    
    return X, y, feature_columns, label_encoder

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    For Viva:
    ---------
    Q: Why do we split data into training and testing sets?
    A: To evaluate how well our model performs on unseen data.
       - Training set (80%): Used to train the model
       - Testing set (20%): Used to evaluate model performance
       This helps detect overfitting (model memorizing training data).
    
    Q: What is random_state?
    A: It's a seed for reproducibility. Using the same random_state
       ensures we get the same split every time we run the code.
    """
    print(f"\n📊 Splitting data (Test size: {test_size*100}%)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    
    For Viva:
    ---------
    Q: Why Random Forest?
    A: Random Forest is chosen because:
       1. Handles both numerical and categorical features well
       2. Provides feature importance (which features matter most)
       3. Robust to outliers
       4. Reduces overfitting through ensemble of trees
       5. Good performance on medium-sized datasets
    
    Q: What is n_estimators?
    A: Number of decision trees in the forest. More trees = 
       more accurate but slower. 100 is a good default.
    
    Q: What is max_depth?
    A: Maximum depth of each tree. Limits how complex each tree can be.
       Helps prevent overfitting.
    """
    print("\n🤖 Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=100,      # 100 decision trees
        max_depth=10,          # Maximum depth to prevent overfitting
        min_samples_split=5,   # Minimum samples to split a node
        min_samples_leaf=2,    # Minimum samples in leaf node
        random_state=42        # For reproducibility
    )
    
    model.fit(X_train, y_train)
    
    print("  ✅ Model training complete!")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance using multiple metrics.
    
    Metrics Explained (For Viva):
    -----------------------------
    
    1. R² Score (Coefficient of Determination):
       - Range: 0 to 1 (can be negative for very poor models)
       - Interpretation: Proportion of variance explained by the model
       - Example: R² = 0.85 means model explains 85% of CGPA variance
       - Higher is better
    
    2. MAE (Mean Absolute Error):
       - Average of absolute differences between predicted and actual values
       - In same units as target (CGPA points)
       - Example: MAE = 0.2 means average error is 0.2 CGPA points
       - Lower is better
    
    3. RMSE (Root Mean Squared Error):
       - Square root of average squared errors
       - Penalizes large errors more than MAE
       - In same units as target (CGPA points)
       - Lower is better
    """
    print("\n📈 Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training data
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Calculate metrics for test data
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    
    print("\n📊 Training Set Performance:")
    print(f"  • R² Score:  {train_r2:.4f}")
    print(f"  • MAE:       {train_mae:.4f} CGPA points")
    print(f"  • RMSE:      {train_rmse:.4f} CGPA points")
    
    print("\n📊 Test Set Performance:")
    print(f"  • R² Score:  {test_r2:.4f}")
    print(f"  • MAE:       {test_mae:.4f} CGPA points")
    print(f"  • RMSE:      {test_rmse:.4f} CGPA points")
    
    # Check for overfitting
    print("\n🔍 Overfitting Analysis:")
    r2_diff = train_r2 - test_r2
    if r2_diff > 0.1:
        print(f"  ⚠️ Possible overfitting detected (R² difference: {r2_diff:.3f})")
    else:
        print(f"  ✅ No significant overfitting (R² difference: {r2_diff:.3f})")
    
    return {
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'y_test': y_test, 'y_test_pred': y_test_pred
    }

def plot_feature_importance(model, feature_names, save_path=None):
    """
    VISUALIZATION 4: Feature Importance Bar Chart (Model Training)
    
    Purpose:
    --------
    Shows which features are most important for predicting CGPA.
    Higher importance = feature has more influence on predictions.
    
    For Viva:
    ---------
    Q: How is feature importance calculated in Random Forest?
    A: It's based on how much each feature reduces impurity (variance)
       across all trees. Features that lead to bigger reductions in
       prediction error are considered more important.
    
    Q: Which feature is most important and why?
    A: Typically HSC Percentage, because past academic performance
       is the strongest predictor of future academic performance.
    """
    print("\n📊 Creating Feature Importance visualization...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))
    
    bars = plt.barh(
        importance_df['Feature'], 
        importance_df['Importance'],
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add value labels on bars
    for bar, importance in zip(bars, importance_df['Importance']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', va='center', fontsize=10)
    
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance for CGPA Prediction\n(Random Forest Model)', 
              fontsize=14, fontweight='bold')
    plt.xlim(0, max(importances) * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Feature importance plot saved to: {save_path}")
    
    plt.show()
    
    # Print insights
    print("\n🔍 Feature Importance Ranking:")
    for _, row in importance_df.sort_values('Importance', ascending=False).iterrows():
        print(f"  • {row['Feature']}: {row['Importance']:.3f}")

def plot_actual_vs_predicted(y_test, y_pred, save_path=None):
    """
    VISUALIZATION 5: Actual vs Predicted CGPA Plot (Model Evaluation)
    
    Purpose:
    --------
    Visually shows how well predictions match actual values.
    Points close to diagonal line = accurate predictions.
    
    For Viva:
    ---------
    Q: How do you interpret this plot?
    A: - The diagonal red line represents perfect predictions
       - Points above the line: Model underestimated CGPA
       - Points below the line: Model overestimated CGPA
       - Points close to line: Accurate predictions
       - Tight clustering around diagonal = good model
    
    Q: What if points are scattered far from the diagonal?
    A: It means the model has high prediction error and may need
       improvement (more features, different algorithm, etc.)
    """
    print("\n📊 Creating Actual vs Predicted visualization...")
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of actual vs predicted
    plt.scatter(y_test, y_pred, alpha=0.6, s=50, c='#3498db', 
                edgecolors='black', linewidth=0.5, label='Predictions')
    
    # Perfect prediction line (diagonal)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
             linewidth=2, label='Perfect Prediction Line')
    
    # Calculate R² for annotation
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Add metrics box
    textstr = f'R² Score: {r2:.3f}\nMAE: {mae:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.xlabel('Actual CGPA', fontsize=12)
    plt.ylabel('Predicted CGPA', fontsize=12)
    plt.title('Actual vs Predicted CGPA\n(Model Evaluation)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    
    # Equal aspect ratio for better interpretation
    plt.axis('equal')
    plt.xlim(min_val - 0.2, max_val + 0.2)
    plt.ylim(min_val - 0.2, max_val + 0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Actual vs Predicted plot saved to: {save_path}")
    
    plt.show()

def save_model(model, label_encoder, feature_names, filepath='student_performance_model.pkl'):
    """
    Save the trained model and preprocessing objects for deployment.
    
    For Viva:
    ---------
    Q: Why do we save the model?
    A: So we can use it later without retraining. In deployment,
       we load the saved model to make predictions on new data.
    """
    print(f"\n💾 Saving model to {filepath}...")
    
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }
    
    joblib.dump(model_data, filepath)
    print(f"  ✅ Model saved successfully!")
    
    return filepath

def run_full_pipeline(filepath):
    """
    Run the complete model training and evaluation pipeline.
    """
    print("\n" + "=" * 60)
    print("STUDENT PERFORMANCE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    X, y, feature_names, label_encoder = load_and_preprocess_data(filepath)
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 3: Train model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate model
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 5: Create visualizations
    print("\n" + "=" * 60)
    print("CREATING MODEL VISUALIZATIONS")
    print("=" * 60)
    
    # Visualization 4: Feature Importance
    plot_feature_importance(model, feature_names, 'feature_importance.png')
    
    # Visualization 5: Actual vs Predicted
    plot_actual_vs_predicted(results['y_test'], results['y_test_pred'], 
                             'actual_vs_predicted.png')
    
    # Step 6: Save model
    save_model(model, label_encoder, feature_names)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, results

# Main execution
if __name__ == "__main__":
    # Path to dataset
    DATA_PATH = "student performance dataset.xlsx"
    
    # Run full pipeline
    model, results = run_full_pipeline(DATA_PATH)
