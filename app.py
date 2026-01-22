"""
Student Performance Prediction - Streamlit Deployment
=====================================================
Single-page web application with ALL visualizations.

Features:
- Interactive data exploration
- All 5 visualizations in one page
- Live CGPA prediction
- Model metrics display

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the student performance dataset."""
    df = pd.read_excel("student performance dataset.xlsx")
    return df

@st.cache_resource
def train_model(df):
    """Train the Random Forest model and return model with metrics."""
    # Preprocess
    df_processed = df.copy()
    label_encoder = LabelEncoder()
    df_processed['Gender_Encoded'] = label_encoder.fit_transform(df_processed['Gender'])
    
    # Features and target
    feature_columns = ['Gender_Encoded', 'Program Duration', 'Year of Student', 'HSC Percentage']
    X = df_processed[feature_columns]
    y = df_processed['CGPA']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return model, label_encoder, feature_columns, X_test, y_test, y_pred, metrics

def create_correlation_heatmap(df):
    """VISUALIZATION 1: Correlation Heatmap"""
    numeric_cols = ['Program Duration', 'Year of Student', 'HSC Percentage', 'CGPA']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdYlBu_r',
        aspect='auto',
        title='📊 Correlation Heatmap'
    )
    fig.update_layout(
        height=500,
        title_x=0.5,
        title_font_size=20
    )
    return fig

def create_hsc_vs_cgpa_scatter(df):
    """VISUALIZATION 2: HSC vs CGPA Scatter Plot"""
    fig = px.scatter(
        df,
        x='HSC Percentage',
        y='CGPA',
        color='Gender',
        trendline='ols',
        title='📈 HSC Percentage vs CGPA',
        color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'},
        opacity=0.7
    )
    
    # Add correlation annotation
    corr = df['HSC Percentage'].corr(df['CGPA'])
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"Correlation: {corr:.3f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=14)
    )
    
    fig.update_layout(
        height=500,
        title_x=0.5,
        title_font_size=20
    )
    return fig

def create_cgpa_by_gender_boxplot(df):
    """VISUALIZATION 3: CGPA by Gender Box Plot"""
    fig = px.box(
        df,
        x='Gender',
        y='CGPA',
        color='Gender',
        title='👥 CGPA Distribution by Gender',
        color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'},
        points='all'
    )
    fig.update_layout(
        height=500,
        title_x=0.5,
        title_font_size=20,
        showlegend=False
    )
    return fig

def create_feature_importance_chart(model, feature_names):
    """VISUALIZATION 4: Feature Importance"""
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)
    
    fig = go.Figure(go.Bar(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(
            color=importances[indices],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f'{imp:.3f}' for imp in importances[indices]],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='🎯 Feature Importance (Model Training)',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=500,
        title_x=0.5,
        title_font_size=20
    )
    return fig

def create_actual_vs_predicted_plot(y_test, y_pred, metrics):
    """VISUALIZATION 5: Actual vs Predicted"""
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color='#3498db',
            opacity=0.6,
            line=dict(width=1, color='black')
        )
    ))
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Add metrics annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"R² Score: {metrics['r2']:.3f}<br>MAE: {metrics['mae']:.3f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=14),
        align="left"
    )
    
    fig.update_layout(
        title='✅ Actual vs Predicted CGPA (Model Evaluation)',
        xaxis_title='Actual CGPA',
        yaxis_title='Predicted CGPA',
        height=500,
        title_x=0.5,
        title_font_size=20
    )
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🎓 Student Performance Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predicting CGPA using Machine Learning | Data Mining Project</p>', unsafe_allow_html=True)
    
    # Load data and train model
    with st.spinner('Loading data and training model...'):
        df = load_data()
        model, label_encoder, feature_names, X_test, y_test, y_pred, metrics = train_model(df)
    
    # Sidebar - Prediction Interface
    st.sidebar.header("🔮 Make a Prediction")
    st.sidebar.markdown("Enter student details to predict CGPA:")
    
    # Input fields
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    program_duration = st.sidebar.selectbox("Program Duration (years)", [4, 5])
    year_of_student = st.sidebar.selectbox("Year of Student", [1, 2, 3, 4, 5])
    hsc_percentage = st.sidebar.slider("HSC Percentage", 45.0, 100.0, 75.0, 0.5)
    
    # Predict button
    if st.sidebar.button("🎯 Predict CGPA", type="primary", use_container_width=True):
        # Prepare input
        gender_encoded = 0 if gender == "Female" else 1
        input_data = np.array([[gender_encoded, program_duration, year_of_student, hsc_percentage]])
        
        # Make prediction
        predicted_cgpa = model.predict(input_data)[0]
        
        # Display result
        st.sidebar.success(f"**Predicted CGPA: {predicted_cgpa:.2f}**")
        
        # Grade classification
        if predicted_cgpa >= 3.7:
            grade = "A+ (Excellent)"
            emoji = "🏆"
        elif predicted_cgpa >= 3.3:
            grade = "A (Very Good)"
            emoji = "🌟"
        elif predicted_cgpa >= 3.0:
            grade = "B+ (Good)"
            emoji = "👍"
        elif predicted_cgpa >= 2.5:
            grade = "B (Satisfactory)"
            emoji = "✅"
        else:
            grade = "Needs Improvement"
            emoji = "📚"
        
        st.sidebar.info(f"{emoji} Expected Grade: **{grade}**")
    
    # Model Performance Metrics
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{metrics['r2']:.3f}", help="Proportion of variance explained (higher is better)")
    with col2:
        st.metric("MAE", f"{metrics['mae']:.3f}", help="Mean Absolute Error in CGPA points (lower is better)")
    with col3:
        st.metric("RMSE", f"{metrics['rmse']:.3f}", help="Root Mean Squared Error (lower is better)")
    
    # Dataset Overview
    st.markdown("---")
    st.subheader("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Avg CGPA", f"{df['CGPA'].mean():.2f}")
    with col3:
        st.metric("Avg HSC %", f"{df['HSC Percentage'].mean():.1f}%")
    with col4:
        gender_ratio = df['Gender'].value_counts()
        st.metric("Gender Ratio (M/F)", f"{gender_ratio.get('Male', 0)}/{gender_ratio.get('Female', 0)}")
    
    # Visualizations in Tabs
    st.markdown("---")
    st.subheader("📈 Data Visualizations")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Correlation Heatmap",
        "2️⃣ HSC vs CGPA",
        "3️⃣ CGPA by Gender",
        "4️⃣ Feature Importance",
        "5️⃣ Model Evaluation"
    ])
    
    with tab1:
        st.markdown("### 📊 Visualization 1: Correlation Heatmap (EDA)")
        st.markdown("""
        **Purpose:** Shows the correlation between all numeric variables.
        
        **Interpretation:**
        - Values close to **+1**: Strong positive correlation
        - Values close to **-1**: Strong negative correlation
        - Values close to **0**: No correlation
        """)
        fig1 = create_correlation_heatmap(df)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Key insight
        corr_cgpa_hsc = df['HSC Percentage'].corr(df['CGPA'])
        st.info(f"💡 **Key Insight:** HSC Percentage has the strongest correlation with CGPA ({corr_cgpa_hsc:.3f})")
    
    with tab2:
        st.markdown("### 📈 Visualization 2: HSC Percentage vs CGPA (EDA)")
        st.markdown("""
        **Purpose:** Shows the relationship between HSC marks and university CGPA.
    
        """)
        fig2 = create_hsc_vs_cgpa_scatter(df)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown("### 👥 Visualization 3: CGPA Distribution by Gender (EDA)")
        st.markdown("""
        **Purpose:** Shows how CGPA varies across different genders.
        
        **Box Plot Shows:**
        - **Median** (middle line)
        - **Interquartile Range** (the box - 25th to 75th percentile)
        - **Outliers** (individual points)
        """)
        fig3 = create_cgpa_by_gender_boxplot(df)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Statistics
        gender_stats = df.groupby('Gender')['CGPA'].agg(['mean', 'median', 'std']).round(3)
        st.dataframe(gender_stats, use_container_width=True)
    
    with tab4:
        st.markdown("### 🎯 Visualization 4: Feature Importance (Model Training)")
        st.markdown("""
        **Purpose:** Shows which features are most important for predicting CGPA.
    
        """)
        fig4 = create_feature_importance_chart(model, feature_names)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Feature ranking
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        st.info(f"💡 **Most Important Feature:** {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']:.3f})")
    
    with tab5:
        st.markdown("### ✅ Visualization 5: Actual vs Predicted CGPA (Model Evaluation)")
        st.markdown("""
        **Purpose:** Visually evaluates model accuracy by comparing actual vs predicted values.
        
        **Interpretation:**
        - Points **on the red line** = Perfect predictions
        - Points **above the line** = Model underestimated
        - Points **below the line** = Model overestimated
        - **Tighter clustering** around diagonal = Better model
        """)
        fig5 = create_actual_vs_predicted_plot(y_test, y_pred, metrics)
        st.plotly_chart(fig5, use_container_width=True)
    
    # Data Sample
    st.markdown("---")
    st.subheader("📄 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Student Performance Prediction</strong> | Data Mining Project</p>
        <p>Built with Streamlit, Scikit-learn, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
