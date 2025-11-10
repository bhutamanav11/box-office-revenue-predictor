import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from exploratory_analysis import ExploratoryAnalyzer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from generate_historical_averages import generate_historical_averages
from prediction_logger import PredictionLogger, log_new_prediction

# Page configuration
st.set_page_config(
    page_title="Movie Revenue Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_historical_averages():
    """Load historical averages if available"""
    if os.path.exists("historical_averages.csv"):
        return pd.read_csv("historical_averages.csv")
    return None

def get_auto_filled_values(genre, actor, director, company, historical_df):
    """Auto-fill vote_count, popularity, and vote_average from historical data."""
    if historical_df is None:
        return None, None, None, "No historical data available"
    
    match_strategies = [
        {
            'name': 'Genre + Actor + Director + Production Company',
            'condition': (historical_df['primary_genre'] == genre) & 
                        (historical_df['lead_actor'] == actor) & 
                        (historical_df['director'] == director) &
                        (historical_df['production_company'] == company)
        },
        {
            'name': 'Genre + Actor + Director',
            'condition': (historical_df['primary_genre'] == genre) & 
                        (historical_df['lead_actor'] == actor) & 
                        (historical_df['director'] == director) &
                        (historical_df['production_company'] == 'Any')
        },
        {
            'name': 'Genre + Actor',
            'condition': (historical_df['primary_genre'] == genre) & 
                        (historical_df['lead_actor'] == actor) & 
                        (historical_df['director'] == 'Any') &
                        (historical_df['production_company'] == 'Any')
        },
        {
            'name': 'Actor Only',
            'condition': (historical_df['primary_genre'] == 'Any') & 
                        (historical_df['lead_actor'] == actor) & 
                        (historical_df['director'] == 'Any') &
                        (historical_df['production_company'] == 'Any')
        },
        {
            'name': 'Director Only',
            'condition': (historical_df['primary_genre'] == 'Any') & 
                        (historical_df['lead_actor'] == 'Any') & 
                        (historical_df['director'] == director) &
                        (historical_df['production_company'] == 'Any')
        },
        {
            'name': 'Genre Only',
            'condition': (historical_df['primary_genre'] == genre) & 
                        (historical_df['lead_actor'] == 'Any') & 
                        (historical_df['director'] == 'Any') &
                        (historical_df['production_company'] == 'Any')
        }
    ]
    
    for strategy in match_strategies:
        filtered = historical_df[strategy['condition']]
        if len(filtered) > 0:
            return (
                int(filtered.iloc[0]['vote_count']), 
                float(filtered.iloc[0]['popularity']),
                float(filtered.iloc[0]['vote_average']),
                f"Matched using: {strategy['name']}"
            )
    
    return 650, 9.5, 6.0, "No match found, using defaults"

def predict_with_confidence(model, scaler, input_df, budget, model_rmse):
    """
    Make predictions with confidence intervals based on model RMSE.
    
    Args:
        model: Trained revenue prediction model
        scaler: Feature scaler (not used for tree models)
        input_df: Input features
        budget: Movie budget
        model_rmse: Model's RMSE for confidence interval calculation
    """
    
    # Revenue prediction
    revenue_pred = model.predict(input_df)[0]
    revenue_pred = max(0, revenue_pred)
    
    # Confidence interval based on RMSE
    revenue_lower = max(0, revenue_pred - model_rmse)
    revenue_upper = revenue_pred + model_rmse
    
    # Calculate ROI from predicted revenue
    roi_pred = (revenue_pred - budget) / budget
    roi_pred = np.clip(roi_pred, -0.99, 50.0)
    
    roi_lower = (revenue_lower - budget) / budget
    roi_upper = (revenue_upper - budget) / budget
    roi_lower = np.clip(roi_lower, -0.99, 50.0)
    roi_upper = np.clip(roi_upper, -0.99, 50.0)
    
    # Profit calculations
    profit = revenue_pred - budget
    profit_lower = revenue_lower - budget
    profit_upper = revenue_upper - budget
    
    return {
        'revenue': revenue_pred,
        'revenue_range': (revenue_lower, revenue_upper),
        'revenue_rmse': model_rmse,
        'roi': roi_pred,
        'roi_range': (roi_lower, roi_upper),
        'roi_percentage': roi_pred * 100,
        'profit': profit,
        'profit_range': (profit_lower, profit_upper),
        'break_even_ratio': revenue_pred / budget
    }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">Movie Revenue Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Box Office Forecasting with Machine Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["Home", "Make Predictions", "Model Performance"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This System")
    st.sidebar.info(
        """
        **Application Information:**
        - 9 ML Algorithms applied
        - TMDB Dataset used
        - XGBoost (Best: R² = 0.769)
        - RMSE-based confidence intervals
        - Historical data intelligence
        - 3 Regression models tested
        - 3 Ensemble models tested
        - 3 tuned ensemble models tested
        """
    )
    
    # ========== PAGE: HOME ==========
    if page == "Home":
        st.markdown("## Overview")
        
        # Model Status
        if os.path.exists("models/model_info.json"):
            with open("models/model_info.json", "r") as f:
                model_info = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Best Model</h4>
                    <p style="font-size: 1.8rem; margin: 0.5rem 0;">{model_info['best_revenue_model_name']}</p>
                    <p style="font-size: 0.9rem;">(Highest R² Score)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>R² Score</h4>
                    <p style="font-size: 1.8rem; margin: 0.5rem 0;">{model_info['revenue_r2_score']:.4f}</p>
                    <p style="font-size: 0.9rem;">({model_info['revenue_r2_score']*100:.1f}% variance explained)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>RMSE</h4>
                    <p style="font-size: 1.8rem; margin: 0.5rem 0;">${model_info['revenue_rmse']/1e6:.1f}M</p>
                    <p style="font-size: 0.9rem;">Average prediction error</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("Models trained and ready for predictions")
        else:
            st.warning("No trained models found. Please train models using the main_script.py")
        
        # Historical Averages Info
        if os.path.exists("historical_averages.csv"):
            st.markdown("## Historical Averages Database")
            
            hist_df = pd.read_csv("historical_averages.csv")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Combinations", f"{len(hist_df):,}")
            with col2:
                st.metric("Unique Actors", f"{len(hist_df['lead_actor'].unique()):,}")
            with col3:
                st.metric("Unique Directors", f"{len(hist_df['director'].unique()):,}")
            with col4:
                st.metric("Unique Genres", f"{len(hist_df['primary_genre'].unique()):,}")
            
            st.caption("Historical averages used for auto-filling vote count, popularity, and ratings")
    
    # ========== PAGE: PREDICTIONS ==========
    elif page == "Make Predictions":
        st.header("Make Revenue Prediction")
        
        if not os.path.exists("models/best_revenue_model.pkl"):
            st.error("Models not found! Please train models first using main_script.py")
            st.stop()
        
        try:
            revenue_model = joblib.load("models/best_revenue_model.pkl")
            scaler = joblib.load("models/scaler.pkl")
            preprocessor = joblib.load("models/preprocessor.pkl")
            
            with open("models/model_info.json", "r") as f:
                model_info = json.load(f)
            
            historical_df = load_historical_averages()
            
            st.success(f"Loaded Model: {model_info['best_revenue_model_name']} (R² = {model_info['revenue_r2_score']:.4f})")
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()
        
        st.markdown("---")
        movie_name = st.text_input("🎬 Movie Title (Required)", 
                                   placeholder="e.g., Avengers: Endgame",
                                   help="Enter the name of the movie you want to predict")
        
        if not movie_name.strip():
            st.warning("Enter a movie title")
        # Input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Budget & Runtime**")
            budget = st.number_input("Budget ($)", min_value=100000, max_value=500000000, 
                                    value=50000000, step=1000000, format="%d",
                                    help="Production budget in US dollars")
            runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=240, 
                                     value=120, step=5)
            release_year = st.number_input("Release Year", min_value=2020, max_value=2030, 
                                          value=2024, step=1)
            release_month = st.selectbox("Release Month", list(range(1, 13)), index=5,
                                        format_func=lambda x: datetime(2024, x, 1).strftime('%B'))
        
        with col2:
            st.markdown("**Cast & Crew**")
            genres = list(preprocessor.label_encoders['primary_genre'].classes_)
            actors = list(preprocessor.label_encoders['lead_actor'].classes_)
            directors = list(preprocessor.label_encoders['director'].classes_)
            companies = list(preprocessor.label_encoders['production_company'].classes_)
            
            genre = st.selectbox("Primary Genre", genres, index=0)
            
            # Actor with custom option
            actor_choice = st.selectbox("Lead Actor", actors[:200] + ['Custom Input'], index=0)
            if actor_choice == 'Custom Input':
                actor_custom = st.text_input("Enter actor name:", value="")
                actor = actor_custom if actor_custom else 'Other'
            else:
                actor = actor_choice
            
            # Director with custom option
            director_choice = st.selectbox("Director", directors[:200] + ['Custom Input'], index=0)
            if director_choice == 'Custom Input':
                director_custom = st.text_input("Enter director name:", value="")
                director = director_custom if director_custom else 'Other'
            else:
                director = director_choice
            
            # Company with custom option
            company_choice = st.selectbox("Production Company", companies[:200] + ['Custom Input'], index=0)
            if company_choice == 'Custom Input':
                company_custom = st.text_input("Enter company name:", value="")
                company = company_custom if company_custom else 'Other'
            else:
                company = company_choice
        
        with col3:
            st.markdown("**Ratings & Popularity**")
            
            if historical_df is not None:
                if st.button("Auto-Fill from Historical Data", 
                           help="Fill ratings based on similar historical movies"):
                    vote_count_auto, popularity_auto, vote_avg_auto, msg = get_auto_filled_values(
                        genre, actor, director, company, historical_df
                    )
                    st.session_state['vote_count'] = vote_count_auto
                    st.session_state['popularity'] = popularity_auto
                    st.session_state['vote_average'] = vote_avg_auto
                    st.session_state['match_msg'] = msg
                
                if 'match_msg' in st.session_state:
                    st.info(st.session_state['match_msg'])
            
            vote_count = st.number_input("Expected Vote Count", min_value=10, max_value=50000,
                                        value=st.session_state.get('vote_count', 2000), step=100)
            popularity = st.number_input("Popularity Score", min_value=0.0, max_value=500.0,
                                        value=float(st.session_state.get('popularity', 20.0)), step=1.0)
            vote_average = st.slider("Expected IMDb Rating", min_value=1.0, max_value=10.0,
                                    value=float(st.session_state.get('vote_average', 7.0)), step=0.1)
        
        st.markdown("---")
        
        if st.button("Predict Revenue", type="primary", use_container_width=True):
            
            with st.spinner("Calculating predictions..."):
                
                try:
                    # Handle custom inputs
                    if actor not in preprocessor.label_encoders['lead_actor'].classes_:
                        actor = 'Other'
                    if director not in preprocessor.label_encoders['director'].classes_:
                        director = 'Other'
                    if company not in preprocessor.label_encoders['production_company'].classes_:
                        company = 'Other'
                    
                    # Season calculation
                    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                                 3: 'Spring', 4: 'Spring', 5: 'Spring',
                                 6: 'Summer', 7: 'Summer', 8: 'Summer',
                                 9: 'Fall', 10: 'Fall', 11: 'Fall'}
                    season = season_map[release_month]
                    
                    # Prepare input
                    input_data = {
                        'budget': budget,
                        'runtime': runtime,
                        'vote_average': vote_average,
                        'vote_count': vote_count,
                        'popularity': popularity,
                        'release_year': release_year,
                        'release_month': release_month,
                        'primary_genre': preprocessor.label_encoders['primary_genre'].transform([genre])[0],
                        'lead_actor': preprocessor.label_encoders['lead_actor'].transform([actor])[0],
                        'director': preprocessor.label_encoders['director'].transform([director])[0],
                        'production_company': preprocessor.label_encoders['production_company'].transform([company])[0],
                        'release_season': preprocessor.label_encoders['release_season'].transform([season])[0],
                        'budget_per_minute': budget / runtime,
                        'vote_weighted_score': vote_average * np.log1p(vote_count),
                        'budget_log': np.log1p(budget),
                        'popularity_log': np.log1p(popularity)
                    }
                    
                    input_df = pd.DataFrame([input_data])
                    
                    # Make predictions with RMSE-based confidence
                    predictions = predict_with_confidence(
                        revenue_model, scaler, input_df, budget, model_info['revenue_rmse']
                    )

                    movie_data_log = {
                        'movie_name': movie_name,
                        'budget': budget,
                        'runtime': runtime,
                        'vote_average': vote_average,
                        'vote_count': vote_count,
                        'popularity': popularity,
                        'release_year': release_year,
                        'release_month': release_month,
                        'primary_genre': genre,
                        'lead_actor': actor,
                        'director': director,
                        'production_company': company,
                        'release_season': season,
                        'budget_per_minute': input_data['budget_per_minute'],
                        'vote_weighted_score': input_data['vote_weighted_score'],
                    }
                    
                    # Log prediction and update historical data
                    log_success = log_new_prediction(
                        movie_data_log, 
                        predictions['revenue'],
                        confidence_interval=predictions['revenue_range']
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Predicted Revenue", f"${predictions['revenue']/1e6:.1f}M")
                        st.caption(f"Range: ${predictions['revenue_range'][0]/1e6:.1f}M - ${predictions['revenue_range'][1]/1e6:.1f}M")
                        st.caption(f"(±${predictions['revenue_rmse']/1e6:.1f}M RMSE)")
                    
                    with col2:
                        st.metric("Expected ROI", f"{predictions['roi_percentage']:.1f}%",
                                delta=f"{predictions['roi']:.2f}x return")
                        st.caption(f"Range: {predictions['roi_range'][0]*100:.1f}% - {predictions['roi_range'][1]*100:.1f}%")
                    
                    with col3:
                        st.metric("Estimated Profit/Loss", f"${predictions['profit']/1e6:.1f}M",
                                delta="Profitable" if predictions['profit'] > 0 else "Loss")
                        st.caption(f"Range: ${predictions['profit_range'][0]/1e6:.1f}M - ${predictions['profit_range'][1]/1e6:.1f}M")
                    
                    with col4:
                        st.metric("Break-Even Ratio", f"{predictions['break_even_ratio']:.2f}x",
                                delta="Profitable" if predictions['break_even_ratio'] > 1 else "Loss")
                        st.caption(f"Budget: ${budget/1e6:.1f}M")
                    
                    # Visualization
                    st.markdown("---")
                    st.subheader("Visual Analysis")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Chart 1: Revenue and Profit with confidence bands
                    categories = ['Revenue ($M)', 'Profit ($M)']
                    predicted = [predictions['revenue']/1e6, predictions['profit']/1e6]
                    lower = [predictions['revenue_range'][0]/1e6, predictions['profit_range'][0]/1e6]
                    upper = [predictions['revenue_range'][1]/1e6, predictions['profit_range'][1]/1e6]
                    
                    x = np.arange(len(categories))
                    width = 0.6
                    
                    bars = ax1.bar(x, predicted, width, 
                                  color=['steelblue', 'coral' if predictions['profit'] > 0 else 'salmon'],
                                  alpha=0.8)
                    
                    # Add confidence bands
                    for i in range(len(categories)):
                        ax1.plot([i, i], [lower[i], upper[i]], 'k-', linewidth=2)
                        ax1.plot([i-0.1, i+0.1], [lower[i], lower[i]], 'k-', linewidth=2)
                        ax1.plot([i-0.1, i+0.1], [upper[i], upper[i]], 'k-', linewidth=2)
                    
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(categories)
                    ax1.set_ylabel('Amount (Million $)')
                    ax1.set_title('Revenue & Profit Predictions (with RMSE bands)', fontweight='bold')
                    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    ax1.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                               f'${predicted[i]:.1f}M',
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontweight='bold')
                    
                    # Chart 2: ROI visualization
                    roi_data = {
                        'Metric': ['ROI (%)', 'Break-Even\nRatio'],
                        'Value': [predictions['roi_percentage'], predictions['break_even_ratio']]
                    }
                    
                    colors = ['green' if predictions['roi'] > 0 else 'red',
                             'green' if predictions['break_even_ratio'] > 1 else 'red']
                    
                    bars2 = ax2.bar(roi_data['Metric'], roi_data['Value'], color=colors, alpha=0.7, width=0.6)
                    ax2.set_ylabel('Value')
                    ax2.set_title('Return on Investment Analysis', fontweight='bold')
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3, label='100% ROI')
                    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
                    ax2.grid(True, alpha=0.3, axis='y')
                    ax2.legend()
                    
                    # Add value labels
                    for bar in bars2:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}',
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Investment Recommendation
                    st.markdown("---")
                    st.subheader("Investment Analysis")
                    
                    if predictions['roi'] > 1.5:
                        st.success(f"""
                        **HIGHLY RECOMMENDED**
                        
                        Strong ROI potential ({predictions['roi_percentage']:.1f}%). 
                        Projected profit: ${predictions['profit']/1e6:.1f}M
                        Low to moderate risk with high return potential.
                        """)
                    elif predictions['roi'] > 0.5:
                        st.info(f"""
                        **RECOMMENDED**
                        
                        Moderate ROI ({predictions['roi_percentage']:.1f}%).
                        Projected profit: ${predictions['profit']/1e6:.1f}M
                        Reasonable risk-reward balance.
                        """)
                    elif predictions['roi'] > 0:
                        st.warning(f"""
                        **CAUTION ADVISED**
                        
                        Low ROI ({predictions['roi_percentage']:.1f}%).
                        Projected profit: ${predictions['profit']/1e6:.1f}M
                        High risk relative to potential return.
                        """)
                    else:
                        st.error(f"""
                        **NOT RECOMMENDED**
                        
                        Negative ROI ({predictions['roi_percentage']:.1f}%).
                        Projected loss: ${abs(predictions['profit'])/1e6:.1f}M
                        Very high risk of financial loss.
                        """)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.exception(e)
    
    # ========== PAGE: MODEL PERFORMANCE ==========
    elif page == "Model Performance":
        st.header("Model Performance & Analysis")
        
        if not os.path.exists("results"):
            st.warning("No training results found. Please train models using main_script.py")
            st.stop()
        
        # Model info
        if os.path.exists("models/model_info.json"):
            with open("models/model_info.json", "r") as f:
                model_info = json.load(f)
            
            st.subheader("Best Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Model</h4>
                    <h3>{model_info['best_revenue_model_name']}</h3>
                    <p style="font-size: 0.9rem;">(Highest R² Score)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>R² Score</h4>
                    <h3>{model_info['revenue_r2_score']:.4f}</h3>
                    <p>{model_info['revenue_r2_score']*100:.1f}% variance explained</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>RMSE</h4>
                    <h3>${model_info['revenue_rmse']/1e6:.1f}M</h3>
                    <p>Average prediction error</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("Model Evaluation Charts")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Model Comparisons", "Data Analysis", "Detailed Evaluations"])
        
        with tab1:
            st.markdown("### Model Performance Comparison")
            if os.path.exists("results/model_comparison.png"):
                st.image("results/model_comparison.png", use_container_width=True)
            else:
                st.warning("Model comparison chart not found")
            
            st.markdown("---")
            
            st.markdown("### Feature Importance Analysis")
            if os.path.exists("results/enhanced_feature_importance_three_models.png"):
                st.image("results/enhanced_feature_importance_three_models.png", use_container_width=True)
            else:
                st.warning("Feature importance chart not found")
        
        with tab2:
            st.markdown("### Distribution Analysis")
            if os.path.exists("results/distribution_analysis.png"):
                st.image("results/distribution_analysis.png", use_container_width=True)
            else:
                st.warning("Distribution analysis chart not found")
            
            st.markdown("---")
            
            st.markdown("### Correlation Heatmap")
            if os.path.exists("results/fig1_correlation_heatmap.png"):
                st.image("results/fig1_correlation_heatmap.png", use_container_width=True)
            else:
                st.warning("Correlation heatmap not found")
            
            st.markdown("---")
            
            st.markdown("### Genre Analysis")
            if os.path.exists("results/genre_analysis.png"):
                st.image("results/genre_analysis.png", use_container_width=True)
            else:
                st.warning("Genre analysis chart not found")
        
        with tab3:
            st.markdown("### Prediction Accuracy Plots")
            if os.path.exists("results/prediction_plots.png"):
                st.image("results/prediction_plots.png", use_container_width=True)
            else:
                st.warning("Prediction plots not found")
            
            st.markdown("---")
            
            st.markdown("### Residual Analysis")
            if os.path.exists("results/residual_analysis.png"):
                st.image("results/residual_analysis.png", use_container_width=True)
            else:
                st.warning("Residual analysis chart not found")
        
        # Historical Averages Info
        if os.path.exists("historical_averages.csv"):
            st.markdown("---")
            st.subheader("Historical Averages Database")
            
            hist_df = pd.read_csv("historical_averages.csv")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Combinations", f"{len(hist_df):,}")
            with col2:
                st.metric("Unique Actors", f"{len(hist_df['lead_actor'].unique()):,}")
            with col3:
                st.metric("Unique Directors", f"{len(hist_df['director'].unique()):,}")
            with col4:
                st.metric("Unique Genres", f"{len(hist_df['primary_genre'].unique()):,}")
            
            with st.expander("Browse Historical Data Sample", expanded=False):
                st.dataframe(hist_df.head(100), use_container_width=True)

if __name__ == "__main__":
    main()