import os
import time
from data_preprocessing import DataPreprocessor
from exploratory_analysis import ExploratoryAnalyzer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from generate_historical_averages import generate_historical_averages

def create_directories():
    """Create necessary directories"""
    directories = ['results', 'models', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directories created successfully.")

def main():
    """Main execution pipeline"""
    print("="*80)
    print("MOVIE REVENUE PREDICTION - ENHANCED RESEARCH PROJECT")
    print("="*80)
    print("Starting comprehensive analysis pipeline...\n")
    
    start_time = time.time()
    
    # Create directories
    create_directories()
    
    # Step 1: Data Preprocessing
    print("STEP 1: DATA PREPROCESSING")
    print("-" * 80)
    preprocessor = DataPreprocessor()
    X, y_revenue, processed_df, encoders = preprocessor.prepare_data()
    
    # Step 2: Exploratory Data Analysis
    print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")
    print("-" * 80)
    analyzer = ExploratoryAnalyzer()
    eda_results = analyzer.comprehensive_eda(processed_df)
    
    # Step 3: Model Training (Revenue Only)
    print("\nSTEP 3: MODEL TRAINING (REVENUE PREDICTION)")
    print("-" * 80)
    trainer = ModelTrainer()
    results, data_splits = trainer.train_models(
        X, y_revenue, 
        perform_tuning=True,
        random_state=42  # FIXED SEED for reproducibility
    )
    
    # Step 4: Statistical Significance Testing
    print("\nSTEP 4: STATISTICAL SIGNIFICANCE TESTING")
    print("-" * 80)
    trainer.statistical_significance_test()
    
    # Step 5: Model Evaluation (Springer-Quality Figures)
    print("\nSTEP 5: COMPREHENSIVE MODEL EVALUATION")
    print("-" * 80)
    evaluator = ModelEvaluator()
    performance_summary = evaluator.comprehensive_evaluation_springer(
        results=results,
        feature_importance_dict=trainer.feature_importance,
        processed_df=processed_df,
        save_path="results/"
    )
    
    # Step 6: Save Models
    print("\nSTEP 6: SAVING MODELS")
    print("-" * 80)
    best_model_name = trainer.save_models()
    
    # Save preprocessor for UI
    import joblib
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("Preprocessor saved to models/preprocessor.pkl")
    
    # Step 7: Generate Historical Averages for UI
    print("\nSTEP 7: GENERATING HISTORICAL AVERAGES")
    print("-" * 80)
    historical_averages = generate_historical_averages()
    
    # Final Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - RESEARCH RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total movies analyzed: {len(processed_df):,}")
    print(f"  Features used: {len(X.columns)}")
    print(f"  Time period: {processed_df['release_year'].min()}-{processed_df['release_year'].max()}")
    
    print(f"\nBest Performing Model:")
    print(f"  Revenue Model: {best_model_name}")
    best_performance = performance_summary.iloc[0]
    print(f"    RÂ² Score: {best_performance['Test_R2']:.4f}")
    print(f"    RMSE: ${best_performance['Test_RMSE']:,.0f}")
    print(f"    MAE: ${best_performance['Test_MAE']:,.0f}")
    
    print(f"\nAnalysis completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    print(f"\n{'='*80}")
    print("GENERATED FILES")
    print("="*80)
    print("\nData Analysis:")
    print("  - results/distribution_analysis.png")
    print("  - results/correlation_heatmap.png")
    print("  - results/genre_analysis.png")
    print("\nModel Evaluation:")
    print("  - results/model_comparison.png")
    print("  - results/prediction_plots.png")
    print("  - results/residual_analysis.png")
    print("  - results/feature_importance.png")
    print("  - results/evaluation_report.txt")
    print("\nSaved Models:")
    print("  - models/best_revenue_model.pkl")
    print("  - models/scaler.pkl")
    print("  - models/preprocessor.pkl")
    print("  - models/model_info.json")
    print("\nHistorical Data:")
    print("  - historical_averages.csv")
    
    print(f"\n{'='*80}")
    print("READY FOR STREAMLIT UI!")
    print("Run: streamlit run app.py")
    print("="*80)

    return {
        'processed_data': processed_df,
        'results': results,
        'performance_summary': performance_summary,
        'best_model': best_model_name,
        'eda_results': eda_results,
        'historical_averages': historical_averages
    }

if __name__ == "__main__":
    results = main()