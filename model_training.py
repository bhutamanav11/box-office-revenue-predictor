import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        
    def initialize_models(self):
        """Initialize different regression models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror'
            )
        
        return models
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for key models"""
        print("Performing hyperparameter tuning...")
        
        tuned_models = {}
        
        # Random Forest tuning
        print("  Tuning Random Forest...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_params,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        rf_grid.fit(X_train, y_train)
        tuned_models['Random Forest (Tuned)'] = rf_grid.best_estimator_
        print(f"    Best params: {rf_grid.best_params_}")
        
        # Gradient Boosting tuning
        print("  Tuning Gradient Boosting...")
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_params,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        gb_grid.fit(X_train, y_train)
        tuned_models['Gradient Boosting (Tuned)'] = gb_grid.best_estimator_
        print(f"    Best params: {gb_grid.best_params_}")
        
        # XGBoost tuning (if available)
        if XGBOOST_AVAILABLE:
            print("  Tuning XGBoost...")
            xgb_params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            xgb_grid = GridSearchCV(
                xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
                xgb_params,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            xgb_grid.fit(X_train, y_train)
            tuned_models['XGBoost (Tuned)'] = xgb_grid.best_estimator_
            print(f"    Best params: {xgb_grid.best_params_}")
        
        print("Hyperparameter tuning completed.")
        return tuned_models
    
    def train_models(self, X, y_revenue, test_size=0.2, perform_tuning=True, random_state=42):
        """Train multiple models with cross-validation - REVENUE ONLY"""
        print("="*70)
        print("STARTING MODEL TRAINING PROCESS")
        print("="*70)
        
        # Train-test split with FIXED random_state
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_revenue, test_size=test_size, random_state=random_state
        )
        
        print(f"\nDataset Split (random_state={random_state}):")
        print(f"  Training set size: {X_train.shape[0]:,} movies")
        print(f"  Test set size: {X_test.shape[0]:,} movies")
        
        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models = self.initialize_models()
        
        # Add hyperparameter tuned models
        if perform_tuning:
            tuned_models = self.hyperparameter_tuning(X_train, y_train)
            models.update(tuned_models)
        
        # ========== TRAIN REVENUE MODELS ==========
        print("\n" + "="*70)
        print("TRAINING REVENUE PREDICTION MODELS")
        print("="*70)
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*40}")
            print(f"Training: {name}")
            print(f"{'='*40}")
            
            # Use scaled data for linear models
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                X_train_use, X_test_use = X_train_scaled, X_test_scaled
            else:
                X_train_use, X_test_use = X_train, X_test
            
            # Cross-validation
            print("  Running 5-fold cross-validation...")
            cv_scores = cross_val_score(
                model, X_train_use, y_train, 
                cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            cv_rmse = np.sqrt(-cv_scores)
            
            # Train on full training set
            print("  Training on full training set...")
            model.fit(X_train_use, y_train)
            y_pred = model.predict(X_test_use)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate MAPE for movies with revenue > $1M
            mask = y_test > 1000000
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            else:
                mape = np.nan
            
            results[name] = {
                'model': model,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'test_r2': r2,
                'test_rmse': rmse,
                'test_mae': mae,
                'test_mape': mape,
                'y_test': y_test,
                'y_pred': y_pred,
                'cv_scores': cv_rmse
            }
            
            print(f"\n  Results:")
            print(f"    CV RMSE: ${cv_rmse.mean():,.0f} (±${cv_rmse.std():,.0f})")
            print(f"    Test R²: {r2:.4f} ({r2*100:.1f}% variance explained)")
            print(f"    Test RMSE: ${rmse:,.0f}")
            print(f"    Test MAE: ${mae:,.0f}")
            if not np.isnan(mape):
                print(f"    Test MAPE: {mape:.2f}%")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = importance_df
                
                print(f"\n  Top 5 important features:")
                for idx, row in importance_df.head().iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")
        
        self.models = {name: result['model'] for name, result in results.items()}
        self.results = results
        
        # Print best model
        print("\n" + "="*70)
        print("TRAINING COMPLETE - BEST MODEL")
        print("="*70)
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        print(f"\nBest Revenue Model: {best_model_name}")
        print(f"  R²: {results[best_model_name]['test_r2']:.4f}")
        print(f"  RMSE: ${results[best_model_name]['test_rmse']:,.0f}")
        
        return results, (X_train, X_test, y_train, y_test)
    
    def statistical_significance_test(self):
        """Perform statistical tests to compare model performance"""
        from scipy import stats
        
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*70)
        
        model_names = list(self.results.keys())
        cv_scores_dict = {name: self.results[name]['cv_scores'] for name in model_names}
        
        # Perform paired t-tests between models
        best_model = max(model_names, key=lambda x: self.results[x]['test_r2'])
        print(f"\nBest performing model: {best_model} (R² = {self.results[best_model]['test_r2']:.4f})")
        
        print(f"\nPaired t-tests comparing {best_model} with other models:")
        
        for name in model_names:
            if name != best_model:
                t_stat, p_value = stats.ttest_rel(
                    cv_scores_dict[best_model], 
                    cv_scores_dict[name]
                )
                significance = "significant" if p_value < 0.05 else "not significant"
                print(f"  {name}: t={t_stat:.3f}, p={p_value:.4f} ({significance})")
    
    def save_models(self, save_dir="models/"):
        """Save trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best revenue model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
        best_model = self.results[best_model_name]['model']
        best_r2 = self.results[best_model_name]['test_r2']
        best_rmse = self.results[best_model_name]['test_rmse']
        
        joblib.dump(best_model, f"{save_dir}best_revenue_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, f"{save_dir}scaler.pkl")
        
        # Save model metadata for UI
        model_info = {
            'best_revenue_model_name': best_model_name,
            'revenue_r2_score': float(best_r2),
            'revenue_rmse': float(best_rmse)
        }
        
        import json
        with open(f"{save_dir}model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Save all models
        for name, model in self.models.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            joblib.dump(model, f"{save_dir}revenue_{safe_name}.pkl")
        
        print(f"\n{'='*70}")
        print("MODELS SAVED")
        print("="*70)
        print(f"Models saved to {save_dir}")
        print(f"Best Revenue Model ({best_model_name}) saved as best_revenue_model.pkl")
        
        return best_model_name

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Load and prepare data
    preprocessor = DataPreprocessor()
    X, y_revenue, _, df, encoders = preprocessor.prepare_data()
    
    # Train models
    trainer = ModelTrainer()
    results, data_split = trainer.train_models(X, y_revenue, perform_tuning=True)
    
    # Statistical significance testing
    trainer.statistical_significance_test()
    
    # Save models
    best_model = trainer.save_models()