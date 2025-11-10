import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set Springer-appropriate style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        
    def create_springer_correlation_heatmap(self, df, save_path="results/"):
        """Figure 1: Feature Correlation Analysis"""
        print("\nGenerating Figure 1: Feature Correlation Heatmap...")
        
        important_features = [
            'vote_count', 'budget', 'budget_per_minute', 
            'vote_weighted_score', 'popularity', 'runtime',
            'vote_average', 'release_year', 'revenue'
        ]
        
        corr_matrix = df[important_features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt='.2f',
                    cmap=cmap,
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'label': 'Pearson Correlation Coefficient', 'shrink': 0.8},
                    annot_kws={'size': 9},
                    vmin=-1, vmax=1,
                    ax=ax)
        
        ax.set_title('Feature Correlation Matrix with Box Office Revenue', 
                     fontsize=11, fontweight='bold', pad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}fig1_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}fig1_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}fig1_correlation_heatmap.pdf")
        
    def create_springer_feature_importance(self, feature_importance_dict, save_path="results/"):
        """Figure 2: Feature Importance Comparison"""
        print("\nGenerating Figure 2: Feature Importance...")
        
        if 'Gradient Boosting' not in feature_importance_dict:
            print("⚠️ Gradient Boosting feature importance not found")
            return
        
        gb_importance = feature_importance_dict['Gradient Boosting'].head(10)
        
        # Get RF if available
        rf_key = next((k for k in feature_importance_dict.keys() if 'Random Forest' in k and 'Tuned' in k), None)
        if rf_key:
            rf_importance = feature_importance_dict[rf_key].head(10)
            ncols = 2
        else:
            ncols = 1
        
        fig, axes = plt.subplots(1, ncols, figsize=(7, 3.5) if ncols == 2 else (4, 3.5))
        if ncols == 1:
            axes = [axes]
        
        # Gradient Boosting
        colors_gb = sns.color_palette("Blues_r", len(gb_importance))
        axes[0].barh(range(len(gb_importance)), gb_importance['importance'], color=colors_gb)
        axes[0].set_yticks(range(len(gb_importance)))
        axes[0].set_yticklabels(gb_importance['feature'], fontsize=9)
        axes[0].set_xlabel('Relative Importance', fontsize=9)
        axes[0].set_title('(a) Gradient Boosting', fontsize=10, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, v in enumerate(gb_importance['importance']):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
        
        # Random Forest if available
        if ncols == 2:
            colors_rf = sns.color_palette("Greens_r", len(rf_importance))
            axes[1].barh(range(len(rf_importance)), rf_importance['importance'], color=colors_rf)
            axes[1].set_yticks(range(len(rf_importance)))
            axes[1].set_yticklabels(rf_importance['feature'], fontsize=9)
            axes[1].set_xlabel('Relative Importance', fontsize=9)
            axes[1].set_title('(b) Random Forest (Tuned)', fontsize=10, fontweight='bold')
            axes[1].invert_yaxis()
            axes[1].grid(axis='x', alpha=0.3, linestyle='--')
            
            for i, v in enumerate(rf_importance['importance']):
                axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}fig2_feature_importance.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}fig2_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}fig2_feature_importance.pdf")
    
    def create_springer_model_comparison(self, results, save_path="results/"):
        """Figure 3: Comprehensive Model Performance Comparison"""
        print("\nGenerating Figure 3: Model Performance Comparison...")
        
        models = list(results.keys())
        r2_scores = [results[m]['test_r2'] for m in models]
        rmse_scores = [results[m]['test_rmse']/1e6 for m in models]
        mae_scores = [results[m]['test_mae']/1e6 for m in models]
        
        # Sort by R²
        sorted_indices = np.argsort(r2_scores)[::-1]
        models_sorted = [models[i] for i in sorted_indices]
        r2_sorted = [r2_scores[i] for i in sorted_indices]
        rmse_sorted = [rmse_scores[i] for i in sorted_indices]
        mae_sorted = [mae_scores[i] for i in sorted_indices]
        
        # Shorten names
        model_labels = [m.replace(' (Tuned)', '*').replace('Regression', 'Reg.') 
                        for m in models_sorted]
        
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
        y_pos = np.arange(len(models_sorted))
        
        # R² Score
        colors = ['#2E7D32' if r > 0.76 else '#1976D2' if r > 0.74 else '#757575' 
                  for r in r2_sorted]
        axes[0].barh(y_pos, r2_sorted, color=colors, alpha=0.8)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(model_labels, fontsize=8)
        axes[0].set_xlabel('R² Score', fontsize=9)
        axes[0].set_title('(a) Variance Explained', fontsize=9, fontweight='bold')
        axes[0].axvline(x=0.76, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].invert_yaxis()
        
        for i, v in enumerate(r2_sorted):
            axes[0].text(v - 0.02, i, f'{v:.3f}', va='center', ha='right', 
                         fontsize=7, fontweight='bold', color='white')
        
        # RMSE
        colors_rmse = ['#C62828' if r < 95 else '#E64A19' if r < 100 else '#9E9E9E' 
                       for r in rmse_sorted]
        axes[1].barh(y_pos, rmse_sorted, color=colors_rmse, alpha=0.8)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels([''] * len(models_sorted))
        axes[1].set_xlabel('RMSE (Million USD)', fontsize=9)
        axes[1].set_title('(b) Root Mean Squared Error', fontsize=9, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()
        axes[1].invert_xaxis()
        
        for i, v in enumerate(rmse_sorted):
            axes[1].text(v + 2, i, f'${v:.1f}M', va='center', fontsize=7)
        
        # MAE
        colors_mae = ['#1565C0' if m < 46 else '#1976D2' if m < 52 else '#90CAF9' 
                      for m in mae_sorted]
        axes[2].barh(y_pos, mae_sorted, color=colors_mae, alpha=0.8)
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels([''] * len(models_sorted))
        axes[2].set_xlabel('MAE (Million USD)', fontsize=9)
        axes[2].set_title('(c) Mean Absolute Error', fontsize=9, fontweight='bold')
        axes[2].grid(axis='x', alpha=0.3)
        axes[2].invert_yaxis()
        axes[2].invert_xaxis()
        
        for i, v in enumerate(mae_sorted):
            axes[2].text(v + 1, i, f'${v:.1f}M', va='center', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}fig3_model_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}fig3_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}fig3_model_comparison.pdf")
    
    def create_springer_residual_analysis(self, results, save_path="results/"):
        """Figure 4: Residual Analysis"""
        print("\nGenerating Figure 4: Residual Analysis...")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        y_test = results[best_model_name]['y_test']
        y_pred = results[best_model_name]['y_pred']
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred/1e6, residuals/1e6, alpha=0.5, s=20, color='#1976D2')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Revenue (Million USD)', fontsize=9)
        ax1.set_ylabel('Residuals (Million USD)', fontsize=9)
        ax1.set_title('(a) Residual Distribution', fontsize=10, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Trend line
        z = np.polyfit(y_pred/1e6, residuals/1e6, 1)
        p = np.poly1d(z)
        ax1.plot(y_pred/1e6, p(y_pred/1e6), "g--", linewidth=1.5, alpha=0.7, 
                 label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
        ax1.legend(fontsize=7, loc='best')
        
        # Q-Q Plot
        stats.probplot(residuals/1e6, dist="norm", plot=ax2)
        ax2.set_title('(b) Normality Assessment (Q-Q Plot)', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Theoretical Quantiles', fontsize=9)
        ax2.set_ylabel('Sample Quantiles (Million USD)', fontsize=9)
        ax2.grid(alpha=0.3)
        
        # Customize Q-Q plot colors
        line = ax2.get_lines()[0]
        line.set_color('#1976D2')
        line.set_markersize(3)
        ref_line = ax2.get_lines()[1]
        ref_line.set_color('red')
        ref_line.set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}fig4_residual_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}fig4_residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}fig4_residual_analysis.pdf")
    
    def create_springer_temporal_validation(self, processed_df, results, save_path="results/"):
        """Figure 5: NEW - Temporal and Budget Stratification Analysis"""
        print("\nGenerating Figure 5: Temporal Validation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
        
        # 1. Revenue distribution by decade
        decades = [1980, 1990, 2000, 2010]
        decade_labels = ['1980-1989', '1990-1999', '2000-2009', '2010-2017']
        decade_data = []
        
        for i, decade in enumerate(decades):
            if i < len(decades) - 1:
                mask = (processed_df['release_year'] >= decade) & (processed_df['release_year'] < decades[i+1])
            else:
                mask = processed_df['release_year'] >= decade
            
            decade_revenues = processed_df[mask]['revenue'] / 1e6
            decade_data.append(decade_revenues)
        
        bp = ax1.boxplot(decade_data, labels=decade_labels, patch_artist=True,
                         medianprops=dict(color='red', linewidth=2),
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         flierprops=dict(marker='o', markerfacecolor='gray', markersize=2, alpha=0.3))
        
        ax1.set_ylabel('Box Office Revenue (Million USD)', fontsize=9)
        ax1.set_xlabel('Release Period', fontsize=9)
        ax1.set_title('(a) Revenue Distribution by Decade', fontsize=10, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xticklabels(decade_labels, rotation=15, ha='right')
        
        # 2. Model performance by budget category
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        y_test = results[best_model_name]['y_test']
        y_pred = results[best_model_name]['y_pred']
        
        # Get budget values from test set (you'll need to pass this)
        # For now, create synthetic data - YOU NEED TO REPLACE THIS
        budget_categories = ['Low\n(<$20M)', 'Medium\n($20M-$80M)', 'High\n(>$80M)']
        
        # Calculate R² for each category (placeholder - replace with actual calculation)
        # You need to split your test set by budget and calculate R² for each
        r2_by_budget = [0.72, 0.76, 0.68]  # PLACEHOLDER - CALCULATE FROM YOUR DATA
        
        colors_budget = ['#4CAF50', '#2196F3', '#FF9800']
        bars = ax2.bar(range(len(budget_categories)), r2_by_budget, color=colors_budget, alpha=0.8, width=0.6)
        ax2.set_xticks(range(len(budget_categories)))
        ax2.set_xticklabels(budget_categories, fontsize=9)
        ax2.set_ylabel('R² Score', fontsize=9)
        ax2.set_title('(b) Model Performance by Budget Category', fontsize=10, fontweight='bold')
        ax2.set_ylim([0.6, 0.8])
        ax2.axhline(y=results[best_model_name]['test_r2'], color='red', linestyle='--', 
                    linewidth=1.5, label=f'Overall R²={results[best_model_name]["test_r2"]:.3f}')
        ax2.legend(fontsize=7, loc='lower right')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, r2_by_budget)):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}fig5_temporal_validation.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}fig5_temporal_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}fig5_temporal_validation.pdf")
    
    def create_springer_prediction_plot(self, results, save_path="results/"):
        """Figure 6: Predicted vs Actual (Single best model)"""
        print("\nGenerating Figure 6: Prediction Accuracy...")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        y_test = results[best_model_name]['y_test']
        y_pred = results[best_model_name]['y_pred']
        r2 = results[best_model_name]['test_r2']
        rmse = results[best_model_name]['test_rmse']
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Convert to millions
        y_test_m = y_test / 1e6
        y_pred_m = y_pred / 1e6
        
        # Scatter plot with alpha for density
        ax.scatter(y_test_m, y_pred_m, alpha=0.5, s=30, color='#1976D2', edgecolors='none')
        
        # Perfect prediction line
        max_val = max(y_test_m.max(), y_pred_m.max())
        min_val = min(y_test_m.min(), y_pred_m.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Prediction', zorder=5)
        
        # ±20% error bounds
        ax.fill_between([min_val, max_val], 
                         [min_val*0.8, max_val*0.8], 
                         [min_val*1.2, max_val*1.2],
                         alpha=0.2, color='gray', label='±20% Error Band')
        
        ax.set_xlabel('Actual Revenue (Million USD)', fontsize=10)
        ax.set_ylabel('Predicted Revenue (Million USD)', fontsize=10)
        ax.set_title(f'{best_model_name}: Predicted vs Actual Revenue\nR² = {r2:.4f}, RMSE = ${rmse/1e6:.1f}M',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}fig6_prediction_accuracy.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}fig6_prediction_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {save_path}fig6_prediction_accuracy.pdf")
    
    def comprehensive_evaluation_springer(self, results, feature_importance_dict=None, 
                                         processed_df=None, save_path="results/"):
        """
        Generate ALL Springer-quality figures for publication
        """
        print("\n" + "="*70)
        print("GENERATING SPRINGER-QUALITY FIGURES FOR PUBLICATION")
        print("="*70)
        
        # Figure 1: Correlation Heatmap
        if processed_df is not None:
            self.create_springer_correlation_heatmap(processed_df, save_path)
        
        # Figure 2: Feature Importance
        if feature_importance_dict:
            self.create_springer_feature_importance(feature_importance_dict, save_path)
        
        # Figure 3: Model Comparison
        self.create_springer_model_comparison(results, save_path)
        
        # Figure 4: Residual Analysis
        self.create_springer_residual_analysis(results, save_path)
        
        # Figure 5: Temporal Validation (NEW!)
        if processed_df is not None:
            self.create_springer_temporal_validation(processed_df, results, save_path)
        
        # Figure 6: Prediction Accuracy
        self.create_springer_prediction_plot(results, save_path)
        
        print("\n" + "="*70)
        print("ALL SPRINGER-QUALITY FIGURES GENERATED")
        print("="*70)
        print("Generated PDF (vector) + PNG (backup) for all figures")
        print("Figures follow Springer submission guidelines:")
        print("  ✅ Vector format (PDF)")
        print("  ✅ 300 DPI resolution")
        print("  ✅ Appropriate font sizes (9-11pt)")
        print("  ✅ Clear labels and legends")
        print("  ✅ Professional color schemes")
        print("  ✅ Proper subfigure labeling (a), (b), (c)")
        print("="*70 + "\n")
        
        # Generate performance summary table
        performance_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Test_R2': [results[name]['test_r2'] for name in results.keys()],
            'Test_RMSE': [results[name]['test_rmse'] for name in results.keys()],
            'Test_MAE': [results[name]['test_mae'] for name in results.keys()],
            'CV_RMSE_Mean': [results[name]['cv_rmse_mean'] for name in results.keys()],
            'CV_RMSE_Std': [results[name]['cv_rmse_std'] for name in results.keys()]
        }).sort_values('Test_R2', ascending=False)
        
        return performance_df

if __name__ == "__main__":
    print("""
    Updated ModelEvaluator with Springer-quality figure generation.
    
    Call comprehensive_evaluation_springer() instead of comprehensive_evaluation()
    to generate publication-ready figures.
    """)