"""
SIMPLE COPY-PASTE CODE TO GENERATE BOTH GRAPHS
Just paste this at the end of your notebook and run it.
Assumes you already have: results, feature_importance_dict
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================================
# GRAPH 1: FEATURE IMPORTANCE (RF + GB + XGBOOST)
# ============================================================================

# Find models
rf_key = [k for k in feature_importance_dict.keys() if 'Random Forest' in k][0]
gb_key = [k for k in feature_importance_dict.keys() if 'Gradient Boosting' in k][0]
xgb_key = [k for k in feature_importance_dict.keys() if 'XGBoost' in k][0]

# Get top 10
rf_imp = feature_importance_dict[rf_key].head(10)
gb_imp = feature_importance_dict[gb_key].head(10)
xgb_imp = feature_importance_dict[xgb_key].head(10)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# RF
axes[0].barh(range(len(rf_imp)), rf_imp['importance'], color=sns.color_palette("Greens_r", 10))
axes[0].set_yticks(range(len(rf_imp)))
axes[0].set_yticklabels(rf_imp['feature'], fontsize=8)
axes[0].set_xlabel('Importance')
axes[0].set_title('(a) Random Forest', fontweight='bold')
axes[0].invert_yaxis()
for i, v in enumerate(rf_imp['importance']):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=7)

# GB
axes[1].barh(range(len(gb_imp)), gb_imp['importance'], color=sns.color_palette("Blues_r", 10))
axes[1].set_yticks(range(len(gb_imp)))
axes[1].set_yticklabels(gb_imp['feature'], fontsize=8)
axes[1].set_xlabel('Importance')
axes[1].set_title('(b) Gradient Boosting', fontweight='bold')
axes[1].invert_yaxis()
for i, v in enumerate(gb_imp['importance']):
    axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=7)

# XGB
axes[2].barh(range(len(xgb_imp)), xgb_imp['importance'], color=sns.color_palette("Oranges_r", 10))
axes[2].set_yticks(range(len(xgb_imp)))
axes[2].set_yticklabels(xgb_imp['feature'], fontsize=8)
axes[2].set_xlabel('Importance')
axes[2].set_title('(c) XGBoost', fontweight='bold')
axes[2].invert_yaxis()
for i, v in enumerate(xgb_imp['importance']):
    axes[2].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=7)

plt.tight_layout()
plt.savefig('feature_importance_3models.pdf', dpi=300, bbox_inches='tight')
plt.savefig('feature_importance_3models.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ SAVED: feature_importance_3models.pdf/png")


# ============================================================================
# GRAPH 2: PREDICTION VS ACTUAL (RF + GB)
# ============================================================================

# Find best RF and GB
rf_keys = [k for k in results.keys() if 'Random Forest' in k]
gb_keys = [k for k in results.keys() if 'Gradient Boosting' in k]
rf_best = max(rf_keys, key=lambda x: results[x]['test_r2'])
gb_best = max(gb_keys, key=lambda x: results[x]['test_r2'])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for idx, (name, key) in enumerate([('Random Forest', rf_best), ('Gradient Boosting', gb_best)]):
    ax = axes[idx]
    
    y_test = results[key]['y_test'] / 1e6
    y_pred = results[key]['y_pred'] / 1e6
    r2 = results[key]['test_r2']
    rmse = results[key]['test_rmse'] / 1e6
    
    # Scatter
    color = '#1976D2' if idx == 0 else '#2E7D32'
    ax.scatter(y_test, y_pred, alpha=0.5, s=30, color=color)
    
    # Perfect line
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect', zorder=5)
    
    # ±20% band
    ax.fill_between([min_val, max_val], [min_val*0.8, max_val*0.8], 
                    [min_val*1.2, max_val*1.2], alpha=0.2, color='gray', label='±20%')
    
    ax.set_xlabel('Actual Revenue (Million USD)')
    ax.set_ylabel('Predicted Revenue (Million USD)')
    ax.set_title(f'({chr(97+idx)}) {name}\nR²={r2:.4f}, RMSE=${rmse:.1f}M', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('prediction_dual_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('prediction_dual_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ SAVED: prediction_dual_comparison.pdf/png")

print("\n" + "="*70)
print("✅ DONE! Both graphs saved as PDF and PNG")
print("="*70)