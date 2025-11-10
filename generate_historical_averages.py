import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor

def generate_historical_averages():
    """Generate historical averages CSV for UI auto-calculation"""
    print("="*70)
    print("GENERATING HISTORICAL AVERAGES FOR AUTO-CALCULATION")
    print("="*70)
    
    # Load the same processed data used for training
    preprocessor = DataPreprocessor()
    X, y_revenue, processed_df, encoders = preprocessor.prepare_data()
    
    print(f"\nProcessing {len(processed_df):,} movies...")
    
    # Calculate averages by different combinations
    
    # 1. Genre + Actor + Director + Production (most specific)
    print("  Calculating: Genre + Actor + Director + Production combinations...")
    full_combo = processed_df.groupby(['primary_genre', 'lead_actor', 'director', 'production_company']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'vote_average': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    full_combo['specificity'] = 'Full'
    
    # 2. Genre + Actor + Director (high specificity)
    print("  Calculating: Genre + Actor + Director combinations...")
    genre_actor_director = processed_df.groupby(['primary_genre', 'lead_actor', 'director']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'vote_average': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    genre_actor_director['production_company'] = 'Any'
    genre_actor_director['specificity'] = 'High'
    
    # 3. Genre + Actor (medium specificity)
    print("  Calculating: Genre + Actor combinations...")
    genre_actor = processed_df.groupby(['primary_genre', 'lead_actor']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'vote_average': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    genre_actor['director'] = 'Any'
    genre_actor['production_company'] = 'Any'
    genre_actor['specificity'] = 'Medium'
    
    # 4. Actor only (for known actors)
    print("  Calculating: Actor-only combinations...")
    actor_only = processed_df.groupby(['lead_actor']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'vote_average': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    actor_only['primary_genre'] = 'Any'
    actor_only['director'] = 'Any'
    actor_only['production_company'] = 'Any'
    actor_only['specificity'] = 'Actor'
    
    # 5. Director only
    print("  Calculating: Director-only combinations...")
    director_only = processed_df.groupby(['director']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'vote_average': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    director_only['primary_genre'] = 'Any'
    director_only['lead_actor'] = 'Any'
    director_only['production_company'] = 'Any'
    director_only['specificity'] = 'Director'
    
    # 6. Genre only (baseline)
    print("  Calculating: Genre-only combinations...")
    genre_only = processed_df.groupby(['primary_genre']).agg({
        'vote_count': 'mean',
        'popularity': 'mean',
        'vote_average': 'mean',
        'revenue': 'mean',
        'budget': 'mean'
    }).reset_index()
    genre_only['lead_actor'] = 'Any'
    genre_only['director'] = 'Any'
    genre_only['production_company'] = 'Any'
    genre_only['specificity'] = 'Genre'
    
    # Combine all levels
    all_averages = pd.concat([
        full_combo,
        genre_actor_director,
        genre_actor,
        actor_only,
        director_only,
        genre_only
    ], ignore_index=True)
    
    # Reorder columns
    all_averages = all_averages[['primary_genre', 'lead_actor', 'director', 'production_company', 
                                  'vote_count', 'popularity', 'vote_average', 'revenue', 'budget', 'specificity']]
    
    # Round values for cleaner display
    all_averages['vote_count'] = all_averages['vote_count'].round().astype(int)
    all_averages['popularity'] = all_averages['popularity'].round(2)
    all_averages['vote_average'] = all_averages['vote_average'].round(2)
    all_averages['revenue'] = all_averages['revenue'].round().astype(int)
    all_averages['budget'] = all_averages['budget'].round().astype(int)
    
    # Save to CSV
    all_averages.to_csv("historical_averages.csv", index=False)
    
    print(f"\n✓ Historical averages saved to historical_averages.csv")
    print(f"✓ Total combinations: {len(all_averages):,}")
    
    # Show statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\nCombinations by specificity:")
    print(all_averages['specificity'].value_counts().to_string())
    
    print("\n\nSample: Genre-level averages")
    genre_samples = all_averages[all_averages['specificity'] == 'Genre'].sort_values('revenue', ascending=False)
    print(genre_samples[['primary_genre', 'vote_count', 'popularity', 'vote_average', 'revenue']].head(10).to_string(index=False))
    
    print("\n\nTop 10 actors by average revenue:")
    actor_revenues = all_averages[all_averages['specificity'] == 'Actor'].sort_values('revenue', ascending=False).head(10)
    print(actor_revenues[['lead_actor', 'vote_count', 'popularity', 'vote_average', 'revenue']].to_string(index=False))
    
    print("\n\nTop 10 directors by average revenue:")
    director_revenues = all_averages[all_averages['specificity'] == 'Director'].sort_values('revenue', ascending=False).head(10)
    print(director_revenues[['director', 'vote_count', 'popularity', 'vote_average', 'revenue']].to_string(index=False))
    
    print("\n" + "="*70)
    print("HISTORICAL AVERAGES GENERATION COMPLETE")
    print("="*70)
    
    return all_averages

if __name__ == "__main__":
    averages_df = generate_historical_averages()