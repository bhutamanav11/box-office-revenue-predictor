import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

class PredictionLogger:
    """
    Handles logging of predictions and updating historical data with new entries
    """
    
    def __init__(self):
        self.predictions_file = "predictions_log.csv"
        self.historical_data_file = "historical_averages.csv"
        self.filtered_data_file = "filtered_processed_data.csv"
        self.new_entries_file = "new_entries_log.csv"
        
    def log_prediction(self, movie_data, prediction, confidence_interval=None):
        """
        Log prediction to CSV with all movie details
        
        Args:
            movie_data: Dictionary with movie features
            prediction: Predicted revenue value
            confidence_interval: Tuple of (lower, upper) bounds
        """
        try:
            # Calculate derived metrics
            budget = movie_data.get('budget', 0)
            roi = ((prediction - budget) / budget) if budget > 0 else 0
            profit = prediction - budget
            
            # Prepare log entry
            log_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'movie_name': movie_data.get('movie_name', 'Unknown'),
                
                # Financial metrics
                'budget': budget,
                'predicted_revenue': prediction,
                'profit': profit,
                'roi': roi,
                'roi_percentage': roi * 100,
                
                # Confidence intervals
                'revenue_lower': confidence_interval[0] if confidence_interval else None,
                'revenue_upper': confidence_interval[1] if confidence_interval else None,
                
                # Movie details
                'runtime': movie_data.get('runtime', 0),
                'release_year': movie_data.get('release_year', 0),
                'release_month': movie_data.get('release_month', 0),
                'release_day': movie_data.get('release_day', 0),
                'release_day_of_week': movie_data.get('release_day_of_week', 0),
                
                # Categorical features
                'primary_genre': movie_data.get('primary_genre', ''),
                'secondary_genre': movie_data.get('secondary_genre', 'None'),
                'lead_actor': movie_data.get('lead_actor', ''),
                'director': movie_data.get('director', ''),
                'production_company': movie_data.get('production_company', ''),
                'release_season': movie_data.get('release_season', ''),
                'holiday_window': movie_data.get('holiday_window', 'Regular'),
                
                # Additional features
                'vote_count': movie_data.get('vote_count', 0),
                'vote_average': movie_data.get('vote_average', 6.5),
                'popularity': movie_data.get('popularity', 0),
                'num_cast': movie_data.get('num_cast', 0),
                'num_genres': movie_data.get('num_genres', 1),
                'is_sequel': movie_data.get('is_sequel', 0),
                'is_friday_release': movie_data.get('is_friday_release', 0),
                
                # Derived features
                'budget_per_minute': movie_data.get('budget_per_minute', 0),
                'vote_weighted_score': movie_data.get('vote_weighted_score', 0),
                'budget_log': np.log1p(budget),
                'popularity_log': np.log1p(movie_data.get('popularity', 0)),
            }
            
            # Create DataFrame
            df_new = pd.DataFrame([log_entry])
            
            # Append or create predictions log
            if os.path.exists(self.predictions_file):
                df_existing = pd.read_csv(self.predictions_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(self.predictions_file, index=False)
                print(f"✅ Prediction logged to {self.predictions_file} (Total: {len(df_combined)} predictions)")
            else:
                df_new.to_csv(self.predictions_file, index=False)
                print(f"✅ Created new predictions log: {self.predictions_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error logging prediction: {e}")
            return False
    
    def update_historical_averages(self, movie_data, prediction):
        """
        Update historical averages with new actor/director/genre combinations
        Only adds if combination doesn't exist
        """
        try:
            # Extract key identifiers
            genre = movie_data.get('primary_genre', 'Unknown')
            actor = movie_data.get('lead_actor', 'Unknown')
            director = movie_data.get('director', 'Unknown')
            company = movie_data.get('production_company', 'Unknown')
            
            vote_count = movie_data.get('vote_count', 0)
            popularity = movie_data.get('popularity', 0)
            vote_average = movie_data.get('vote_average', 6.5)
            
            # Check if historical data exists
            if not os.path.exists(self.historical_data_file):
                print(f"⚠️ Historical data file not found: {self.historical_data_file}")
                return False
            
            # Load existing historical data
            df_historical = pd.read_csv(self.historical_data_file)
            
            # Check if this exact combination exists
            exact_match = df_historical[
                (df_historical['primary_genre'] == genre) &
                (df_historical['lead_actor'] == actor) &
                (df_historical['director'] == director) &
                (df_historical['production_company'] == company)
            ]
            
            new_entries = []
            
            # If exact combination doesn't exist, add it
            if exact_match.empty:
                new_entry = {
                    'primary_genre': genre,
                    'lead_actor': actor,
                    'director': director,
                    'production_company': company,
                    'vote_count': vote_count,
                    'popularity': popularity,
                    'vote_average': vote_average,
                    'count': 1,
                    'source': 'prediction',
                    'added_date': datetime.now().strftime("%Y-%m-%d")
                }
                new_entries.append(new_entry)
                print(f"✅ Added new combination: {genre} + {actor} + {director} + {company}")
            
            # Add generic combinations if they don't exist
            combinations_to_check = [
                # Genre + Actor + Director
                {'primary_genre': genre, 'lead_actor': actor, 'director': director, 'production_company': 'Any'},
                # Genre + Actor
                {'primary_genre': genre, 'lead_actor': actor, 'director': 'Any', 'production_company': 'Any'},
                # Actor only
                {'primary_genre': 'Any', 'lead_actor': actor, 'director': 'Any', 'production_company': 'Any'},
                # Director only
                {'primary_genre': 'Any', 'lead_actor': 'Any', 'director': director, 'production_company': 'Any'},
                # Genre only
                {'primary_genre': genre, 'lead_actor': 'Any', 'director': 'Any', 'production_company': 'Any'},
            ]
            
            for combo in combinations_to_check:
                match = df_historical[
                    (df_historical['primary_genre'] == combo['primary_genre']) &
                    (df_historical['lead_actor'] == combo['lead_actor']) &
                    (df_historical['director'] == combo['director']) &
                    (df_historical['production_company'] == combo['production_company'])
                ]
                
                if match.empty:
                    new_entry = {
                        **combo,
                        'vote_count': vote_count,
                        'popularity': popularity,
                        'vote_average': vote_average,
                        'count': 1,
                        'source': 'prediction',
                        'added_date': datetime.now().strftime("%Y-%m-%d")
                    }
                    new_entries.append(new_entry)
            
            # Add all new entries
            if new_entries:
                df_new_entries = pd.DataFrame(new_entries)
                df_updated = pd.concat([df_historical, df_new_entries], ignore_index=True)
                df_updated.to_csv(self.historical_data_file, index=False)
                print(f"✅ Updated historical averages: Added {len(new_entries)} new combinations")
                
                # Also save to separate new entries log
                if os.path.exists(self.new_entries_file):
                    df_existing_new = pd.read_csv(self.new_entries_file)
                    df_all_new = pd.concat([df_existing_new, df_new_entries], ignore_index=True)
                    df_all_new.to_csv(self.new_entries_file, index=False)
                else:
                    df_new_entries.to_csv(self.new_entries_file, index=False)
                
                print(f"✅ New entries also logged to {self.new_entries_file}")
            else:
                print(f"ℹ️ All combinations already exist in historical data")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating historical averages: {e}")
            return False
    
    def save_filtered_processed_data(self, processed_df):
        """
        Save the filtered and processed dataset for future reference
        
        Args:
            processed_df: DataFrame with processed movie data
        """
        try:
            # Add metadata
            metadata = {
                'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_movies': len(processed_df),
                'date_range': f"{processed_df['release_year'].min()}-{processed_df['release_year'].max()}",
                'columns': list(processed_df.columns)
            }
            
            # Save processed data
            processed_df.to_csv(self.filtered_data_file, index=False)
            
            # Save metadata
            metadata_file = self.filtered_data_file.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n{'='*70}")
            print("FILTERED DATA SAVED")
            print('='*70)
            print(f"✅ Saved {len(processed_df):,} processed movies")
            print(f"✅ File: {self.filtered_data_file}")
            print(f"✅ Metadata: {metadata_file}")
            print(f"✅ Date range: {metadata['date_range']}")
            print(f"✅ Features: {len(metadata['columns'])} columns")
            print('='*70 + '\n')
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving filtered data: {e}")
            return False
    
    def get_prediction_statistics(self):
        """
        Get statistics about logged predictions
        """
        try:
            if not os.path.exists(self.predictions_file):
                return None
            
            df = pd.read_csv(self.predictions_file)
            
            stats = {
                'total_predictions': len(df),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                'average_predicted_revenue': df['predicted_revenue'].mean(),
                'average_budget': df['budget'].mean(),
                'average_roi': df['roi'].mean(),
                'profitable_predictions': len(df[df['profit'] > 0]),
                'loss_predictions': len(df[df['profit'] <= 0]),
                'top_genres': df['primary_genre'].value_counts().head(5).to_dict(),
                'top_actors': df['lead_actor'].value_counts().head(5).to_dict(),
                'top_directors': df['director'].value_counts().head(5).to_dict(),
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting prediction statistics: {e}")
            return None
    
    def display_statistics(self):
        """
        Display prediction statistics in a formatted way
        """
        stats = self.get_prediction_statistics()
        
        if stats is None:
            print("No prediction statistics available")
            return
        
        print("\n" + "="*70)
        print("PREDICTION STATISTICS")
        print("="*70)
        print(f"Total Predictions: {stats['total_predictions']:,}")
        print(f"Date Range: {stats['date_range']}")
        print(f"\nAverage Metrics:")
        print(f"  Predicted Revenue: ${stats['average_predicted_revenue']:,.0f}")
        print(f"  Budget: ${stats['average_budget']:,.0f}")
        print(f"  ROI: {stats['average_roi']:.2f}x")
        print(f"\nProfitability:")
        print(f"  Profitable: {stats['profitable_predictions']} ({stats['profitable_predictions']/stats['total_predictions']*100:.1f}%)")
        print(f"  Loss: {stats['loss_predictions']} ({stats['loss_predictions']/stats['total_predictions']*100:.1f}%)")
        print(f"\nTop 5 Genres:")
        for genre, count in stats['top_genres'].items():
            print(f"  {genre}: {count}")
        print(f"\nTop 5 Actors:")
        for actor, count in stats['top_actors'].items():
            print(f"  {actor}: {count}")
        print(f"\nTop 5 Directors:")
        for director, count in stats['top_directors'].items():
            print(f"  {director}: {count}")
        print("="*70 + "\n")

# Example usage functions
def save_training_data(processed_df):
    """
    Convenience function to save processed training data
    
    Args:
        processed_df: DataFrame from data preprocessing
    """
    logger = PredictionLogger()
    return logger.save_filtered_processed_data(processed_df)

def log_new_prediction(movie_data, predicted_revenue, confidence_interval=None):
    """
    Convenience function to log a new prediction
    
    Args:
        movie_data: Dictionary with movie features
        predicted_revenue: Predicted revenue value
        confidence_interval: Optional tuple of (lower, upper) bounds
    """
    logger = PredictionLogger()
    
    # Log the prediction
    success = logger.log_prediction(movie_data, predicted_revenue, confidence_interval)
    
    # Update historical averages with new combinations
    if success:
        logger.update_historical_averages(movie_data, predicted_revenue)
    
    return success

if __name__ == "__main__":
    # Example: Display current prediction statistics
    logger = PredictionLogger()
    logger.display_statistics()