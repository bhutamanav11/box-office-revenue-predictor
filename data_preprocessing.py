import pandas as pd
import numpy as np
import ast
import json
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.column_mapping = {}
        
    def load_and_merge_data(self, movies_path="movies_metadata.csv", credits_path="credits.csv"):
        """Load and merge movies metadata with credits data"""
        print("Loading movie metadata...")
        
        # Load movies metadata
        movies_df = pd.read_csv(movies_path, low_memory=False)
        
        # Load credits
        credits_df = pd.read_csv(credits_path)
        
        print(f"Movies metadata shape: {movies_df.shape}")
        print(f"Credits shape: {credits_df.shape}")
        
        # Clean and merge
        movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
        credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')
        
        # Merge on ID
        df = movies_df.merge(credits_df, on='id', how='inner')
        print(f"Merged dataset shape: {df.shape}")
        
        return df
    
    def load_flexible_datasets(self, file_dict, merge_column='id'):
        """
        Load and merge multiple datasets flexibly
        """
        print(f"Loading {len(file_dict)} dataset(s)...")
        
        dataframes = {}
        for name, path in file_dict.items():
            df = pd.read_csv(path, low_memory=False)
            print(f"  - {name}: {df.shape}")
            dataframes[name] = df
        
        # Merge all dataframes
        if len(dataframes) == 1:
            merged_df = list(dataframes.values())[0]
        else:
            merged_df = list(dataframes.values())[0]
            
            for name, df in list(dataframes.items())[1:]:
                if merge_column in merged_df.columns and merge_column in df.columns:
                    merged_df[merge_column] = pd.to_numeric(merged_df[merge_column], errors='coerce')
                    df[merge_column] = pd.to_numeric(df[merge_column], errors='coerce')
                    
                    merged_df = merged_df.merge(df, on=merge_column, how='inner')
                    print(f"  Merged with {name}: {merged_df.shape}")
                else:
                    print(f"  Warning: Merge column '{merge_column}' not found in {name}")
        
        print(f"Final merged dataset: {merged_df.shape}")
        return merged_df
    
    def extract_json_field(self, json_str, field_key, index=0):
        """Extract specific field from JSON string"""
        try:
            if pd.isna(json_str):
                return 'Unknown'
            data = ast.literal_eval(json_str)
            if data and len(data) > index:
                return data[index].get(field_key, 'Unknown')
            return 'Unknown'
        except:
            return 'Unknown'
    
    def get_director(self, crew_str):
        """Extract director from crew JSON"""
        try:
            if pd.isna(crew_str):
                return 'Unknown'
            crew = ast.literal_eval(crew_str)
            directors = [person['name'] for person in crew if person['job'] == 'Director']
            return directors[0] if directors else 'Unknown'
        except:
            return 'Unknown'
    
    def feature_engineering(self, df):
        """Extract and engineer features from raw data with flexible column detection"""
        print("Engineering features with flexible column detection...")
        
        df = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"  Warning: Column '{col}' not found, using default values")
                if col == 'vote_average':
                    df[col] = 6.5
                elif col == 'vote_count':
                    df[col] = 1000
                elif col == 'popularity':
                    df[col] = 10.0
        
        # Filter valid movies
        df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
        if 'runtime' in df.columns and df['runtime'].notna().any():
            df = df[df['runtime'] > 0]
        else:
            df['runtime'] = 120  # Default runtime
        
        # Extract release date features
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df = df.dropna(subset=['release_date'])
            df['release_year'] = df['release_date'].dt.year
            df['release_month'] = df['release_date'].dt.month
        else:
            print("  Warning: 'release_date' not found, using current year")
            df['release_year'] = 2024
            df['release_month'] = 6
        
        # Filter movies from 1980 onwards
        df = df[df['release_year'] >= 1980]
        
        # Extract features from JSON fields (with fallbacks)
        if 'genres' in df.columns:
            df['primary_genre'] = df['genres'].apply(lambda x: self.extract_json_field(x, 'name'))
        else:
            df['primary_genre'] = 'Unknown'
        
        if 'cast' in df.columns:
            df['lead_actor'] = df['cast'].apply(lambda x: self.extract_json_field(x, 'name'))
        else:
            df['lead_actor'] = 'Unknown'
        
        if 'crew' in df.columns:
            df['director'] = df['crew'].apply(self.get_director)
        else:
            df['director'] = 'Unknown'
        
        if 'production_companies' in df.columns:
            df['production_company'] = df['production_companies'].apply(lambda x: self.extract_json_field(x, 'name'))
        else:
            df['production_company'] = 'Unknown'
        
        # Create season feature
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['release_season'] = df['release_month'].apply(get_season)
        
        # Feature engineering
        df['budget_per_minute'] = df['budget'] / df['runtime']
        df['vote_weighted_score'] = df['vote_average'] * np.log1p(df['vote_count'])
        df['budget_log'] = np.log1p(df['budget'])
        df['popularity_log'] = np.log1p(df['popularity'])
        
        # ROI calculation (properly bounded)
        df['roi'] = (df['revenue'] - df['budget']) / df['budget']
        # Clip extreme ROI values for better model training
        df['roi'] = df['roi'].clip(-0.99, 50)  # Max 50x ROI, min -99% loss
        
        # Select final features
        feature_cols = [
            'budget', 'runtime', 'vote_average', 'vote_count', 'popularity',
            'release_year', 'release_month', 'primary_genre', 'lead_actor',
            'director', 'production_company', 'release_season',
            'budget_per_minute', 'vote_weighted_score', 'budget_log', 'popularity_log'
        ]
        
        df = df[feature_cols + ['revenue', 'roi']].dropna()
        
        print(f"Final dataset shape after feature engineering: {df.shape}")
        print(f"Columns present: {list(df.columns)}")
        return df
    
    def encode_categorical_features(self, df, top_n=50):
        """Encode categorical variables with 'Other' and 'Custom' options"""
        categorical_cols = ['primary_genre', 'lead_actor', 'director', 'production_company', 'release_season']
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            # Keep top categories + add 'Other' and 'Custom'
            if col in ['lead_actor', 'director', 'production_company']:
                top_categories = df[col].value_counts().head(top_n).index.tolist()
                # Add Other and Custom as special categories
                top_categories.extend(['Other', 'Custom'])
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in top_categories else 'Other'
                )
            
            # Label encode
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        return df_encoded
    
    def prepare_data(self, movies_path="movies_metadata.csv", credits_path="credits.csv"):
        """Main preprocessing pipeline"""
        # Load and merge data
        raw_df = self.load_and_merge_data(movies_path, credits_path)
        
        # Feature engineering
        processed_df = self.feature_engineering(raw_df)
        
        # Encode categorical features
        encoded_df = self.encode_categorical_features(processed_df)
        
        # Separate features and targets
        X = encoded_df.drop(['revenue', 'roi'], axis=1)
        y_revenue = encoded_df['revenue']
        
        return X, y_revenue, processed_df, self.label_encoders
    
    def get_required_columns(self):
        """Return list of required columns"""
        return {
            'required': ['budget', 'revenue'],
            'recommended': ['runtime', 'release_date', 'vote_average', 'vote_count', 
                          'popularity', 'genres', 'cast', 'crew', 'production_companies'],
            'description': """
            Required Columns:
            - budget: Movie budget in dollars (numeric) - REQUIRED
            - revenue: Movie revenue in dollars (numeric) - REQUIRED (TARGET)
            
            Recommended Columns (will use defaults if missing):
            - runtime: Movie duration in minutes (numeric) - default: 120
            - release_date: Release date (date format) - default: current year
            - vote_average: IMDb rating (0-10) - default: 6.5
            - vote_count: Number of votes (numeric) - default: 1000
            - popularity: TMDB popularity score (numeric) - default: 10.0
            - genres: Genre information (JSON format) - default: 'Unknown'
            - cast: Cast information (JSON format) - default: 'Unknown'
            - crew: Crew including director (JSON format) - default: 'Unknown'
            - production_companies: Production company info (JSON) - default: 'Unknown'
            """
        }

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X, y_revenue, y_roi, df, encoders = preprocessor.prepare_data()
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Features shape: {X.shape}")
    print(f"Revenue target shape: {y_revenue.shape}")
    print(f"ROI target shape: {y_roi.shape}")