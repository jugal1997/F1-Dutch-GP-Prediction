import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    # Load raw Dutch GP data saved from fetch_dutch_gp.py
    df = pd.read_csv('dutch_gp_results.csv')

    # Print initial info
    print("Initial data shape:", df.shape)
    print(df.head())

    # Step 1: Select relevant columns to keep
    columns_to_keep = [
        'Year',
        'FullName',          # Driver name
        'TeamName',            # Constructor/team name (or 'Constructor')
        'Position',        # Finishing position
        'GridPosition',            # Starting grid position
        'Status',          # Race status (Finished, Accident...)
        'Points',          # Points scored
        'Laps'             # Laps completed
    ]

    # Rename 'Constructor' column to 'Team' if needed
    #if 'Constructor' in df.columns and 'Team' not in df.columns:
     #   df.rename(columns={'Constructor': 'Team'}, inplace=True)

    df = df[columns_to_keep]

    # Step 2: Convert 'Position' to numeric; non-numeric values become NaN
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')

    # Drop rows without valid finishing position
    df_clean = df.dropna(subset=['Position']).copy()

    # Convert 'Position' to integers
    df_clean['Position'] = df_clean['Position'].astype(int)

    # Step 3: Create target variable: 1 if winner, else 0
    df_clean['Winner'] = (df_clean['Position'] == 1).astype(int)

    # Step 4: Create feature flag whether the driver finished the race
    df_clean['Finished'] = df_clean['Status'].apply(lambda x: 1 if 'Finished' in str(x) else 0)

    # Step 5: Encode categorical variables using LabelEncoder
    le_driver = LabelEncoder()
    df_clean['Driver_encoded'] = le_driver.fit_transform(df_clean['FullName'])

    le_team = LabelEncoder()
    df_clean['Team_encoded'] = le_team.fit_transform(df_clean['TeamName'])

    # Step 6: Select features and target
    feature_cols = ['Year', 'GridPosition', 'Points', 'Laps', 'Finished', 'Driver_encoded', 'Team_encoded']
    X = df_clean[feature_cols]
    y = df_clean['Winner']

    print("Sample feature data:")
    print(X.head())
    print("Sample target data:")
    print(y.head())

    # Save processed data to CSV for model training
    df_clean.to_csv('dutch_gp_processed.csv', index=False)
    print("Data preprocessing complete. Saved to 'dutch_gp_processed.csv'.")

if __name__ == "__main__":
    main()
