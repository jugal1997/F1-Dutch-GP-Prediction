import fastf1
from fastf1 import plotting
import pandas as pd

# Enable cache (optional, for faster repeated runs)
#fastf1.Cache.enable_cache('./cache')

# Example: get Dutch GP data for 2021-2024
years = [2021, 2022, 2023, 2024]
sessions = ['R']  # 'R' for Race

race_results = []

for year in years:
    try:
        session = fastf1.get_session(year, 'Netherlands', sessions[0])
        session.load(laps=False, telemetry=False, weather=False)
        results = session.results
        results['Year'] = year
        race_results.append(results)
        print(f"Pulled {year} Dutch GP results!")
    except Exception as e:
        print(f"Error for {year}: {e}")

# Combine and save
if race_results:
    all_results = pd.concat(race_results)
    all_results.to_csv('dutch_gp_results.csv', index=False)
    print("Saved Dutch GP results to dutch_gp_results.csv")
else:
    print("No results found.")
