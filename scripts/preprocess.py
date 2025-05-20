import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class MissingDict(dict):
    __missing__ = lambda self, key: key


# Team name mapping, matching "home team" with "opponent" team names in FBref
LEAGUE_TEAM_MAPPINGS = {
    "Premier League": {
        "Brighton and Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd",
        "Newcastle United": "Newcastle Utd",
        "Tottenham Hotspur": "Tottenham",
        "Wolverhampton Wanderers": "Wolves",
        "Nottingham Forest": "Nott'ham Forest",
        "Sheffield United": "Sheffield Utd",
        "West Bromwich Albion": "West Brom",
        "Huddersfield Town": "Huddersfield",
        "West Ham United": "West Ham",
    },
    "Serie A": {
        "Internazionale": "Inter"
    },
    "La Liga": {
        "Real Betis": "Betis"
    },
    "Ligue 1": {
        "Paris Saint Germain": "Paris SG"
    },
    "Bundesliga": {
        "Bayer Leverkusen": "Leverkusen",
        "Eintracht Frankfurt": "Eint Frankfurt",
        "St Pauli": "St. Pauli",
        "Monchengladbach": "Gladbach"
    }
}

LEAGUE_OPP_MAPPINGS = {
    "La Liga": {
        "Alavés": "Alaves",
        "Almería": "Almeria",
        "Atlético Madrid": "Athletico Madrid",
        "Cádiz": "Cadiz",
        "Leganés": "Leganes"
    },
    "Ligue 1": {
        "Nîmes": "Nimes",
        "Paris S-G": "Paris SG",
        "Saint-Étienne": "Saint Etienne"
    },
    "Bundesliga": {
        "Düsseldorf": "Dusseldorf",
        "Greuther Fürth": "Greuther Furth",
        "Köln": "Koln"
    }
}


class BaseDataPreprocess:
    """
    Class for preprocessing scraped football match data from FBref for XGBoost modeling.

    Handles data cleaning, feature engineering, and stats calculations
    with strength-based weighting and rolling averages.
    """

    """ Original rolling cols
    ROLLING_COLS = [
        "GF", "GA", "xG", "xGA", "Poss",  # Fixtures page stats
        "Sh", "SoT", "SoT%", "G/Sh", "G/SoT", "Dist", "FK_x",  # Shooting stats 1
        "PK", "PKatt", "npxG", "npxG/Sh", "G-xG", "np:G-xG",  # Shooting stats 2
        "SoTA", "PSxG", "PSxG+/-", "Opp",  # GK stats
        "1/3", "TotDist", "PrgDist", "xAG", "xA", "KP", "CrsPA", "PrgP", "PPA",  # Passing stats
        "Dead", "FK_y", "TB", "Sw", "Crs", "TI", "CK", "Off", "Blocks",  # Pass type stats
        "GCA", "PassLive", "PassDead", "SCA",  # GCA stats
        "Att Pen", "PrgC", "Mis", "Dis"  # Possession stats
    ]
    
    Old v2 rolling cols
    col_predictor = ["gf", "ga", "xg", "xga", "sot", "poss", "xag", "att pen",
                "npxg/sh", "kp", "ppa", "gca", "sca", "np:g-xg", "psxg+/-", "1/3", "cmp%"]
    """

    ROLLING_COLS = [
        "GF", "GA", "xG", "xGA", "Poss",  # Fixtures page stats
        "Sh", "SoT", "SoT%",  # Shooting stats 1
        "npxG", "npxG/Sh",  # Shooting stats 2
        "SoTA", "PSxG", "PSxG+/-", "Opp",  # GK stats
        "1/3", "xAG", "xA", "KP", "CrsPA", "PrgP", "PPA",  # Passing stats
        "FK_y", "TB", "Crs", "CK",  # Pass type stats
        "GCA", "PassLive", "PassDead", "SCA",  # GCA stats
        "Att Pen", "PrgC" # Possession stats
    ]

    # Rolling average weights - most recent games have higher weight
    ROLLING_WEIGHTS = np.array([0.1602, 0.178, 0.1978, 0.2198, 0.2442])

    ROLLING_WINDOW_SIZE = 5
    MIN_ROLLING_PERIODS = 5

    def __init__(
            self,
            league_name: str,
            match_path: str,
            output_path: str,
            file_name: str,
            strength_type: str,
            elo_history_file: str,
            elo_scaling_type: str = "log",
            min_weight: float = 0.6,
            max_weight: float = 1.2,
            start_date: str = "2020-08-01",

    ):
        """
        Initialize the data preprocessor.

        Args:
            league_name: Name of the football league
            folder_path: Path to the directory containing input file
            file_name: Name of the input file
            elo_scaling_type: Method for scaling ELO ratings ("linear" or "log")
            min_weight: Minimum weight for strength adjustment
            max_weight: Maximum weight for strength adjustment
            start_date: Filter data from this date onwards
            strength_file: Path to team strength data file
        """
        self.league_name = league_name
        self.folder_path = match_path
        self.output_path = output_path
        self.strength_type = strength_type
        self.file_name = file_name
        self.elo_scaling_type = elo_scaling_type
        self.elo_history_file = elo_history_file
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.start_date = start_date

        # Load the dataset
        self.df = pd.read_csv(os.path.join(match_path, file_name))

        # Define column groups for easier reference
        self.all_predictor_cols = self.ROLLING_COLS

    def clean_data(self) -> None:
        """
        Ensures datetime data format and calls add_time_features.
        """
        # Convert date column to datetime and filter by start date
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df[self.df["Date"] > self.start_date]

        if self.df.empty:
            raise ValueError(f"No data found after start_date filter for {self.league_name}")

        # Create previous date column for merging with elo ratings
        self.df["Date_Prev"] = self.df["Date"] - pd.Timedelta(days=1)
        self.df["Date_Prev"] = pd.to_datetime(self.df["Date_Prev"])

        # Reset index and handle index column
        self.df = self.df.reset_index(drop=True)
        if self.df.columns[0].startswith("Unnamed"):
            self.df = self.df.drop(columns=[self.df.columns[0]])

        # Add temporal information
        self.add_time_features()

    def add_time_features(self) -> None:
        """
        Add season and month features based on match date.
        """
        # Add month column in YYYY-MM format
        self.df["Month"] = self.df["Date"].apply(
            lambda date: f"{date.year}-{date.month:02d}"
        )

    def fix_names(self) -> None:
        """
        Standardise team and opponent names using league-specific mappings, based on Fbref data.
        """
        # Apply team name mapping
        league_mapping = LEAGUE_TEAM_MAPPINGS.get(self.league_name, {})
        mapping = MissingDict(**league_mapping)
        self.df["Team"] = self.df["Team"].map(mapping)

        # Apply opponent name mapping
        opp_league_mapping = LEAGUE_OPP_MAPPINGS.get(self.league_name, {})
        mapping = MissingDict(**opp_league_mapping)
        self.df["Opponent"] = self.df["Opponent"].map(mapping)

    def create_predictors(self) -> None:
        """
        Create a few additional prediction features based on basic match data.
        """
        # Encode categorical features
        self.df["Venue_Code"] = self.df["Venue"].astype("category").cat.codes
        self.df["Opp_Code"] = self.df["Opponent"].astype("category").cat.codes

        # Extract hour from time
        self.df["Hour"] = self.df["Time"].str.extract(r'(\d+):', expand=False).astype(float)

        # Calculate goal difference and expected goal difference
        self.df["GD"] = self.df["GF"] - self.df["GA"]
        self.df["xGD"] = self.df["xG"] - self.df["xGA"]

    def create_target(self) -> None:
        """
        Create target variable representing the MATCH outcome (Home Win, Draw, Away Win).
        - For Home rows: W -> 0 (Home Win), D -> 1 (Draw), L -> 2 (Away Win, because Home Lost)
        - For Away rows: W -> 2 (Away Win), D -> 1 (Draw), L -> 0 (Home Win, because Away Lost)
        The 'Result' column in FBref is always from the perspective of the 'Team' in that row.
        """

        # Define the mappings
        # Mapping for when the 'Team' in the row is the HOME team
        home_team_perspective_map = {
            "W": 0,  # Home team won -> Home Win
            "D": 1,  # Home team drew -> Draw
            "L": 2  # Home team lost -> Away Win
        }

        # Mapping for when the 'Team' in the row is the AWAY team
        away_team_perspective_map = {
            "W": 2,  # Away team won -> Away Win
            "D": 1,  # Away team drew -> Draw
            "L": 0  # Away team lost -> Home Win
        }

        # Initialize the Target column
        self.df["Target"] = np.nan

        # Apply mapping for Home rows
        home_rows = self.df["Venue"] == "Home"
        self.df.loc[home_rows, "Target"] = self.df.loc[home_rows, "Result"].map(home_team_perspective_map)

        # Apply mapping for Away rows
        away_rows = self.df["Venue"] == "Away"
        self.df.loc[away_rows, "Target"] = self.df.loc[away_rows, "Result"].map(away_team_perspective_map)

        # Handle any potential missing "Result" values if they exist (e.g., " " becomes None)
        self.df["Target"] = self.df["Target"].fillna(value=np.nan)

    def add_dynamic_elo(self) -> None:
        try:
            df_elo = pd.read_csv(self.elo_history_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Elo history file not found: {self.elo_history_file}")

        df_elo = df_elo.sort_values(by="Date")
        self.df = self.df.sort_values(by="Date_Prev")

        # Ensure string type for merging columns
        df_elo["Team"] = df_elo["Team"].astype(str)
        self.df["Team"] = self.df["Team"].astype(str)
        self.df["Opponent"] = self.df["Opponent"].astype(str)

        df_elo["Date"] = pd.to_datetime(df_elo["Date"], errors="coerce")

        # Filter df_elo for the current league
        df_elo_league_specific = df_elo[df_elo["Competition"] == self.league_name].copy()

        df_elo_league_specific["month_day"] = df_elo_league_specific["Date"].dt.strftime("%m-%d")
        df_elo_league_specific = df_elo_league_specific[df_elo_league_specific["month_day"] == "08-01"]

        if df_elo_league_specific.empty:
            warnings.warn(f"No Elo data found for league '{self.league_name}' in elo_history_file. "
                          "Will use global fallback Elo for NaNs.")
            elo_comp_season_min_for_apply = pd.Series(dtype=float)  # Empty, so .get() will fail
        else:
            try:
                # This is what will be used by your fill_elo_nan
                elo_comp_season_min_for_apply = df_elo_league_specific.groupby("Season")["Elo"].min()
            except KeyError as e:  # Should be caught by earlier checks, but good to have
                raise KeyError(f"Missing one of 'Season', 'Elo' in df_elo_league_specific for groupby: {e}")

        global_fallback_elo_for_apply = df_elo_league_specific["Elo"].mean() if not df_elo_league_specific.empty else (
            df_elo["Elo"].mean() if not df_elo.empty else 1500)

        df_elo["Date"] = pd.to_datetime(df_elo["Date"])

        # Merge team elo
        self.df = pd.merge_asof(
            self.df,
            df_elo[["Date", "Team", "Elo"]],
            left_on="Date_Prev",
            right_on="Date",
            by="Team",
            direction="backward",
            suffixes=("", "_elo_date_team")
        ).rename(columns={"Elo": 'Team_Elo'})
        if "Date_elo_date_team" in self.df.columns:
            self.df.drop(columns=["Date_elo_date_team"], inplace=True)

        # Merge opponent elo
        self.df = pd.merge_asof(
            self.df,
            df_elo[["Date", "Team", "Elo"]],
            left_on="Date_Prev",
            right_on="Date",
            left_by="Opponent",  # Group self.df by Opponent
            right_by="Team",  # Group elo_df by Team
            direction="backward",
            suffixes=("", "_elo_date_opponent")
        ).rename(columns={"Elo": "Opponent_Elo"})

        if "Date_elo_date_opponent" in self.df.columns:
            self.df.drop(columns=["Date_elo_date_opponent"], inplace=True)

        def fill_elo_nan(row, elo_column_name):
            if pd.isna(row[elo_column_name]):
                # Try to get min Elo using Season from self.df for the current league
                # elo_comp_season_min_for_apply is already filtered for self.league_name
                min_elo_for_season = elo_comp_season_min_for_apply.get(row["Season"])
                if pd.notna(min_elo_for_season):
                    return min_elo_for_season
                else:
                    # Fallback if Season not in our precalculated minimums for this league
                    relevant_team_name = row['Team'] if elo_column_name == 'Team_Elo' else row['Opponent']
                    if not elo_comp_season_min_for_apply.empty:
                         warnings.warn(f"No min Elo found for Season '{row['Season']}' in league '{self.league_name}'. "
                                       f"Using fallback Elo {global_fallback_elo_for_apply:.2f} for team '{relevant_team_name}' on date {row['Date']}.")
                    return global_fallback_elo_for_apply # Use the calculated fallback
            return row[elo_column_name]

        self.df["Team_Elo"] = self.df.apply(lambda row: fill_elo_nan(row, "Team_Elo"), axis=1)
        self.df["Opponent_Elo"] = self.df.apply(lambda row: fill_elo_nan(row, "Opponent_Elo"), axis=1)

    def calculate_venue_specific_rolling_averages(self) -> None:
        """Calculates rolling averages for raw stats and opponent ELO faced,
           grouped by Team, Season, and Venue. Adds columns like 'GF_rolling'.
        """
        cols_to_roll_raw = self.ROLLING_COLS
        cols_to_roll_context = ["Opponent_Elo"]  # Elo of the opponent in that specific match row
        all_roll_cols = cols_to_roll_raw + cols_to_roll_context

        # Ensure data is sorted correctly *before* grouping for rolling calc
        self.df = self.df.sort_values(by=["Team", "Season", "Venue", "Date"])

        # Apply rolling calculations within each group
        # The lambda function now passes the specific rolling window size and weights
        self.df = self.df.groupby(["Team", "Season", "Venue"], group_keys=False, sort=False).apply(
            lambda group:self._apply_rolling_to_group(
                group,
                all_roll_cols,
                window_size=self.ROLLING_WINDOW_SIZE,
                min_periods=self.MIN_ROLLING_PERIODS,
                weights=self.ROLLING_WEIGHTS  # Use the 5-game weights
            )
        )

    def _apply_rolling_to_group(self,
                                group: pd.DataFrame,
                                cols_to_roll: List[str],
                                window_size: int,
                                min_periods: int,
                                weights: np.ndarray) -> pd.DataFrame:

        group = group.copy()  # Avoid SettingWithCopyWarning
        for col in cols_to_roll:
            # Ensure column is numeric, coercing errors
            group[col] = pd.to_numeric(group[col], errors='coerce')
            if group[col].isnull().all():  # Skip if all values are NaN after coercion
                group[f"{col}_rolling"] = np.nan
                continue

            # Apply rolling average
            # Slicing weights to match the actual number of non-NaNs in the window if it's smaller than window_size
            # This is complex to do perfectly with apply(); pandas handles NaNs correctly by default
            group[f"{col}_rolling"] = (
                group[col]
                .rolling(window=window_size, min_periods=min_periods, closed="left")
                .apply(lambda x: np.average(x[~np.isnan(x)], weights=weights[-len(x[~np.isnan(x)]):]) if len(x[~np.isnan(x)]) > 0 else np.nan, raw=False)
            )
        return group

    def combine_match_rows(self) -> pd.DataFrame | List[str]:
        """Combines Home and Away perspective rows for each match into a single row."""

        # Create a unique match ID (Date + sorted team names)
        self.df["Match_ID"] = self.df["Date"].dt.strftime("%Y-%m-%d") + '_' + \
                              self.df.apply(lambda row: '_'.join(sorted([row["Team"], row["Opponent"]])), axis=1)

        # Separate Home and Away views
        df_home = self.df[self.df["Venue"] == "Home"].copy()
        df_away = self.df[self.df["Venue"] == "Away"].copy()

        # Columns to bring from each view
        # Match info from home_df, specific stats from both
        home_cols_to_rename = {
            "Team": "Home_Team", "Opponent": "Away_Team",
            "Team_Elo": "Home_Elo", "Opponent_Elo": "Away_Elo_from_HomeView",  # Opponent's Elo from Home's perspective
            'Venue_Code': 'Venue_Code', 'Hour': 'Hour'  # Take from Home view
            # Add all ROLLING_COLS and Opponent_Elo_rolling with Home_ prefix
        }
        for col in self.ROLLING_COLS + ["Opponent_Elo"]:  # Opponent_Elo_rolling is what we want
            home_cols_to_rename[f"{col}_rolling"] = f"Home_{col}_rolling"

        away_cols_to_rename = {
            "Team": "Away_Team_check", "Opponent": "Home_Team_check",
            "Team_Elo": "Away_Elo", "Opponent_Elo": "Home_Elo_from_AwayView",  # Opponent's Elo from Away's perspective
            # Add all ROLLING_COLS and Opponent_Elo_rolling with Away_ prefix
        }
        for col in self.ROLLING_COLS + ["Opponent_Elo"]:
            away_cols_to_rename[f"{col}_rolling"] = f"Away_{col}_rolling"

        df_home = df_home.rename(columns=home_cols_to_rename)
        df_away = df_away.rename(columns=away_cols_to_rename)

        # Select necessary columns before merge to avoid too many duplicates
        home_essential_cols = ["Match_ID", "Date", "Time", "Comp", "Round", "Day", "Season",
                               "GF", "GA",
                               "Home_Team", "Away_Team", "Result",  # Result is from Home perspective
                               "Home_Elo", "Away_Elo_from_HomeView", "Venue_Code", "Hour"] + \
                              [f"Home_{col}_rolling" for col in self.ROLLING_COLS + ["Opponent_Elo"]]

        away_essential_cols = ["Match_ID", "Away_Elo", "Home_Elo_from_AwayView"] + \
                              [f"Away_{col}_rolling" for col in self.ROLLING_COLS + ["Opponent_Elo"]]

        df_home_subset = df_home[home_essential_cols].copy()
        df_away_subset = df_away[away_essential_cols].copy()

        # Merge
        df_merged = pd.merge(df_home_subset, df_away_subset, on='Match_ID', how='inner')

        # Standardize Target based on Home team's result
        # Result 'W' for Home team means Home Win (0)
        # Result 'D' for Home team means Draw (1)
        # Result 'L' for Home team means Away Win (2) (since Home lost)
        target_map_final = {"W": 0, "D": 1, "L": 2}
        df_merged["Target"] = df_merged["Result"].map(target_map_final)

        result_map_final = {"W": "Home Win", "D": "Draw", "L": "Away Win"}
        df_merged["Result"] = df_merged["Result"].map(result_map_final)

        # Finalize Elo columns (take the direct one, e.g., Home_Elo from home view)
        # Away_Elo from away view is the one we want.
        # We can drop Away_Elo_from_HomeView and Home_Elo_from_AwayView or check consistency
        if "Home_Elo_from_AwayView" in df_merged.columns:  # Sanity check
            df_merged.drop(columns=["Away_Elo_from_HomeView", "Home_Elo_from_AwayView"], inplace=True, errors="ignore")

        # Calculate Relative Elo
        df_merged["Relative_Elo"] = df_merged["Home_Elo"] - df_merged["Away_Elo"]

        # Rename rolling opponent Elo columns for clarity
        df_merged.rename(columns={
            "Home_Opponent_Elo_rolling": "Home_Schedule_Strength_rolling",
            "Away_Opponent_Elo_rolling": "Away_Schedule_Strength_rolling"
        }, inplace=True)

        static_cols = [
            "Hour",
            # "Home_Elo",
            # "Away_Elo",
            "Relative_Elo",
            "Home_Schedule_Strength_rolling",
            "Away_Schedule_Strength_rolling"
        ]

        # Generate Home rolling columns
        home_rolling_cols = [f"Home_{col}_rolling" for col in self.ROLLING_COLS]

        # Generate Away rolling columns
        away_rolling_cols = [f"Away_{col}_rolling" for col in self.ROLLING_COLS]

        # Concatenate all lists
        predictor_cols = static_cols + home_rolling_cols + away_rolling_cols

        df_merged = df_merged.sort_values(by="Date", ascending=True)

        return df_merged, predictor_cols

    def save_data(self, df_matches: pd.DataFrame) -> None:
        """
        Save processed data to output file in CSV format.
        """
        folder_path = self.output_path
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"processed_{self.file_name}")
        df_matches.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")

    def run(self) -> pd.DataFrame | List[str]:
        """
        Run the full preprocessing pipeline.

        Returns:
            (DataFrame): The processed DataFrame
        """
        print(f"Starting preprocessing for {self.league_name} data...")
        self.clean_data()
        self.fix_names()
        self.create_predictors()
        self.create_target()
        self.add_dynamic_elo()
        self.calculate_venue_specific_rolling_averages()
        self.df = self.df.copy()

        merged_matches, predictor_cols = self.combine_match_rows()

        # Filter out rows where teams haven't played 5 home/away games yet
        if predictor_cols and not merged_matches.empty:

            print(f"Shape of processed_df before final dropna: {merged_matches.shape}")
            merged_matches.dropna(subset=["Home_Poss_rolling", "Away_Poss_rolling"], inplace=True)
            processed_df = merged_matches.reset_index(drop=True)
            print(f"Shape of processed_df after final dropna: {processed_df.shape}")

        self.save_data(merged_matches)
        print("Preprocessing complete")
        return merged_matches, predictor_cols


if __name__ == "__main__":
    # For isolated running
    preprocessor = BaseDataPreprocess(
        league_name="Bundesliga",
        match_path="../original-data/matches/",
        output_path="../processed-data/matches/",
        strength_type="Strength_Fpl",
        file_name="bundesliga_matches.csv",
        elo_scaling_type="log",  # input "log"/"linear"
        elo_history_file="../processed-data/teams/processed_elo.csv",
        min_weight=0.7,
        max_weight=1.3,
    )
    processed_data, predictor_cols = preprocessor.run()

