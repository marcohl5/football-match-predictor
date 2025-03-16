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

    ROLLING_COLS = [
        "GF", "GA", "xG", "xGA", "Poss",  # Fixtures page stats
        "SoT", "SoT%", "G/Sh", "G/SoT", "Dist", "FK_x",  # Shooting stats 1
        "PK", "PKatt", "npxG", "npxG/Sh", "G-xG", "np:G-xG",  # Shooting stats 2
        "SoTA", "PSxG", "PSxG+/-", "Opp",  # GK stats
        "1/3", "TotDist", "PrgDist", "xAG", "xA", "KP", "CrsPA", "PrgP", "PPA",  # Passing stats
        "Dead", "FK_y", "TB", "Sw", "Crs", "TI", "CK", "Off", "Blocks",  # Pass type stats
        "GCA", "PassLive", "PassDead", "SCA",  # GCA stats
        "Att Pen", "PrgC", "Mis", "Dis"  # Possession stats
    ]

    # Default feature columns
    POSITIVE_COLS = [
        "GF", "xG", "Poss",  # Fixtures page stats
        "SoT", "npxG", "npxG/Sh", "np:G-xG",  # Shooting stats
        "PSxG+/-",  # GK stats
        "1/3", "PrgDist", "xAG", "KP", "CrsPA", "PrgP", "PPA",  # Passing stats
        "TB",  # Pass type stats
        "GCA", "PassLive", "PassDead", "SCA",  # GCA stats
        "Att Pen", "PrgC"  # Possession stats
    ]

    NEGATIVE_COLS = [
        "GA", "xGA",  # Fixtures page stats
        "SoTA"  # GK stats
    ]

    NEUTRAL_COLS = [
        "SoT%", "G/Sh", "G/SoT", "FK_x", "PK", "PKatt", "Dist", "G-xG",   # Shooting stats
        "PSxG", "PSxG+/-", "Opp",  # GK stats
        "TotDist",  # Passing stats
        "Dead", "FK_y", "Sw", "Crs", "TI", "CK", "Off", "Blocks",  # Pass type stats
        "Mis", "Dis"  # Possession stats
    ]

    # Rolling average weights - most recent games have higher weight
    ROLLING_WEIGHTS = np.array([0.12, 0.113, 0.126, 0.14, 0.155, 0.173, 0.192])

    def __init__(
            self,
            league_name: str,
            match_path: str,
            output_path: str,
            file_name: str,
            strength_file: str,
            strength_type: str,
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
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.start_date = start_date
        self.strength_file = strength_file

        # Load the dataset
        self.df = pd.read_csv(os.path.join(match_path, file_name))

        # Define column groups for easier reference
        self.all_predictor_cols = self.POSITIVE_COLS + self.NEGATIVE_COLS + self.NEUTRAL_COLS

    def clean_data(self) -> None:
        """
        Ensures datetime data format and calls add_time_features.
        """
        # Convert date column to datetime and filter by start date
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df[self.df["Date"] > self.start_date]

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

        # Add season column (e.g., 2020-2021)
        self.df["Season"] = self.df["Date"].apply(
            lambda date: (
                f"{date.year}-{date.year + 1}" if date.month >= 7
                else f"{date.year - 1}-{date.year}"
            )
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
        Create target variable from match result.
        """
        result_mapping = {"W": 2, "D": 1, "L": 0, " ": None}
        self.df["Target"] = self.df["Result"].map(result_mapping)

    def add_strength(self) -> None:
        """
        Add team strength ratings from external elo team ratings based on season.
        Elo ratings can be applied at season start or monthly (season start is used).
        """
        # Load team strength data
        strength_df = pd.read_csv(self.strength_file)
        strength_type = self.strength_type  # Column containing strength ratings, can use 'Strength' or 'Strength_Fpl'
        weight_date = "Season"  # Time period for strength (month/season)

        # Merge home team strength
        self.df = self.df.merge(
            strength_df[["Team", strength_type, weight_date]],
            left_on=[weight_date, "Team"],
            right_on=[weight_date, "Team"],
            how="left",
            suffixes=("", "_Home")
        )
        self.df.rename(columns={strength_type: "Home_Strength"}, inplace=True)

        # Merge away team strength
        self.df = self.df.merge(
            strength_df[["Team", strength_type, weight_date]],
            left_on=[weight_date, "Opponent"],
            right_on=[weight_date, "Team"],
            how="left",
            suffixes=("", "_Away")
        )
        self.df.rename(columns={strength_type: "Away_Strength"}, inplace=True)

        # Drop duplicate column
        self.df.drop(columns=["Team_Away"], inplace=True)

        # Fill missing strength values
        min_strength = 1.8
        self.df["Home_Strength"] = self.df["Home_Strength"].fillna(min_strength)
        self.df["Away_Strength"] = self.df["Away_Strength"].fillna(min_strength)

        # Calculate relative strength
        self.df["Relative_Strength"] = self.df["Home_Strength"] - self.df["Away_Strength"]

    def calculate_weight(self, team_strength: float, opponent_strength: float) -> float:
        """
        Calculate adjustment weight for predictors based on team and opponent strength.

        Parameters:
        team_strenth (float): strength of subject team, calculated in 'add_strength'
        opponent_strength (float): strength of the opponent team, calculated in 'add_strength'

        Returns:
        (float): calculated resulting weight clamped to pre-set values
        """
        if self.elo_scaling_type == "linear":
            # Linear scaling of weight based on strength difference
            weight = 1 + (opponent_strength - team_strength) / 5
        if self.elo_scaling_type == "log":
            # Logarithmic scaling for more nuanced adjustment
            diff = opponent_strength - team_strength
            weight = 1 + np.log1p(abs(diff)) * np.sign(diff) / 5

        # Clamp weight to specified range
        return max(self.min_weight, min(weight, self.max_weight))

    def weighted_stats(self) -> List[str]:
        """
        Apply strength-based weighting to relevant predictors.

        Returns:
            (List[str]): list of weighted predictor column names
        """
        penalty_factor = 1  # Factor for negative stat adjustment

        for index, row in self.df.iterrows():
            # Get team and opponent strength ratings
            team_strength = row["Home_Strength"]
            opponent_strength = row["Away_Strength"]

            # Calculate adjustment weight
            weight = self.calculate_weight(team_strength, opponent_strength)

            # Apply weights to positive stats (higher values are better)
            for col in self.POSITIVE_COLS:
                self.df.loc[index, f"{col}_weighted"] = row[col] * weight

            # Apply inverse weights to negative stats (lower values are better)
            for col in self.NEGATIVE_COLS:
                self.df.loc[index, f"{col}_weighted"] = row[col] * (1 - penalty_factor * (weight - 1))

            # No weight adjustment to neutral cols
            for col in self.NEUTRAL_COLS:
                self.df.loc[index, f"{col}_weighted"] = row[col]

        # Return list of weighted column names
        return [f"{c}_weighted" for c in self.all_predictor_cols]

    def rolling_averages(self, weighted_cols: List[str]) -> List[str]:
        """
        Calculate weighted rolling averages for team statistics using weighted predictors

        Parameters:
            weighted_cols(List): List of weighted predictor column names
        """
        # Define columns for rolling average calculation
        cols = self.ROLLING_COLS
        new_cols = [f"{c}_rolling" for c in cols]

        # Group by team and season for rolling calculations
        self.df = self.df.groupby(["Team", "Season"], group_keys=False).apply(
            lambda group: self.calculate_group_rolling_avg(group, cols, weighted_cols, new_cols)
        )

        # Remove rows with missing rolling averages
        columns_to_check = new_cols + [f"{col}_weighted_rolling" for col in cols]
        self.df = self.df.dropna(subset=columns_to_check)

        # Sort by date and reset index
        self.df = self.df.sort_values(by="Date").reset_index(drop=True)

        return [f"{col}_weighted_rolling" for col in cols] + ["Venue_Code",
                                                              "Hour",
                                                              "Opp_Code",
                                                              "Away_Strength"]

    def calculate_group_rolling_avg(
            self,
            group: pd.DataFrame,
            cols: List[str],
            weighted_cols: List[str],
            new_cols: List[str]
    ) -> pd.DataFrame:
        """
        Perform rolling average calculation, using pre-set rolling weights and window size.
        Calculated using rows grouped by season and team.
        Removes rows for which there wasn't enough historical data to perform rolling average (min 7).

        Parameters:
            group (DataFrame): DataFrame containing matches for a single team and season
            cols (List): Original stat columns to calculate rolling averages for
            weighted_cols (List): Weighted stat columns to calculate rolling averages for
            new_cols (List): Names for new rolling average columns

        Returns:
            group (DataFrame): Updated group with rolling averages
        """
        # Sort by date
        group = group.sort_values("Date")

        # Pre-allocate all columns to avoid fragmentation
        for col in new_cols:
            group[col] = np.nan

        for col in cols:
            group[f"{col}_weighted_rolling"] = np.nan

        # Ensure all columns are numeric
        for col in cols + weighted_cols:
            group[col] = pd.to_numeric(group[col], errors="coerce")

        # Calculate weighted rolling averages for original columns
        rolling_stats_original = (
            group[cols]
            .rolling(len(self.ROLLING_WEIGHTS), min_periods=len(self.ROLLING_WEIGHTS), closed="left")
            .apply(lambda x: np.average(x, weights=self.ROLLING_WEIGHTS), raw=True)
        )

        # Assign one column at a time to avoid alignment issues
        for i, col in enumerate(new_cols):
            group[col] = rolling_stats_original.iloc[:, i]

        # Calculate weighted rolling averages for weighted columns
        rolling_stats_weighted = (
            group[weighted_cols]
            .rolling(len(self.ROLLING_WEIGHTS), min_periods=len(self.ROLLING_WEIGHTS), closed="left")
            .apply(lambda x: np.average(x, weights=self.ROLLING_WEIGHTS), raw=True)
        )

        # Assign one column at a time to avoid alignment issues
        for i, col in enumerate(cols):
            group[f"{col}_weighted_rolling"] = rolling_stats_weighted.iloc[:, i]

        return group

    def save_data(self) -> None:
        """
        Save processed data to output file in CSV format.
        """
        folder_path = self.output_path
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"processed_{self.file_name}")
        self.df.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")

    def run(self) -> tuple[pd.DataFrame, List[str]]:
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
        self.add_strength()
        weighted_cols = self.weighted_stats()
        pred_columns = self.rolling_averages(weighted_cols)
        self.save_data()
        print("Preprocessing complete")
        return self.df, pred_columns


if __name__ == "__main__":
    # For isolated running
    preprocessor = BaseDataPreprocess(
        league_name="Serie A",
        match_path="../original-data/matches/",
        output_path="../processed-data/matches/",
        strength_type="Strength_Fpl",
        file_name="seriea_matches.csv",
        elo_scaling_type="log",  # input "log"/"linear"
        strength_file="../processed-data/teams/processed_szn_start_elo.csv",
        min_weight=0.7,
        max_weight=1.3
    )
    processed_data, predictor_cols = preprocessor.run()

