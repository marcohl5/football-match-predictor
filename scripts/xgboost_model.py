import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from typing import Dict, List, Optional, Union
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class XGBoostFootballModel:
    """
    XGBoost model for predicting football match outcomes with proper train/validation/test splits.
    Postprocessing of results is also included.
    """

    def __init__(self,
                 data_path: str,
                 model_output_path: str,
                 params: dict,
                 pred_cols: List[str],
                 df: pd.DataFrame = None,
                 ):
        """
        Initialise the XGBoost football prediction model.

        Parameters:
            data_path (str): Path to the processed match data used as predictors
            output_path (str): Path where outputs will be saved
        """
        self.data_path = data_path
        self.output_path = model_output_path
        self.model = None
        self.performance = {}
        self.params = params
        self.df = df

        if self.df is None or self.df.empty:
            self.df = pd.read_csv(data_path)

        self.features=pred_cols
        if self.features is None:
            self.features = [
                "Venue_Code", "Hour", "Opp_Code",
                # "Relative_Strength",
                "Away_Strength",
                "GF_weighted_rolling", "GA_weighted_rolling",
                "xG_weighted_rolling", "xGA_weighted_rolling",
                "SoT_weighted_rolling", "Poss_weighted_rolling",
                "xAG_weighted_rolling",
                # "Att Pen_weighted_rolling",
                "npxG/Sh_weighted_rolling", "KP_weighted_rolling",
                "CK_weighted_rolling",
                "PPA_weighted_rolling", "GCA_weighted_rolling",
                "SCA_weighted_rolling", "np:G-xG_weighted_rolling",
                "PSxG+/-_weighted_rolling", "1/3_weighted_rolling"
            ]
    def prepare_data(self) -> tuple[pd.DataFrame]:
        """
        Split data into training, validation, and test sets based on dates.

        Returns:
            (tuple): X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Training set: Seasons 2020-2021 to 2022-2023
        train_data = self.df[(self.df["Date"] < "2023-08-01") &
                             (self.df["Date"] > "2020-08-01")]

        # Validation set: Roughly first half of 2023-2024 season
        val_data = self.df[(self.df["Date"] < "2024-01-01") &
                           (self.df["Date"] >= "2023-08-01")]

        # Test set: Roughly second half of 2023-2024 season
        test_data = self.df[(self.df["Date"] >= "2024-01-01") &
                            (self.df["Date"] < "2024-08-01")]

        # Split into features and target
        X_train = train_data[self.features]
        y_train = train_data["Target"]

        X_val = val_data[self.features]
        y_val = val_data["Target"]

        X_test = test_data[self.features]
        y_test = test_data["Target"]

        # print(self.features)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train the XGBoost model with early stopping.

        Parameters:
            X_train (DataFrame): Training features
            y_train (DataFrame): Training target
            X_val (DataFrame): Validation features
            y_val (DataFrame): Validation target

        Returns:
            trained XGBoost model
        """
        # Initialise model with optimised hyperparameters
        if self.params is None:
            self.params = {
                "eval_metric": "mlogloss",
                "early_stopping_rounds": 10,
                "colsample_bytree": 0.8624235873312975,
                "gamma": 1.6925896014626893,
                "max_depth": 17,
                "min_child_weight": 3,
                "reg_alpha": 79,
                "reg_lambda": 0.5284193927768616,
                "use_label_encoder": False,
                "verbosity": 1
            }


        self.model = xgb.XGBClassifier(
            params=self.params
        )

        # Train model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        return self.model

    def evaluate_model(self, X_val, y_val, X_test, y_test):
        """
        Evaluate the model on validation and test sets with multiple metrics.

        Parameters:
            X_val: Validation features
            y_val: Validation target
            X_test: Test features
            y_test: Test target

        Returns:
            (dict): Dictionary of performance metrics
        """
        # Predict on validation set
        val_probs = self.model.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)

        # Predict on test set
        test_probs = self.model.predict_proba(X_test)
        test_preds = np.argmax(test_probs, axis=1)

        # Calculate metrics
        self.performance = {
            "validation": {
                "accuracy": accuracy_score(y_val, val_preds),
                "f1_macro": f1_score(y_val, val_preds, average="macro"),
                "f1_weighted": f1_score(y_val, val_preds, average="weighted"),
                "confusion_matrix": confusion_matrix(y_val, val_preds),
                "classification_report": classification_report(y_val, val_preds, output_dict=True)
            },
            "test": {
                "accuracy": accuracy_score(y_test, test_preds),
                "f1_macro": f1_score(y_test, test_preds, average="macro"),
                "f1_weighted": f1_score(y_test, test_preds, average="weighted"),
                "confusion_matrix": confusion_matrix(y_test, test_preds),
                "classification_report": classification_report(y_test, test_preds, output_dict=True)
            }
        }

        # Print key performance metrics
        print("\n=== Model Performance ===")
        print(f"\nValidation Metrics:")
        print(f"Accuracy: {self.performance['validation']['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {self.performance['validation']['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {self.performance['validation']['f1_macro']:.4f}")

        print(f"\nTest Metrics:")
        print(f"Accuracy: {self.performance['test']['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {self.performance['test']['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {self.performance['test']['f1_macro']:.4f}")

        return self.performance

    def analyse_feature_importance(self) -> None | pd.DataFrame:
        """
        Analyse and visualise feature importance from the trained model.

        Returns:
            feature_importance (DataFrame): Contains feature importance analysis results
        """
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return

        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            "Feature": self.features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Importance", y="Feature", data=feature_importance)
        plt.title("Feature Importance", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/figures/feature_importance.png")

        print("\n=== Top 5 Most Important Features ===")
        print(feature_importance.head())

        return feature_importance

    def process_match_predictions(self, X_test: pd.DataFrame, y_test) -> tuple[pd.DataFrame, float]:
        """
        Process predictions and calculate match probabilities.

        Parameters:
            X_test (DataFrame): Test features
            y_test (DataFrame): Test target

        Returns:
            match_df (DataFrame): Processed predictions with match probabilities
            accuracy (float): Model accuracy based on pre-processed results
        """
        # Get test data with dates
        test_data = self.df[(self.df["Date"] >= "2024-01-01") &
                            (self.df["Date"] < "2024-08-01")].copy().reset_index(drop=True)

        # Get probabilities
        test_probs = self.model.predict_proba(X_test)
        probs_df = pd.DataFrame(test_probs, columns=["0", "1", "2"])

        # Combine with test data
        combined_df = pd.concat([test_data, probs_df], axis=1)

        # Select relevant columns
        combined_df = combined_df[["Date", "Time", "Comp", "Round", "Day",
                                   "Venue", "Result", "GF", "GA", "Opponent",
                                   "Season", "Team", "Target",
                                   "0", "1", "2"]]

        # Merge home and away teams for the same match
        merged_df = combined_df.merge(combined_df,
                                      left_on=["Date", "Team"],
                                      right_on=["Date", "Opponent"])

        # Extract relevant columns and clean up
        match_df = self.process_merged_predictions(merged_df)

        # Calculate model accuracy
        correct_predictions = (match_df["actual_result"] == match_df["predicted_result"]).sum()
        total_matches = len(match_df)
        accuracy = correct_predictions / total_matches * 100

        print(f"\n=== Match Prediction Results ===")
        print(f"Correct predictions: {correct_predictions}/{total_matches}")
        print(f"Accuracy: {accuracy:.2f}%")

        # Save processed predictions
        match_df.to_csv(f"{self.output_path}/results/match_predictions.csv", index=False)

        return match_df, accuracy

    def process_merged_predictions(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Function to process the merged predictions dataframe.

        Parameters:
            merged_df (DataFrame): DataFrame with pre-processed merged match predictions

        Returns:
            (DataFrame): Cleaned and processed match predictions
        """
        # Keep only home matches to avoid duplicates
        match_df = merged_df[~merged_df.Venue_x.str.contains("Away")].copy()

        # Rename columns for clarity
        match_df = match_df.rename(columns={
            "Team_x": "away_team",
            "Team_y": "home_team",
            "Result_y": "home_result",
            "2_x": "away_w",
            "1_x": "away_d",
            "0_x": "away_l",
            "2_y": "home_w",
            "1_y": "home_d",
            "0_y": "home_l"
        })

        # Calculate normalised probabilities
        match_df["home_win_prob"] = match_df["home_w"] * match_df["away_l"]
        match_df["draw_prob"] = match_df["home_d"] * match_df["away_d"]
        match_df["away_win_prob"] = match_df["away_w"] * match_df["home_l"]

        # Normalise probabilities
        prob_sum = match_df["home_win_prob"] + match_df["draw_prob"] + match_df["away_win_prob"]
        match_df["home_win_prob"] = match_df["home_win_prob"] / prob_sum
        match_df["draw_prob"] = match_df["draw_prob"] / prob_sum
        match_df["away_win_prob"] = match_df["away_win_prob"] / prob_sum

        # Determine predicted result based on highest probability
        match_df["predicted_result"] = match_df[["home_win_prob", "draw_prob", "away_win_prob"]].idxmax(axis=1)
        match_df["predicted_result"] = match_df["predicted_result"].map({
            "home_win_prob": "home_win",
            "draw_prob": "draw",
            "away_win_prob": "away_win"
        })

        # Map actual results
        match_df["actual_result"] = match_df["home_result"].map({
            "W": "home_win",
            "D": "draw",
            "L": "away_win"
        })

        # Select final columns
        return match_df[["Date", "Comp_y", "Round_y", "home_team", "away_team",
                         "actual_result", "predicted_result",
                         "home_win_prob", "draw_prob", "away_win_prob"]]

    def plot_prediction_distribution(self, match_df: pd.DataFrame) -> None:
        """
        Plot the distribution of correct vs incorrect predictions by probability.

        Parameters:
            match_df (DataFrame): DataFrame with match predictions
        """
        # Get max probability for each prediction
        match_df["max_prob"] = match_df[["home_win_prob", "draw_prob", "away_win_prob"]].max(axis=1)
        match_df["is_correct"] = match_df["actual_result"] == match_df["predicted_result"]

        # Separate correct and incorrect predictions
        correct_preds = match_df.loc[match_df["is_correct"], "max_prob"]
        incorrect_preds = match_df.loc[~match_df["is_correct"], "max_prob"]

        # Plot histogram
        bins = np.linspace(0.3, 1.0, 25)
        plt.figure(figsize=(12, 8))

        plt.hist(incorrect_preds, bins, alpha=0.5, color="red",
                 edgecolor="#1E212A", label="Incorrect Predictions")
        plt.hist(correct_preds, bins, alpha=0.5, color="green",
                 edgecolor="#1E212A", label="Correct Predictions")

        plt.xlabel("Prediction Confidence (Probability)")
        plt.ylabel("Number of Matches")
        plt.title(f"Prediction Accuracy by Confidence Level",
                  fontsize=16, fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(f"{self.output_path}/figures/prediction_distribution.png")

    def analyse_prediction_skewness(self, match_df: pd.DataFrame) -> tuple[int, int]:
        """
        Analyse the skewness of the model's predictions.

        Parameters:
            match_df (DataFrame): DataFrame with match predictions

        Returns:
            prediction_counts (int): Number of predictions made
            accuracy_counts (int): Number of accurate predictions made
        """
        # Calculate skewness of predictions
        prediction_counts = match_df["predicted_result"].value_counts()
        actual_counts = match_df["actual_result"].value_counts()

        print("\n=== Prediction Distribution Analysis ===")
        print("\nPredicted Results Distribution:")
        for result, count in prediction_counts.items():
            print(f"{result}: {count} ({count / len(match_df) * 100:.1f}%)")

        print("\nActual Results Distribution:")
        for result, count in actual_counts.items():
            print(f"{result}: {count} ({count / len(match_df) * 100:.1f}%)")

        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(3)
        width = 0.35

        # Ensure consistent order
        categories = ["home_win", "draw", "away_win"]
        predicted_values = [prediction_counts.get(cat, 0) for cat in categories]
        actual_values = [actual_counts.get(cat, 0) for cat in categories]

        rects1 = ax.bar(x - width / 2, predicted_values, width, label="Predicted")
        rects2 = ax.bar(x + width / 2, actual_values, width, label="Actual")

        ax.set_ylabel("Frequency")
        ax.set_title("Prediction Distribution vs Actual Distribution")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_path}/figures/prediction_skewness.png")

        # Calculate statistical skewness of probabilities
        probability_skewness = {
            "home_win_prob": match_df["home_win_prob"].skew(),
            "draw_prob": match_df["draw_prob"].skew(),
            "away_win_prob": match_df["away_win_prob"].skew()
        }

        print("\nProbability Skewness (statistical):")
        for outcome, skew in probability_skewness.items():
            print(f"{outcome}: {skew:.4f}")

        return prediction_counts, actual_counts

    def run_pipeline(self) -> dict:
        """
        Run the complete model pipeline from data preparation to evaluation.

        Returns:
            (dict): Performance metrics and processed match predictions
        """
        print("=== Starting Football Match Prediction Pipeline ===")

        # Prepare data
        print("\nPreparing data splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # Train model
        print("\nTraining XGBoost model...")
        self.train_model(X_train, y_train, X_val, y_val)

        # Evaluate model
        print("\nEvaluating model performance...")
        self.evaluate_model(X_val, y_val, X_test, y_test)

        # Analyse feature importance
        print("\nAnalysing feature importance...")
        self.analyse_feature_importance()

        # Process match predictions
        print("\nProcessing match predictions...")
        match_df, accuracy = self.process_match_predictions(X_test, y_test)

        # Visualisation
        print("\nGenerating visualisations...")
        self.plot_prediction_distribution(match_df)
        self.analyse_prediction_skewness(match_df)

        print(f"\n=== Pipeline Complete ===")
        print(f"Model accuracy on test set: {self.performance['test']['accuracy']:.4f}")
        print(f"Match prediction accuracy: {accuracy:.2f}%")

        return {
            "performance": self.performance,
            "match_predictions": match_df
        }


if __name__ == "__main__":
    # For isolated running
    model = XGBoostFootballModel(
        data_path="../processed-data/matches/processed_seriea_matches.csv",
        model_output_path="../output",
        params=None,
        pred_cols=None,
        df=None,
        )
    results = model.run_pipeline()
