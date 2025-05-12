import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from typing import Dict, List, Optional  # Removed Union as it wasn't strictly needed here
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # For XGBoost related future warnings


class XGBoostFootballModel:
    """
    XGBoost model for predicting football match outcomes (Home Win, Draw, Away Win)
    using a single model trained on match-centric data.
    """

    def __init__(self,
                 model_output_path: str,
                 params: dict,
                 pred_cols: List[str],
                 df: pd.DataFrame,  # df is now required, data_path can be removed or made optional
                 data_path: Optional[str] = None,  # Made data_path optional
                 ):
        """
        Initialise the XGBoost football prediction model.

        Parameters:
            model_output_path (str): Path where outputs (figures, results) will be saved.
            params (dict): Hyperparameters for the XGBoost model.
            pred_cols (List[str]): List of feature column names to be used for training.
            df (pd.DataFrame): The processed DataFrame where each row is a unique match.
            data_path (Optional[str]): Path to the processed match data (used if df is None).
        """
        self.data_path = data_path
        self.output_path = model_output_path
        self.model = None
        self.performance = {}
        self.params = params
        self.df = df
        self.features = pred_cols  # Renamed pred_cols to self.features for consistency

        if self.df is None:
            if self.data_path:
                print(f"Loading data from {self.data_path}")
                self.df = pd.read_csv(self.data_path)
                # Ensure 'Date' is datetime if loading from CSV
                if 'Date' in self.df.columns:
                    self.df['Date'] = pd.to_datetime(self.df['Date'])
            else:
                raise ValueError("DataFrame 'df' or 'data_path' must be provided.")

        if self.df.empty:
            raise ValueError("Provided DataFrame 'df' is empty.")

        if not self.features:  # Check if the list is empty
            # Define a more sensible default list based on the new structure
            # This default is just a placeholder and should ideally always be passed via pred_cols
            warnings.warn("No 'pred_cols' provided. Using a generic default list which might not be optimal.")
            self.features = [
                'Venue_Code', 'Hour', 'Home_Elo', 'Away_Elo', 'Relative_Elo',
                'Home_Schedule_Strength_rolling', 'Away_Schedule_Strength_rolling',
                # Add some example rolling cols, this needs to match your actual generated columns
                'Home_GF_rolling', 'Away_GF_rolling', 'Home_GA_rolling', 'Away_GA_rolling',
                'Home_xG_rolling', 'Away_xG_rolling', 'Home_xGA_rolling', 'Away_xGA_rolling'
            ]
            # Ensure these default features exist in the dataframe
            missing_default_features = [f for f in self.features if f not in self.df.columns]
            if missing_default_features:
                raise ValueError(f"Default features are missing from the DataFrame: {missing_default_features}. "
                                 "Please provide 'pred_cols'.")

        # Ensure all specified features are in the DataFrame
        missing_features = [f for f in self.features if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"Features specified in 'pred_cols' are missing from the DataFrame: {missing_features}")

        # Ensure Target column exists
        if "Target" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'Target' column (0: Home Win, 1: Draw, 2: Away Win).")

    def prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into training, validation, and test sets based on dates.
        The DataFrame self.df is assumed to have one row per match.

        Returns:
            (tuple): X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Ensure 'Date' column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Training set: Seasons 2020-2021 to 2022-2023
        train_data = self.df[(self.df["Date"] < "2023-08-01") &
                             (self.df["Date"] > "2020-08-01")].copy()  # Use .copy()

        # Validation set: Roughly first half of 2023-2024 season
        val_data = self.df[(self.df["Date"] < "2024-01-01") &
                           (self.df["Date"] >= "2023-08-01")].copy()

        # Test set: Roughly second half of 2023-2024 season
        test_data = self.df[(self.df["Date"] >= "2024-01-01") &
                            (self.df["Date"] < "2024-08-01")].copy()  # Assuming end of season is before Aug 1st

        if train_data.empty or val_data.empty or test_data.empty:
            warnings.warn("One or more data splits (train, val, test) are empty. "
                          "Check date ranges and data availability.")
            # Depending on strictness, you might raise an error here
            # For now, let's allow it to proceed but it will likely fail in training/evaluation

        # Split into features and target
        # Ensure 'Target' is integer type for XGBoost classification
        X_train = train_data[self.features]
        y_train = train_data["Target"].astype(int)

        X_val = val_data[self.features]
        y_val = val_data["Target"].astype(int)

        X_test = test_data[self.features]
        y_test = test_data["Target"].astype(int)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame,
                    y_val: pd.Series) -> xgb.XGBClassifier:
        """
        Train the XGBoost model with early stopping.
        """
        # Default parameters if None provided (can be tuned externally)
        if self.params is None:
            self.params = {
                "objective": "multi:softprob",  # For multiclass probability
                "num_class": 3,  # Home Win, Draw, Away Win
                "eval_metric": "mlogloss",
                "early_stopping_rounds": 10,
                "use_label_encoder": False,  # Deprecated, set to False
                "verbosity": 1,
                # Add other common XGBoost params like eta, max_depth, subsample, colsample_bytree
                # These would ideally come from hyperparameter tuning
                "eta": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            }
        else:  # Ensure essential params are present if user provides some
            self.params.setdefault("objective", "multi:softprob")
            self.params.setdefault("num_class", 3)
            self.params.setdefault("eval_metric", "mlogloss")
            self.params.setdefault("use_label_encoder", False)

        # XGBClassifier takes hyperparameters directly, not nested under 'params'
        # Unpack the params dictionary for the constructor
        model_params = self.params.copy()
        # early_stopping_rounds is a fit param, not init param for XGBClassifier directly
        # some params like 'num_class' are inferred if objective is multi:*

        self.model = xgb.XGBClassifier(**model_params)

        # Train model with early stopping
        # Note: 'early_stopping_rounds' is passed to fit() method
        fit_params = {}
        if "early_stopping_rounds" in self.params:
            fit_params["early_stopping_rounds"] = self.params["early_stopping_rounds"]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,  # Can be set to True or an int for verbosity during training
            **fit_params
        )
        return self.model

    def evaluate_model(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the model on validation and test sets.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Predict on validation set
        val_probs = self.model.predict_proba(X_val)  # Probabilities for [class_0, class_1, class_2]
        val_preds = np.argmax(val_probs, axis=1)  # Predicted class (0, 1, or 2)

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
                "classification_report": classification_report(y_val, val_preds, output_dict=True, zero_division=0)
            },
            "test": {
                "accuracy": accuracy_score(y_test, test_preds),
                "f1_macro": f1_score(y_test, test_preds, average="macro"),
                "f1_weighted": f1_score(y_test, test_preds, average="weighted"),
                "confusion_matrix": confusion_matrix(y_test, test_preds),
                "classification_report": classification_report(y_test, test_preds, output_dict=True, zero_division=0)
            }
        }

        # Print key performance metrics (similar to before)
        print("\n=== Model Performance ===")
        # ... (printing logic remains the same) ...
        print(f"\nValidation Metrics:")
        print(f"Accuracy: {self.performance['validation']['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {self.performance['validation']['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {self.performance['validation']['f1_macro']:.4f}")

        print(f"\nTest Metrics:")
        print(f"Accuracy: {self.performance['test']['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {self.performance['test']['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {self.performance['test']['f1_macro']:.4f}")

        return self.performance

    def analyse_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Analyse and visualise feature importance. (Largely unchanged)
        """
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None  # Return None instead of just printing

        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({  # Renamed to avoid conflict
            "Feature": self.features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(12, max(8, len(self.features) * 0.3)))  # Dynamic height
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
        plt.title("Feature Importance", fontsize=16)
        plt.tight_layout()
        # Ensure output directory for figures exists
        os.makedirs(os.path.join(self.output_path, "figures"), exist_ok=True)
        plt.savefig(f"{self.output_path}/figures/feature_importance.png")
        plt.close()  # Close plot to free memory

        print("\n=== Top Features ===")  # Print more than 5 if available
        print(feature_importance_df.head(min(10, len(feature_importance_df))))

        return feature_importance_df

    def process_match_predictions(self, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, float]:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # X_test comes from prepare_data. Assume it has the correct rows (features)
        # and its index corresponds to the original self.df rows for the test set.

        print(f"DEBUG process_match_predictions: Shape of input X_test: {X_test.shape}")

        # Get the full rows from self.df that X_test represents
        # This ensures all original columns are present and perfectly aligned with X_test
        if X_test.empty:
            warnings.warn("X_test is empty in process_match_predictions.")
            return pd.DataFrame(), 0.0

        test_data_full_aligned = self.df.loc[X_test.index].copy()

        print(
            f"DEBUG process_match_predictions: Shape of test_data_full_aligned (from X_test.index): {test_data_full_aligned.shape}")

        if len(X_test) != len(test_data_full_aligned):
            # This should ideally not happen if X_test.index is valid for self.df
            raise ValueError("Critical misalignment: X_test.index could not fully re-select rows from self.df.")

        test_probs_array = self.model.predict_proba(X_test)

        # Create probs_df using the index of test_data_full_aligned (which is same as X_test.index)
        probs_df = pd.DataFrame(test_probs_array,
                                columns=["home_win_prob", "draw_prob", "away_win_prob"],
                                index=test_data_full_aligned.index)

        # Join probabilities to the aligned full data
        match_predictions_df = test_data_full_aligned.join(probs_df)

        print(
            f"DEBUG process_match_predictions: Shape of match_predictions_df after join: {match_predictions_df.shape}")
        print(
            f"DEBUG process_match_predictions: NaNs in away_win_prob after join: {match_predictions_df['away_win_prob'].isnull().sum()}")
        if match_predictions_df['away_win_prob'].isnull().any():
            warnings.warn("NaNs found in probability columns after join! Check index alignment.")
            # print(match_predictions_df[match_predictions_df['away_win_prob'].isnull()].head())

        # Determine predicted result string
        # Ensure no NaNs in probabilities before argmax, or handle it
        # If probabilities have NaNs, argmax on that row might be problematic
        # For rows with NaN probs, predicted_result_code might become NaN or raise error
        # Safest: only calculate for rows where probs are not NaN
        valid_prob_rows = match_predictions_df[["home_win_prob", "draw_prob", "away_win_prob"]].notna().all(axis=1)

        match_predictions_df["predicted_result_code"] = np.nan  # Initialize
        if valid_prob_rows.any():  # if there are any valid rows
            match_predictions_df.loc[valid_prob_rows, "predicted_result_code"] = np.argmax(
                match_predictions_df.loc[valid_prob_rows, ["home_win_prob", "draw_prob", "away_win_prob"]].values,
                axis=1
            )

        map_code_to_string = {0: "home_win", 1: "draw", 2: "away_win"}  # Consistent with Target: 0:Away, 1:Draw, 2:Home
        match_predictions_df["predicted_result"] = match_predictions_df["predicted_result_code"].map(map_code_to_string)
        match_predictions_df["actual_result"] = match_predictions_df["Target"].map(map_code_to_string)

        # Calculate accuracy only on rows where prediction was possible
        # Or on all rows if you want to penalize for failed predictions (NaNs)
        valid_predictions_df = match_predictions_df.dropna(subset=["predicted_result", "actual_result"])
        if not valid_predictions_df.empty:
            correct_predictions = (
                        valid_predictions_df["actual_result"] == valid_predictions_df["predicted_result"]).sum()
            total_matches_for_accuracy = len(valid_predictions_df)
            accuracy = (
                        correct_predictions / total_matches_for_accuracy * 100) if total_matches_for_accuracy > 0 else 0.0
        else:
            correct_predictions = 0
            total_matches_for_accuracy = 0
            accuracy = 0.0

        print(f"\n=== Match Prediction Results (on Test Set) ===")
        print(f"Total test samples from X_test: {len(X_test)}")
        print(f"Matches with valid probabilities for prediction: {int(valid_prob_rows.sum())}")
        print(f"Matches with valid actual & predicted results for accuracy calc: {total_matches_for_accuracy}")
        print(f"Correct discrete predictions: {correct_predictions}/{total_matches_for_accuracy}")
        print(f"Discrete prediction accuracy: {accuracy:.2f}%")

        output_cols = ['Date', 'Time', 'Comp', 'Round', 'Day', 'Home_Team', 'Away_Team',
                       'Target', 'actual_result', 'predicted_result',
                       'home_win_prob', 'draw_prob', 'away_win_prob']
        output_cols_exist = [col for col in output_cols if col in match_predictions_df.columns]
        match_predictions_df_output = match_predictions_df[output_cols_exist]

        os.makedirs(os.path.join(self.output_path, "results"), exist_ok=True)
        match_predictions_df_output.to_csv(f"{self.output_path}/results/match_predictions.csv", index=False)

        return match_predictions_df_output, accuracy

    def plot_prediction_distribution(self, match_df: pd.DataFrame) -> None:
        """
        Plot the distribution of correct vs incorrect predictions by probability.
        'match_df' should be the output from process_match_predictions.
        """
        if not all(col in match_df.columns for col in
                   ["home_win_prob", "draw_prob", "away_win_prob", "actual_result", "predicted_result"]):
            print("Skipping plot_prediction_distribution: DataFrame is missing required probability or result columns.")
            return

        # Get max probability for each prediction
        match_df["max_prob"] = match_df[["home_win_prob", "draw_prob", "away_win_prob"]].max(axis=1)
        match_df["is_correct"] = match_df["actual_result"] == match_df["predicted_result"]

        # Separate correct and incorrect predictions
        correct_preds = match_df.loc[match_df["is_correct"], "max_prob"]
        incorrect_preds = match_df.loc[~match_df["is_correct"], "max_prob"]

        # Plot histogram (largely unchanged, ensure directory exists)
        # ... (plotting code remains similar, ensure save path is correct) ...
        bins = np.linspace(0.3, 1.0, 25)  # Min prob for a 3-class is ~0.33
        plt.figure(figsize=(12, 8))

        plt.hist(incorrect_preds, bins=bins, alpha=0.5, color="red",
                 edgecolor="#1E212A", label="Incorrect Predictions")
        plt.hist(correct_preds, bins=bins, alpha=0.5, color="green",
                 edgecolor="#1E212A", label="Correct Predictions")

        plt.xlabel("Prediction Confidence (Max Probability of Predicted Outcome)")
        plt.ylabel("Number of Matches")
        plt.title(f"Prediction Accuracy by Confidence Level",
                  fontsize=16, fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        os.makedirs(os.path.join(self.output_path, "figures"), exist_ok=True)
        plt.savefig(f"{self.output_path}/figures/prediction_distribution.png")
        plt.close()

    def analyse_prediction_skewness(self, match_df: pd.DataFrame) -> Optional[tuple[pd.Series, pd.Series]]:
        """
        Analyse the skewness of the model's predictions.
        'match_df' should be the output from process_match_predictions.
        """
        if not all(col in match_df.columns for col in ["predicted_result", "actual_result"]):
            print("Skipping analyse_prediction_skewness: DataFrame is missing result columns.")
            return None

        # Calculate skewness of predictions (largely unchanged)
        # ... (skewness analysis and plotting code remains similar, ensure save path) ...
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
        categories = ["home_win", "draw", "away_win"]  # Ensure this order matches your Target mapping if needed

        predicted_values = [prediction_counts.get(cat, 0) for cat in categories]
        actual_values = [actual_counts.get(cat, 0) for cat in categories]

        ax.bar(x - width / 2, predicted_values, width, label="Predicted")
        ax.bar(x + width / 2, actual_values, width, label="Actual")

        ax.set_ylabel("Frequency")
        ax.set_title("Prediction Distribution vs Actual Distribution")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        plt.tight_layout()
        os.makedirs(os.path.join(self.output_path, "figures"), exist_ok=True)
        plt.savefig(f"{self.output_path}/figures/prediction_skewness.png")
        plt.close()

        # Probability skewness
        if all(col in match_df.columns for col in ["home_win_prob", "draw_prob", "away_win_prob"]):
            probability_skewness = {
                "home_win_prob": match_df["home_win_prob"].skew(),
                "draw_prob": match_df["draw_prob"].skew(),
                "away_win_prob": match_df["away_win_prob"].skew()
            }
            print("\nProbability Skewness (statistical):")
            for outcome, skew_val in probability_skewness.items():  # Renamed skew to skew_val
                print(f"{outcome}: {skew_val:.4f}")
        else:
            print("\nSkipping probability skewness: probability columns missing.")

        return prediction_counts, actual_counts

    def run_pipeline(self) -> dict:
        """
        Run the complete model pipeline from data preparation to evaluation.
        """
        print("=== Starting Football Match Prediction Pipeline (Single Model Version) ===")

        if self.df.empty:
            print("ERROR: DataFrame is empty. Cannot run pipeline.")
            return {"error": "DataFrame empty"}

        # Prepare data
        print("\nPreparing data splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()

        if X_train.empty or X_val.empty or X_test.empty:
            print("ERROR: One or more data splits are empty. Pipeline cannot continue.")
            print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
            return {"error": "Empty data splits"}

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
        # Pass X_test and y_test to process_match_predictions
        match_predictions_df, discrete_accuracy = self.process_match_predictions(X_test, y_test)

        # Visualisation
        if not match_predictions_df.empty:
            print("\nGenerating visualisations...")
            self.plot_prediction_distribution(match_predictions_df)
            self.analyse_prediction_skewness(match_predictions_df)
        else:
            print("\nSkipping visualisations as match_predictions_df is empty.")

        print(f"\n=== Pipeline Complete ===")
        if "test" in self.performance and "accuracy" in self.performance["test"]:
            print(f"Model accuracy on test set (from evaluate_model): {self.performance['test']['accuracy']:.4f}")
        print(f"Discrete prediction accuracy (from process_match_predictions): {discrete_accuracy:.2f}%")

        return {
            "performance": self.performance,
            "match_predictions": match_predictions_df  # This is now the direct prediction output
        }


if __name__ == "__main__":
  print ("temp")
