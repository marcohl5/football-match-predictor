import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import json
import os
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class XGBoostHyperparameterTuner:
    """
    A simple class for hyperparameter tuning of the XGBoost football prediction model.
    """

    def __init__(self,
                 data_path: str,
                 pred_cols: List[str],
                 df: pd.DataFrame = None,
                 output_path="./output/",
                 max_evals=100):
        """
        Initialise the hyperparameter tuner.

        Parameters:
            data_path (str): Path to the CSV dataset
            output_path (str): Path where outputs will be saved
            max_evals (int): Maximum number of evaluations for hyperparameter tuning
        """
        self.data_path = data_path
        self.output_path = output_path
        self.max_evals = max_evals
        self.df = df

        if self.df is None or self.df.empty:
            self.df = pd.read_csv(data_path)

        self.features = pred_cols

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

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

    def prepare_data(self) -> tuple:
        """
        Split data into training and validation sets based on dates.

        Returns:
            (tuple): X_train, X_val, y_train, y_val
        """
        # Training set: Seasons 2020-2021 to 2022-2023
        train_data = self.df[(self.df["Date"] < "2023-08-01") &
                             (self.df["Date"] > "2020-08-01")]

        # Validation set: First half of 2023-2024 season
        val_data = self.df[(self.df["Date"] < "2024-01-01") &
                           (self.df["Date"] >= "2023-08-01")]

        # Split into features and target
        X_train = train_data[self.features]
        y_train = train_data["Target"]

        X_val = val_data[self.features]
        y_val = val_data["Target"]

        # print(self.features)

        return X_train, X_val, y_train, y_val

    def objective_function(self, space: dict) -> dict:
        """
        Objective function for hyperparameter optimization.

        Parameters:
            space (dict): Hyperparameter configuration to evaluate

        Returns:
            dict: Results including loss value and status
        """
        X_train, X_val, y_train, y_val = self.prepare_data()

        # Initialize XGBoost classifier with current hyperparameters
        clf = xgb.XGBClassifier(
            n_estimators=int(space["n_estimators"]),
            max_depth=int(space["max_depth"]),
            gamma=space["gamma"],
            reg_alpha=int(space["reg_alpha"]),
            reg_lambda=space["reg_lambda"],
            min_child_weight=int(space["min_child_weight"]),
            colsample_bytree=space["colsample_bytree"],
            seed=space["seed"],
            eval_metric="mlogloss",
            early_stopping_rounds=10,
            use_label_encoder=False
        )

        # Train model with early stopping
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        # Make predictions
        y_pred = clf.predict(X_val)

        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average="weighted")
        f1_macro = f1_score(y_val, y_pred, average="macro")

        # Print progress
        print(f"Accuracy: {accuracy:.4f}, F1 (weighted): {f1_weighted:.4f}, F1 (macro): {f1_macro:.4f}")

        return {"loss": -accuracy, "status": STATUS_OK}

    def run_tuning(self) -> dict:
        """
        Run the hyperparameter tuning process, saving them in a json file too.

        Returns:
            (dict): Best hyperparameters in dict
        """
        print("=== Starting XGBoost Hyperparameter Tuning ===")

        # Define hyperparameter space
        space = {
            "max_depth": hp.quniform("max_depth", 3, 18, 1),
            "gamma": hp.uniform("gamma", 1, 9),
            "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
            "n_estimators": 180,
            "seed": 42
        }

        # Initialize trials object to store results
        trials = Trials()

        # Run hyperparameter optimization
        print(f"\nRunning optimization with {self.max_evals} evaluations...")
        best_params = fmin(
            fn=self.objective_function,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials
        )

        # Convert integer hyperparameters from float to int
        for param in ["max_depth", "min_child_weight", "reg_alpha"]:
            if param in best_params:
                best_params[param] = int(best_params[param])

        print("\n=== Best Hyperparameters ===")
        for param, value in best_params.items():
            print(f"{param}: {value}")

        # Save best parameters to file
        with open(f"{self.output_path}/best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)

        return best_params


if __name__ == "__main__":
    # For isolated running
    tuner = XGBoostHyperparameterTuner(data_path="../processed-data/matches/processed_seriea_matches.csv",
                                       pred_cols=None,
                                       df=None,
                                       output_path="../output",
                                       max_evals=100)
    best_params = tuner.run_tuning()


