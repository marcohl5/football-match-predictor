import pandas as pd
import numpy as np  # For np.log in hyperopt space if needed
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import json
import os
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # For XGBoost related future warnings


class XGBoostHyperparameterTuner:
    """
    Hyperparameter tuning for the XGBoost football prediction model (multiclass).
    """

    def __init__(self,
                 pred_cols: List[str],
                 df: pd.DataFrame,
                 league_name: str,
                 data_path: Optional[str] = None,  # Made data_path optional
                 output_path: str = "./output/",
                 max_evals: int = 100):
        """
        Initialise the hyperparameter tuner.
        """
        self.data_path = data_path
        self.output_path = output_path
        self.max_evals = max_evals
        self.df = df
        self.league_name = league_name
        self.features = pred_cols

        if self.df is None:
            if self.data_path:
                print(f"Loading data from {self.data_path}")
                self.df = pd.read_csv(self.data_path)
                if 'Date' in self.df.columns:  # Ensure Date is datetime if loaded
                    self.df['Date'] = pd.to_datetime(self.df['Date'])
            else:
                raise ValueError("DataFrame 'df' or 'data_path' must be provided.")

        if self.df.empty:
            raise ValueError("Provided DataFrame 'df' is empty.")

        if not self.features:
            # It's better to require pred_cols than to have a default that might not match.
            raise ValueError("Parameter 'pred_cols' must be provided and cannot be empty.")

        missing_features = [f for f in self.features if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"Features specified in 'pred_cols' are missing from the DataFrame: {missing_features}")

        if "Target" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'Target' column.")

        os.makedirs(self.output_path, exist_ok=True)

    def prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and validation sets based on dates.
        Returns: (X_train, X_val, y_train, y_val)
        """
        # Ensure 'Date' column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Training set: Seasons 2020-2021 to 2022-2023
        train_data = self.df[(self.df["Date"] < "2023-08-01") &
                             (self.df["Date"] > "2020-08-01")].copy()

        # Validation set: First half of 2023-2024 season
        val_data = self.df[(self.df["Date"] < "2024-01-01") &
                           (self.df["Date"] >= "2023-08-01")].copy()

        if train_data.empty or val_data.empty:
            warnings.warn("Training or Validation data split is empty. "
                          "Check date ranges and data availability for tuning.")
            # Return empty DataFrames/Series if splits are empty to avoid errors downstream in objective
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='int'), pd.Series(dtype='int')

        X_train = train_data[self.features]
        y_train = train_data["Target"].astype(int)  # Ensure Target is int

        X_val = val_data[self.features]
        y_val = val_data["Target"].astype(int)  # Ensure Target is int

        return X_train, X_val, y_train, y_val

    def objective_function(self, space: dict) -> dict:
        """
        Objective function for hyperparameter optimization.
        """
        X_train, X_val, y_train, y_val = self.prepare_data()

        if X_train.empty or X_val.empty:
            return {"loss": float('inf'), "status": STATUS_OK, "message": "Empty data split"}

        # Define early_stopping_rounds for tuning - can be fixed or part of 'space'
        fixed_early_stopping_rounds = 10  # Or space.get('early_stopping_rounds', 10)

        model_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_estimators': int(space.get("n_estimators", 180)),
            'learning_rate': space.get("learning_rate", 0.1),
            'max_depth': int(space["max_depth"]),
            'gamma': space["gamma"],
            'reg_alpha': int(space.get("reg_alpha", 0)),
            'reg_lambda': space.get("reg_lambda", 1),
            'min_child_weight': int(space["min_child_weight"]),
            'colsample_bytree': space["colsample_bytree"],
            'subsample': space.get("subsample", 1.0),
            'seed': space.get("seed", 42),
            'eval_metric': "mlogloss",
            'use_label_encoder': False,
            'early_stopping_rounds': fixed_early_stopping_rounds  # ADDED HERE for __init__
        }

        clf = xgb.XGBClassifier(**model_params)

        # For early stopping, provide eval_set.
        # The early_stopping_rounds from model_params will be used by fit.
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],  # Use validation set for early stopping evaluation
            verbose=False
            # No explicit early_stopping_rounds kwarg in fit()
        )

        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        loss = -accuracy

        return {"loss": loss, "status": STATUS_OK, "accuracy": accuracy}

    def run_tuning(self) -> dict:
        """
        Run the hyperparameter tuning process.
        """
        print("=== Starting XGBoost Hyperparameter Tuning (Multiclass) ===")

        # Define hyperparameter space
        space = {
            "n_estimators": hp.quniform("n_estimators", 100, 500, 50),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
            "max_depth": hp.quniform("max_depth", 3, 10, 1),
            "gamma": hp.uniform("gamma", 0, 5),
            "reg_alpha": hp.quniform("reg_alpha", 0, 100, 1),  # L1 regularization
            "reg_lambda": hp.uniform("reg_lambda", 0, 5),  # L2 regularization
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
            "subsample": hp.uniform("subsample", 0.6, 1.0),  # Row subsampling
            "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
            "seed": 42  # Fixed seed for reproducibility of XGBoost within a trial
        }

        trials = Trials()

        print(f"\nRunning optimization with {self.max_evals} evaluations...")
        best_hyperparams = fmin(
            fn=self.objective_function,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(42)  # For reproducibility of hyperopt's search process
        )

        # fmin returns the values that minimize the loss.
        # These are the optimal hyperparameter values.
        # Ensure types are correct for XGBoost
        for param_name in ["max_depth", "min_child_weight", "reg_alpha", "n_estimators"]:
            if param_name in best_hyperparams:
                best_hyperparams[param_name] = int(best_hyperparams[param_name])

        # Find the trial corresponding to the best result to get accuracy
        # and other metrics if you stored them in the trial result
        best_trial_loss = float('inf')
        best_trial_accuracy = 0.0
        if trials.trials:  # Check if trials list is not empty
            valid_trials = [t for t in trials.trials if t['result']['status'] == STATUS_OK and 'loss' in t['result']]
            if valid_trials:
                best_trial = min(valid_trials, key=lambda t: t['result']['loss'])
                best_trial_loss = best_trial['result']['loss']
                best_trial_accuracy = best_trial['result'].get('accuracy', -best_trial_loss)  # Get accuracy if stored

        print(
            f"\nBest validation accuracy achieved during tuning: {best_trial_accuracy:.4f} (Loss: {best_trial_loss:.4f})")
        print("\n=== Best Hyperparameters Found ===")
        for param, value in best_hyperparams.items():
            print(f"{param}: {value}")

        # Prepare the full parameter set for the model
        final_model_params = best_hyperparams.copy()
        final_model_params['objective'] = 'multi:softprob'
        final_model_params['num_class'] = 3
        final_model_params['eval_metric'] = 'mlogloss'
        final_model_params['use_label_encoder'] = False
        # 'early_stopping_rounds' is a fit parameter, typically added when training the final model
        # final_model_params['early_stopping_rounds'] = 10 # Add this if you want it in the JSON for model init

        # Save best parameters to file
        league_name_strip = self.league_name.lower().replace(" ", "")
        output_file_path = os.path.join(self.output_path, f"tuned_params_{league_name_strip}.json")
        with open(output_file_path, "w") as f:
            json.dump(final_model_params, f, indent=4)
        print(f"Best parameters saved to {output_file_path}")

        return final_model_params  # Return the full set for model instantiation


if __name__ == "__main__":
    # For isolated running
    tuner = XGBoostHyperparameterTuner(data_path="../processed-data/matches/processed_seriea_matches.csv",
                                       pred_cols=None,
                                       df=None,
                                       league_name="Bundesliga",
                                       output_path="../output",
                                       max_evals=100)
    best_params = tuner.run_tuning()


