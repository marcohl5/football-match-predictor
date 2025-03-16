from scripts.preprocess import BaseDataPreprocess
from scripts.model_tuning import XGBoostHyperparameterTuner
from scripts.xgboost_model import XGBoostFootballModel

# *********************** INSERT PREDICTION PARAMETERS ***********************

# League Name Parameter Input:
# 'Premier League' / 'Serie A' / 'La Liga'
league_name = "Premier League"

# 'Strength' / 'Strength_Fpl'
strength_type = "Strength_Fpl"

# *********************** Parameters that shouldn't need to be touched ***********************
match_path = "original-data/matches/"
strength_file = "processed-data/teams/processed_szn_start_elo.csv"
output_path = "processed-data/matches/"
model_output_path = "output"

# *********************** OPTIONAL custom predictor selector ***********************
predictor_columns = [
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

# strip league name
league_name_strip = league_name.lower().replace(" ", "")

# *********************** RUN ***********************

# Initialise and run preprocessing process
preprocessor = BaseDataPreprocess(
        league_name=league_name,
        match_path=match_path,
        strength_file=strength_file,
        strength_type=strength_type,
        output_path=output_path,
        file_name=f"{league_name_strip}_matches.csv",
        elo_scaling_type="log",  # input "log"/"linear"
        min_weight=0.7,  # PL performs best with 0.7, 1.3 min/max weight
        max_weight=1.3
    )
processed_data, all_predictor_cols = preprocessor.run()

# Initialise and run hyperparameter tuning
tuner = XGBoostHyperparameterTuner(data_path="",
                                   # data_path=f"processed-data/matches/processed_{league_name_strip}_matches.csv"
                                   pred_cols=all_predictor_cols,
                                   df=processed_data,
                                   output_path="output",
                                   max_evals=100)
best_params = tuner.run_tuning()
best_params.update({"early_stopping_rounds": 10})

# Initialise and run xgBoost model
model = XGBoostFootballModel(
        data_path="",
        model_output_path=model_output_path,
        params=best_params,
        pred_cols=all_predictor_cols,
        df=processed_data,
)

results = model.run_pipeline()
