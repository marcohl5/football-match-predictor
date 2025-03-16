## Football Match Outcome Predictor

The aim of the project (WIP) is to develop a tool able to
predict the outcome of football matches based on historical data to a reasonable degree of accuracy. 


The project includes:
* **Data Cleaning with Feature and Target Creation** - Multiclass classification (W/D/L).
* **Feature Engineering** - Weighted rolling averages and weighted statistics based on strength disparity between teams.
* **Hyperparameter Tuning** - The tool tunes the hyperparameters of the XGBoost model for each run using a validation set.
* **XGBoost Model** - Predictions for each team were calculated independently, with the probability for each match
outcome (Home Win/Draw/Away Win) joined using the joint probability of independent complementary events.

The model is built on advanced team statistics sourced from FBref, scraped using a script personally developed.
Its structure allows for predictions across any football league that provides the required advanced team statistics.

The current version of the model only generates predictions for historical data and does not support live or updated predictions. 
Predictions are available for the following leagues at the moment:
* Premier League 
* La Liga
* Serie A

#### Further Improvements
* Provide live predictions using an auto-updating script for every game week
* Compare prediction accuracy vs bookmaker odds
* Present live predictions using a streamlit interface


#### Limitations
* Not accurately accounting for player form
* Not accounting for injuries/player un-availabilities (Line-up consideration)
* Not accounting for other competitions and rest between games (Teams might rest players before a cup final)