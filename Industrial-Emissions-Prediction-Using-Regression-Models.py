# Industrial-Emissions-Prediction-Using-Regression-Models
# -------------------------------------------------------
# Generates a synthetic industrial dataset and trains regression models to predict
# annual CO2-equivalent emissions (tons). Saves dataset, model performance metrics,
# plots, and trained model files.
#
# Outputs:
#  - ./data/industrial_emissions_synthetic.csv
#  - ./results/performance_table.csv
#  - ./results/observed_vs_pred_<model>.png
#  - ./results/feature_importances_<model>.png  (for tree models)
#  - ./models/<model>.joblib
#
# Dependencies:
#  pip install numpy pandas matplotlib seaborn scikit-learn joblib openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

RND = 42

def generate_synthetic_industrial_data(n=500, random_state=RND):
    rng = np.random.default_rng(random_state)
    plant_ids = [f"P{str(i).zfill(4)}" for i in range(n)]
    production_volume = np.round(rng.normal(200000, 80000, size=n)).clip(20000, None)
    fuel_types = rng.choice(["coal", "gas", "oil", "biomass"], size=n, p=[0.35, 0.4, 0.2, 0.05])
    operating_hours = np.round(rng.normal(7000, 800, size=n)).clip(1000, 8760)
    efficiency = np.round(np.clip(rng.normal(0.85, 0.08, size=n), 0.4, 0.98), 3)
    maintenance_score = np.round(np.clip(rng.normal(70, 15, size=n), 10, 100), 1)
    ambient_temperature_C = np.round(rng.normal(25, 8, size=n), 2)
    ambient_humidity = np.round(np.clip(rng.normal(0.6, 0.15, size=n), 0.1, 0.99), 3)

    fuel_emission_factor = np.array([2.5 if f=="coal" else 1.2 if f=="gas" else 1.8 if f=="oil" else 0.9 for f in fuel_types])
    base_emissions = production_volume * fuel_emission_factor * 1e-3
    eff_factor = (1.0 / efficiency)
    maint_factor = (1.0 + (50 - maintenance_score) / 200.0)
    hours_factor = 1.0 + (operating_hours - 7000) / 20000.0
    met_factor = 1.0 + 0.002 * (ambient_temperature_C - 20) + 0.05 * (ambient_humidity - 0.5)

    emissions = base_emissions * eff_factor * maint_factor * hours_factor * met_factor
    coal_penalty = np.where(fuel_types=="coal", 1.0 + (50 - maintenance_score)/150.0, 1.0)
    emissions *= coal_penalty
    noise = rng.normal(scale=0.05 * emissions)
    emissions = np.maximum(0.0, emissions + noise)

    df = pd.DataFrame({
        "plant_id": plant_ids,
        "production_volume_tons_per_year": production_volume,
        "fuel_type": fuel_types,
        "operating_hours_per_year": operating_hours,
        "efficiency": efficiency,
        "maintenance_score": maintenance_score,
        "ambient_temperature_C": ambient_temperature_C,
        "ambient_humidity": ambient_humidity,
        "emissions_tons_CO2eq": np.round(emissions, 3)
    })

    return df

def build_preprocessor():
    numeric_features = ["production_volume_tons_per_year", "operating_hours_per_year",
                        "efficiency", "maintenance_score", "ambient_temperature_C", "ambient_humidity"]
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_features = ["fuel_type"]
    categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop="first"))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor, numeric_features, categorical_features

def train_models(df, out_dir=".", random_state=RND):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)

    X = df.drop(columns=["plant_id", "emissions_tons_CO2eq"])
    y = df["emissions_tons_CO2eq"].values

    preprocessor, numeric_features, categorical_features = build_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    models = {
        "LinearRegression": Pipeline([("pre", preprocessor), ("lr", LinearRegression())]),
        "Ridge": Pipeline([("pre", preprocessor), ("ridge", Ridge(alpha=1.0, random_state=random_state))]),
        "RandomForest": Pipeline([("pre", preprocessor), ("rf", RandomForestRegressor(n_estimators=200, random_state=random_state))]),
        "GradientBoosting": Pipeline([("pre", preprocessor), ("gb", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=random_state))])
    }

    perf_records = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        perf_records.append({"model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        trained_models[name] = model

        plt.figure(figsize=(6,5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        mmin = min(y_test.min(), y_pred.min())
        mmax = max(y_test.max(), y_pred.max())
        plt.plot([mmin, mmax], [mmin, mmax], linestyle="--", color="red")
        plt.xlabel("Observed emissions (tons CO2-eq)")
        plt.ylabel("Predicted emissions (tons CO2-eq)")
        plt.title(f"Observed vs Predicted - {name}")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "results", f"observed_vs_pred_{name}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        if name in ["RandomForest", "GradientBoosting"]:
            pre = model.named_steps["pre"]
            num_feats = numeric_features
            ohe = pre.named_transformers_["cat"].named_steps["onehot"]
            cat_cols = ohe.get_feature_names_out(categorical_features).tolist()
            feat_names = num_feats + cat_cols
            importances = model.named_steps[name.lower() if name!="GradientBoosting" else "gb"].feature_importances_
            fi_series = pd.Series(importances, index=feat_names).sort_values(ascending=False)
            fi_path = os.path.join(out_dir, "results", f"feature_importances_{name}.png")
            plt.figure(figsize=(8,4))
            sns.barplot(x=fi_series.values, y=fi_series.index)
            plt.title(f"Feature importances - {name}")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(fi_path, dpi=150)
            plt.close()
            fi_series.to_csv(os.path.join(out_dir, "results", f"feature_importances_{name}.csv"))

        joblib.dump(model, os.path.join(out_dir, "models", f"{name}.joblib"))

    perf_df = pd.DataFrame(perf_records).set_index("model")
    perf_df.to_csv(os.path.join(out_dir, "results", "performance_table.csv"))
    print("\nModel performance:\n", perf_df.round(4))

    best_model_name = perf_df["RMSE"].idxmin()
    best_model = trained_models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    preds_df = X_test.copy().reset_index(drop=True)
    preds_df["observed_emissions"] = y_test
    preds_df["predicted_emissions"] = np.round(y_pred_best, 3)
    preds_df.to_csv(os.path.join(out_dir, "results", "test_predictions_best_model.csv"), index=False)

    return perf_df, trained_models

def main():
    out_dir = "industrial_emissions_project_output"
    os.makedirs(out_dir, exist_ok=True)

    print("Generating synthetic dataset...")
    df = generate_synthetic_industrial_data(n=500, random_state=RND)
    data_path = os.path.join(out_dir, "data", "industrial_emissions_synthetic.csv")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    print("Saved synthetic dataset to:", data_path)

    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    plt.figure(figsize=(6,4))
    sns.histplot(df["emissions_tons_CO2eq"], bins=40, kde=True)
    plt.xlabel("Emissions (tons CO2-eq)")
    plt.title("Distribution of Synthetic Emissions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "results", "emissions_distribution.png"), dpi=150)
    plt.close()

    perf_df, models = train_models(df, out_dir=out_dir)

    print("\nAll outputs saved under:", out_dir)
    print("Models saved to:", os.path.join(out_dir, "models"))
    print("Results saved to:", os.path.join(out_dir, "results"))

if __name__ == "__main__":
    main()
