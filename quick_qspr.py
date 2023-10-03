import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mol2vec(mols, model):
    if type(mols) != list:
        x_sentence = [MolSentence(mol2alt_sentence(mols, 1))]
    else:
        x_sentence = [MolSentence(mol2alt_sentence(x, 1)) for x in mols]
    x_molvec = [DfVec(x) for x in sentences2vec(x_sentence, model, unseen="UNK")]
    x_molvec = np.array([x.vec for x in x_molvec])
    return x_molvec


def objective(trial, x_train, y_train, x_valid, y_valid):
    param = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    gbm = lgb.train(
        param, lgb.Dataset(x_train, y_train), valid_sets=[lgb.Dataset(x_valid, y_valid)]
    )
    y_pred = gbm.predict(x_valid)
    return mean_squared_error(y_valid, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LGBM model from CSV file.")
    parser.add_argument(
        "-in", "--input_file", type=str, help="Path to the input CSV file."
    )
    parser.add_argument(
        "-x",
        "--x_column",
        type=str,
        help="Name of the column to be used as input features.",
    )
    parser.add_argument(
        "-y",
        "--y_column",
        type=str,
        help="Name of the column to be used as target variable.",
    )
    parser.add_argument(
        "-o",
        "--output_model",
        type=str,
        default="your_model",
        help="Name of the output model file.",
    )
    args = parser.parse_args()

    input_file = args.input_file
    x_column = args.x_column
    y_column = args.y_column
    output_model = args.output_model + ".pkl"

    df = pd.read_csv(input_file)
    mols = [Chem.MolFromSmiles(i) for i in df[x_column]]
    x = mol2vec(mols, word2vec.Word2Vec.load("./PI1M.pkl"))
    y = df[y_column].values

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    with open("scaler_model.pkl", "wb") as f:
        pickle.dump(scaler, f)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y_scaled, test_size=0.3, random_state=4396
    )

    print("start optuna")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, x_valid, y_valid),
        n_trials=100,
        n_jobs=-1,
    )

    best_params = study.best_params
    print("train with best params")

    gbm_best = lgb.train(
        best_params,
        lgb.Dataset(x_train, y_train),
        valid_sets=[lgb.Dataset(x_valid, y_valid)],
    )

    gbm_final = lgb.train(best_params, lgb.Dataset(x, y_scaled))

    y_pred = gbm_best.predict(x_valid)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_valid_inv = scaler.inverse_transform(y_valid)

    mse = mean_squared_error(y_valid_inv, y_pred_inv)
    rmse = mse**0.5
    print("RMSE:", rmse)

    r2 = r2_score(y_valid_inv, y_pred_inv)
    print("R2:", r2)

    print("saving model")
    with open(output_model, "wb") as f:
        pickle.dump(gbm_final, f)

    print("finish")
