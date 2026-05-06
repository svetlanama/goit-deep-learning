"""
Tree-models training stage (LightGBM + XGBoost + CatBoost) — запускається в окремому
Python-процесі з ноутбуку через subprocess.run(), щоб уникнути OpenMP-deadlock
між libomp PyTorch і libomp LightGBM на Apple Silicon.

Очікує заздалегідь обчислені feature matrices у cache/ та CSV-файли в DATA_DIR.

Usage:
    python train_trees.py --cache_dir cache --data_dir data \
        --submission submission_dz_final.csv --n_trials 25
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import optuna
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.optimize import minimize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--submission", default="submission_dz_final.csv")
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--hpo_estimators", type=int, default=1200,
                   help="n_estimators під час Optuna HPO (менше — швидше)")
    p.add_argument("--cv_estimators",  type=int, default=2500,
                   help="n_estimators під час фінального 5-fold CV")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2))
    return p.parse_args()


def find_csv(data_dir: Path, name: str) -> Path:
    hits = sorted(data_dir.glob(f"**/{name}"))
    if not hits:
        raise FileNotFoundError(f"{name} not in {data_dir}")
    return hits[0]


def main():
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cache = Path(args.cache_dir)
    data_dir = Path(args.data_dir)

    print(f"=== train_trees.py ===")
    print(f"  threads:    {args.threads}")
    print(f"  n_trials:   {args.n_trials}")
    print(f"  n_folds:    {args.n_folds}")
    print(f"  seed:       {args.seed}")

    # 1) Завантажити cached features
    img_train = np.load(cache / "img_train.npy")
    img_test  = np.load(cache / "img_test.npy")
    txt_train = np.load(cache / "txt_train.npy")
    txt_test  = np.load(cache / "txt_test.npy")
    tab_train = np.load(cache / "tab_train.npy")
    tab_test  = np.load(cache / "tab_test.npy")

    X_train = np.hstack([img_train, txt_train, tab_train]).astype(np.float32)
    X_test  = np.hstack([img_test,  txt_test,  tab_test ]).astype(np.float32)

    train_csv = find_csv(data_dir, "train.csv")
    test_csv  = find_csv(data_dir, "test.csv")
    train_df  = pd.read_csv(train_csv)
    test_df   = pd.read_csv(test_csv)
    y_train   = train_df["AdoptionSpeed"].astype(np.float32).values
    y_strat   = train_df["AdoptionSpeed"].astype(int).values

    print(f"\nX_train: {X_train.shape}  X_test: {X_test.shape}  y: {y_train.shape}")

    # 2) Helpers
    K = args.num_classes
    LABELS = list(range(K))

    def qwk(y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=LABELS)

    class OptimizedRounder:
        def __init__(self, K=K):
            self.K = K
            self.coef_ = None

        @staticmethod
        def _to_int(x, c, K):
            c = sorted(c)
            out = np.full_like(x, K - 1, dtype=int)
            for k in range(K - 2, -1, -1):
                out[x < c[k]] = k
            return out

        def _loss(self, c, x, y):
            return -qwk(y, self._to_int(x, c, self.K))

        def fit(self, x, y):
            inits = [
                list(np.linspace(0.5, self.K - 1.5, self.K - 1)),
                list(np.quantile(x, np.linspace(1.0 / self.K, 1 - 1.0 / self.K, self.K - 1))),
                list(np.quantile(x, np.linspace(0.5 / self.K, 1 - 0.5 / self.K, self.K - 1))),
            ]
            best, best_score = None, -np.inf
            for init in inits:
                res = minimize(self._loss, init, args=(x, y), method="Nelder-Mead",
                               options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400})
                s = -res.fun
                if s > best_score:
                    best_score, best = s, res.x
            self.coef_ = sorted(best)
            return self

        def predict(self, x):
            return self._to_int(x, self.coef_, self.K)

    # 3) Optuna HPO (hold-out 20%)
    X_tr, X_va, y_tr, y_va, ys_tr, ys_va = train_test_split(
        X_train, y_train, y_strat, test_size=0.2, stratify=y_strat, random_state=args.seed
    )

    def hpo_qwk(y_va, pred):
        r = OptimizedRounder().fit(pred, y_va.astype(int))
        return qwk(y_va.astype(int), r.predict(pred))

    def lgbm_obj(trial):
        p = dict(
            objective="regression", metric="rmse", verbosity=-1,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            n_estimators=args.hpo_estimators, random_state=args.seed,
            n_jobs=args.threads, num_threads=args.threads,
        )
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        return hpo_qwk(y_va, m.predict(X_va))

    def xgb_obj(trial):
        p = dict(
            objective="reg:squarederror", eval_metric="rmse",
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            n_estimators=args.hpo_estimators, random_state=args.seed, tree_method="hist",
            n_jobs=args.threads,
        )
        m = xgb.XGBRegressor(**p, early_stopping_rounds=50, verbosity=0)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return hpo_qwk(y_va, m.predict(X_va))

    def cb_obj(trial):
        p = dict(
            loss_function="RMSE",
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
            iterations=args.hpo_estimators, random_seed=args.seed, verbose=False,
            thread_count=args.threads,
        )
        m = CatBoostRegressor(**p, early_stopping_rounds=50)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        return hpo_qwk(y_va, m.predict(X_va))

    best_params = {}
    for name, obj in [("lgbm", lgbm_obj), ("xgb", xgb_obj), ("cat", cb_obj)]:
        print(f"\n=== Optuna {name} ({args.n_trials} trials) ===")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=args.seed))
        t0 = time.time()
        study.optimize(obj, n_trials=args.n_trials, show_progress_bar=False)
        print(f"   best QWK = {study.best_value:.4f}  |  {time.time()-t0:.1f}s")
        best_params[name] = study.best_params

    Path("best_params.json").write_text(json.dumps(best_params, indent=2))

    # 4) 5-fold CV
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    def lgbm_fp(X_tr, y_tr, X_va, y_va, X_te):
        p = dict(best_params["lgbm"], objective="regression", metric="rmse",
                 verbosity=-1, n_estimators=args.cv_estimators, random_state=args.seed,
                 n_jobs=args.threads, num_threads=args.threads)
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)])
        return m.predict(X_va), m.predict(X_te)

    def xgb_fp(X_tr, y_tr, X_va, y_va, X_te):
        p = dict(best_params["xgb"], objective="reg:squarederror", eval_metric="rmse",
                 n_estimators=args.cv_estimators, random_state=args.seed, tree_method="hist",
                 n_jobs=args.threads)
        m = xgb.XGBRegressor(**p, early_stopping_rounds=75, verbosity=0)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m.predict(X_va), m.predict(X_te)

    def cb_fp(X_tr, y_tr, X_va, y_va, X_te):
        p = dict(best_params["cat"], loss_function="RMSE",
                 iterations=args.cv_estimators, random_seed=args.seed, verbose=False,
                 thread_count=args.threads)
        m = CatBoostRegressor(**p, early_stopping_rounds=75)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        return m.predict(X_va), m.predict(X_te)

    fit_fns = {"lgbm": lgbm_fp, "xgb": xgb_fp, "cat": cb_fp}
    oof = {k: np.zeros(len(X_train), dtype=np.float32) for k in fit_fns}
    test_preds = {k: np.zeros(len(X_test), dtype=np.float32) for k in fit_fns}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_strat), 1):
        print(f"\n--- Fold {fold}/{args.n_folds} ---")
        Xt, Xv = X_train[tr_idx], X_train[va_idx]
        yt, yv = y_train[tr_idx], y_train[va_idx]
        for name, fn in fit_fns.items():
            t0 = time.time()
            oof_pred, te_pred = fn(Xt, yt, Xv, yv, X_test)
            oof[name][va_idx] = oof_pred
            test_preds[name] += te_pred / args.n_folds
            r = OptimizedRounder().fit(oof_pred, yv.astype(int))
            sc = qwk(yv.astype(int), r.predict(oof_pred))
            print(f"   {name:5s} fold-QWK = {sc:.4f}  |  {time.time()-t0:.1f}s")

    # 5) Blend + thresholds
    oof_stack  = np.column_stack([oof[k]        for k in fit_fns])
    test_stack = np.column_stack([test_preds[k] for k in fit_fns])

    def neg_blended_qwk(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return 0.0
        w = w / w.sum()
        b = oof_stack @ w
        r = OptimizedRounder().fit(b, y_train.astype(int))
        return -qwk(y_train.astype(int), r.predict(b))

    best_w, best_score = None, np.inf
    for init in [[1/3]*3, [0.5,0.25,0.25], [0.25,0.5,0.25], [0.25,0.25,0.5]]:
        res = minimize(neg_blended_qwk, init, method="Nelder-Mead",
                       options={"xatol":1e-3, "fatol":1e-4, "maxiter":200})
        if res.fun < best_score:
            best_score, best_w = res.fun, res.x
    weights = np.clip(best_w, 0, None); weights = weights / weights.sum()
    oof_blend  = oof_stack  @ weights
    test_blend = test_stack @ weights
    print(f"\nBlend weights: lgbm={weights[0]:.3f} xgb={weights[1]:.3f} cat={weights[2]:.3f}")
    print(f"Blended-OOF QWK: {-best_score:.4f}")

    rounder = OptimizedRounder().fit(oof_blend, y_train.astype(int))
    oof_int = rounder.predict(oof_blend)
    final_qwk = qwk(y_train.astype(int), oof_int)
    print(f"\nFinal OOF QWK = {final_qwk:.4f}")
    print(f"Cut-points:    {[f'{c:.3f}' for c in rounder.coef_]}")

    np.save(cache / "oof_blend.npy", oof_blend)
    np.save(cache / "test_blend.npy", test_blend)

    # 6) Submission
    test_int = rounder.predict(test_blend)
    sub = pd.DataFrame({"PetID": test_df["PetID"], "AdoptionSpeed": test_int.astype(int)})
    sub.to_csv(args.submission, index=False)
    print(f"\n💾 Saved → {args.submission}  ({sub.shape[0]} rows)")
    print(sub["AdoptionSpeed"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
