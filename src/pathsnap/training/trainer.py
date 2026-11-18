from pathlib import Path
import json, time
import joblib
import pandas as pd
from .evaluation import classification_metrics
from .splits import group_splits
from ..models.estimators.logistic_regression import build_logistic_regression


def train_exploratory(frame: pd.DataFrame, output: Path, n_splits: int = 5) -> dict:
    features=[c for c in frame.columns if c in {"distance_m","heading_difference_deg","speed_mps","time_gap_s","movement_m","segment_length_m","distance_to_start_m","distance_to_end_m","start_degree","end_degree"}]
    data=frame.copy(); data[features]=data[features].replace([float("inf"),float("-inf")],float("nan")).fillna(0.0)
    model=build_logistic_regression(); fold_metrics=[]; start=time.perf_counter()
    for train_idx,val_idx in group_splits(data.trip_id.to_numpy(), n_splits):
        model.fit(data.iloc[train_idx][features],data.iloc[train_idx].label)
        pred=model.predict(data.iloc[val_idx][features]); prob=model.predict_proba(data.iloc[val_idx][features])
        fold_metrics.append(classification_metrics(data.iloc[val_idx].label,pred,prob))
    model.fit(data[features],data.label); output.mkdir(parents=True,exist_ok=True); joblib.dump(model,output/"logistic_regression.joblib")
    result={"experiment":"EXPLORATORY","features":features,"folds":fold_metrics,"training_seconds":time.perf_counter()-start,"rows":len(data),"trip_ids":sorted(data.trip_id.unique().tolist())}
    (output/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
