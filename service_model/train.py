import re
import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.metrics import recall_score, f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class DateCyclicalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        dt_col = pd.to_datetime(X.iloc[:, 0], errors='coerce').fillna(pd.Timestamp('2000-01-01'))
        df_out = pd.DataFrame(index=X.index)
        df_out['year'] = dt_col.dt.year
        df_out['is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
        month = dt_col.dt.month
        dow = dt_col.dt.dayofweek
        df_out['month_sin'] = np.sin(2 * np.pi * month / 12)
        df_out['month_cos'] = np.cos(2 * np.pi * month / 12)
        df_out['day_sin'] = np.sin(2 * np.pi * dow / 7)
        df_out['day_cos'] = np.cos(2 * np.pi * dow / 7)
        return df_out
    def get_feature_names_out(self, input_features=None):
        return np.array(['year', 'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos'])

class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10.0, skip_pattern='flag_'):
        self.threshold = threshold
        self.skip_pattern = skip_pattern
        self.columns_to_keep_ = None
    def fit(self, X, y=None):
        curr_X = pd.DataFrame(X)
        curr_X.columns = curr_X.columns.astype(str)
        curr_X = curr_X.copy()
        cols_to_skip = [c for c in curr_X.columns if
                        any(pref in c for pref in ['flag_', 'cat_tr__', 'binary_tr__', 'ME', 'MA'])
                        or curr_X[c].nunique() <= 2]
        cols_to_check = [c for c in curr_X.columns if c not in cols_to_skip]
        if not cols_to_check:
            self.columns_to_keep_ = curr_X.columns
            return self
        if len(curr_X) > 1000:
            check_df = curr_X[cols_to_check].sample(n=1000, random_state=42)
        else:
            check_df = curr_X[cols_to_check]
        check_df = check_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        check_df['_const'] = 1
        while True:
            columns = check_df.columns
            if len(columns) <= 1: break
            vif = [variance_inflation_factor(check_df.values, i) for i in range(check_df.shape[1])]
            max_vif = max(vif[:-1])
            if max_vif > self.threshold:
                max_idx = vif.index(max_vif)
                check_df.drop(columns[max_idx], axis=1, inplace=True)
            else:
                break
        final_numeric_cols = check_df.drop('_const', axis=1).columns.tolist()
        self.columns_to_keep_ = final_numeric_cols + cols_to_skip
        return self
    def transform(self, X):
        curr_X = pd.DataFrame(X)
        curr_X.columns = curr_X.columns.astype(str)
        return curr_X[self.columns_to_keep_]

def f_get_df_types_3(df: pd.DataFrame, data_type: str) -> list:
    if data_type == 'int':
        df_out = df.select_dtypes(include=['int', 'int64', 'float', 'float64', 'int32', 'float32'])
    elif data_type == 'str':
        df_out = df.select_dtypes(include=['object', 'string'])
    elif data_type == 'date':
        df_out = df.select_dtypes(include=['datetime64', 'datetimetz'])
    else:
        df_out = pd.DataFrame()
    return list(df_out.columns)


filename = 'df_in.xlsx'
data_path = os.path.join(BASE_DIR, filename)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at: {data_path}")

df_in = pd.read_excel(data_path)
df_temp = df_in.copy()

lst_all = [col for col in df_temp.columns if col != 'flag_status_event']
lst_to_drop = ['status']
binary_prefixes = r'^(flag_|ME|MA|HC|PP|KA|OF|OS|OT|ALL)'
lst_binary = [col for col in lst_all if re.match(binary_prefixes, col)]
lst_category = f_get_df_types_3(df_temp, 'str')
lst_category = [c for c in lst_category if c not in lst_binary and c not in lst_to_drop]
lst_date = f_get_df_types_3(df_temp, 'date')
lst_skew = ['duration_event', 'duration_process', 'cnt_guest', 'cnt_place',
            'event_budget', 'cnt_meals', 'cnt_manager']
used_cols = set(lst_binary) | set(lst_category) | set(lst_date) | set(lst_skew) | set(lst_to_drop)
lst_pass = [col for col in lst_all if col not in used_cols]

preprocessor = ColumnTransformer(transformers=[
    ('date_tr', DateCyclicalTransformer(), lst_date),
    ('cat_tr', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), lst_category),
    ('log_tr', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'))]), lst_skew),
    ('binary_tr', SimpleImputer(strategy='constant', fill_value=0), lst_binary),
    ('num_pass', 'passthrough', lst_pass)])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('vif_filter', VIFSelector(threshold=5.0)),
    ('clf', LogisticRegression(
        C=1.0,
        solver='liblinear',
        random_state=42,
        class_weight='balanced'))])

y = df_temp['flag_status_event']
X = df_temp.drop('flag_status_event', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')

with mlflow.start_run():
    mlflow.log_metric("f1_macro", f1)
    mlflow.sklearn.log_model(model_pipeline, "model")
    model_save_path = os.path.join(BASE_DIR, 'full_model_pipeline.joblib')
    joblib.dump(model_pipeline, model_save_path)

print(f"F1 Score: {f1:.4f}")
print(f"[OK] Model has been saved to: {model_save_path}")