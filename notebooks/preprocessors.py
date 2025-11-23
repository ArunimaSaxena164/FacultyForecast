import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Your feature groups
numeric_cols = [
    'years_at_institution', 'base_salary', 'teaching_load',
    'research_funding', 'department_size', 'admin_support',
    'work_life_balance', 'promotion_opportunities',
    'publications_last_3_years', 'student_evaluation_avg'
]

categorical_cols = ['academic_rank', 'tenure_status', 'institution_type']

class NumericPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians = {}
        self.clip_vals = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=numeric_cols)
        for col in numeric_cols:
            self.medians[col] = X[col].median()
        for col in numeric_cols:
            low = X[col].quantile(0.01)
            high = X[col].quantile(0.99)
            self.clip_vals[col] = (low, high)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=numeric_cols)
        missing_flags = X.isna().astype(int)
        missing_flags.columns = [f"{col}_missing_flag" for col in missing_flags.columns]

        for col in numeric_cols:
            X[col] = X[col].fillna(self.medians[col])
        for col in numeric_cols:
            low, high = self.clip_vals[col]
            X[col] = X[col].clip(low, high)

        return pd.concat([X, missing_flags], axis=1)

class CategoricalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fill_values = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=categorical_cols)
        for col in categorical_cols:
            self.fill_values[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=categorical_cols)
        missing_flags = X.isna().astype(int)
        missing_flags.columns = [f"{col}_missing_flag" for col in missing_flags.columns]

        for col in categorical_cols:
            X[col] = X[col].fillna(self.fill_values[col])

        return pd.concat([X, missing_flags], axis=1)
