from ._core import ping
from .preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from .model_selection import train_test_split, GridSearchCV, KFold
from .linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, LinearSVC
from .metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score
)
from .tree import DecisionTreeClassifier, DecisionTreeRegressor
from .cluster import KMeans, DBSCAN
from .decomposition import PCA
from .neighbors import KNeighborsClassifier
from .text import CountVectorizer, TfidfVectorizer
from .impute import IterativeImputer
from .ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from .pipeline import Pipeline, ColumnTransformer
from .naive_bayes import GaussianNB
from .svm import SVC
from .device import validate_device, is_gpu, DEVICE_CPU, DEVICE_GPU, DEVICE_CUDA

from .polars_compat import from_polars, from_polars_X, from_polars_y, make_polars_pipeline

from .deep_learning import to_pytorch, to_jax
from .distributed import get_backend
from .automl import BayesianOptimizer
from .neural_network import MLPClassifier

from .feature_selection import RFE
from .time_series import AutoRegressive
from .survival import SurvivalForest

__version__ = "1.4.0"
__all__ = [
    "ping",
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    "OneHotEncoder",
    "train_test_split",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "LogisticRegression",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_auc_score",
    "mean_squared_error",
    "r2_score",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "KMeans",
    "DBSCAN",
    "PCA",
    "KNeighborsClassifier",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "Pipeline",
    "LinearSVC",
    "ColumnTransformer",
    "GaussianNB",
    "SVC",
    "to_pytorch",
    "to_jax",
    "get_backend",
    "BayesianOptimizer",
    "CountVectorizer",
    "TfidfVectorizer",
    "IterativeImputer",
    "MLPClassifier",
    "RFE",
    "AutoRegressive",
    "SurvivalForest",
]

