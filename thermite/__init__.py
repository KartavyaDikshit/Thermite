from ._core import ping
from .preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from .model_selection import train_test_split, GridSearchCV, KFold, SuccessiveHalvingSearchCV, StratifiedKFold, TimeSeriesSplit, GroupKFold
from .linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, LinearSVC
from .metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score,
    log_loss, mean_absolute_percentage_error, pairwise_distances
)
from .tree import DecisionTreeClassifier, DecisionTreeRegressor
from .cluster import KMeans, DBSCAN, SpectralClustering
from .manifold import TSNE, UMAP
from .decomposition import PCA
from .neighbors import KNeighborsClassifier, LocalOutlierFactor
from .text import CountVectorizer, TfidfVectorizer, Word2Vec
from .impute import IterativeImputer
from .ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
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
from .multi_output import MultiOutputRegressor
from .graph import Node2Vec
from .recommender import ALS
from .quantum import QSVC
from .causal import TLearner

__version__ = "1.8.0"
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
    "log_loss",
    "mean_absolute_percentage_error",
    "pairwise_distances",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "KMeans",
    "DBSCAN",
    "SpectralClustering",
    "TSNE",
    "UMAP",
    "PCA",
    "KNeighborsClassifier",
    "LocalOutlierFactor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "IsolationForest",
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
    "MultiOutputRegressor",
    "Node2Vec",
    "Word2Vec",
    "SuccessiveHalvingSearchCV",
    "StratifiedKFold",
    "TimeSeriesSplit",
    "GroupKFold",
    "ALS",
    "QSVC",
    "TLearner",
]

def __getattr__(name):
    import warnings
    import importlib
    try:
        sklearn_module = importlib.import_module("sklearn")
        if hasattr(sklearn_module, name):
            attr = getattr(sklearn_module, name)
            warnings.warn(f"'{name}' not found in thermite. Falling back to sklearn.")
            return attr
        submodule = importlib.import_module(f"sklearn.{name}")
        warnings.warn(f"'{name}' not found in thermite. Falling back to sklearn.{name}.")
        return submodule
    except ImportError:
        raise AttributeError(f"module 'thermite' has no attribute '{name}'")

