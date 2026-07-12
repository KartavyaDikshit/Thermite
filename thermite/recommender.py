from ._core import ALS as CoreALS
import numpy as np

class ALS:
    def __init__(self, factors: int = 10, iterations: int = 10, regularization: float = 0.1):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self._core = CoreALS(factors=factors, iterations=iterations, regularization=regularization)

    def fit(self, R: np.ndarray):
        """Fit the ALS recommender model.

        Parameters
        ----------
        R : np.ndarray
            A 2D numpy array of shape (n_users, n_items) representing the rating matrix.
            Zeroes indicate missing ratings.
        """
        R_arr = np.ascontiguousarray(R, dtype=np.float64)
        self._core.fit(R_arr)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the rating of a user for an item.

        Parameters
        ----------
        user_id : int
            The user index.
        item_id : int
            The item index.

        Returns
        -------
        float
            The predicted rating.
        """
        return self._core.predict(user_id, item_id)
