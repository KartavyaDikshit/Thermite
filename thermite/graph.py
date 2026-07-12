import numpy as np
from typing import Dict, List
from . import _core

class Node2Vec:
    """
    Node2Vec graph embedding algorithm.
    """
    def __init__(self, p: float = 1.0, q: float = 1.0, walk_length: int = 80, num_walks: int = 10, embedding_dim: int = 128):
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self._model = _core.Node2Vec(p, q, walk_length, num_walks, embedding_dim)

    def fit(self, adjacency_list: Dict[int, List[int]]):
        self._model.fit(adjacency_list)
        return self

    @property
    def embeddings_(self) -> np.ndarray:
        return self._model.embeddings
