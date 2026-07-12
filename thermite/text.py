from thermite._core import CountVectorizer as _CountVectorizer, TfidfVectorizer as _TfidfVectorizer

class CountVectorizer:
    def __init__(self, lowercase: bool = True):
        self._core = _CountVectorizer(lowercase=lowercase)
        self.lowercase = lowercase

    def fit(self, raw_documents):
        self._core.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        return self._core.transform(raw_documents)

    def fit_transform(self, raw_documents):
        return self._core.fit_transform(raw_documents)

    @property
    def vocabulary_(self):
        return self._core.vocabulary


class TfidfVectorizer:
    def __init__(self, lowercase: bool = True):
        self._core = _TfidfVectorizer(lowercase=lowercase)
        self.lowercase = lowercase

    def fit(self, raw_documents):
        self._core.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        return self._core.transform(raw_documents)

    def fit_transform(self, raw_documents):
        return self._core.fit_transform(raw_documents)

    @property
    def vocabulary_(self):
        return self._core.vocabulary

    @property
    def idf_(self):
        return self._core.idf

class Word2Vec:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 5):
        from thermite._core import Word2Vec as _Word2Vec
        self._core = _Word2Vec(vector_size, window, min_count)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count

    def fit(self, sentences):
        self._core.fit(sentences)
        return self

    @property
    def embeddings_(self):
        return self._core.embeddings
