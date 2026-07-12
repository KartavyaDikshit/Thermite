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
