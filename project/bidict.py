class BiDict[K, V](dict):
    """Bidirectional dict"""

    inverse: dict[V, K]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse = {value: key for (key, value) in self.items()}

    def __setitem__(self, key: K, value: V):
        super().__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key: K):
        del self.inverse[self[key]]
        super().__delitem__(key)
