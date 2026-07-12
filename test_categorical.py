import numpy as np
from thermite.tree import DecisionTreeClassifier
from thermite.ensemble import RandomForestClassifier

def test_categorical_tree():
    # A simple dataset where a categorical feature perfectly splits the target
    # Feature 0: Continuous
    # Feature 1: Categorical (0, 1, 2)
    # Target: 1 if Feature 1 is 0 or 2, else 0
    X = np.array([
        [1.2, 0],
        [1.5, 0],
        [2.3, 1],
        [2.1, 1],
        [3.4, 2],
        [3.6, 2],
    ], dtype=np.float64)
    y = np.array([1, 1, 0, 0, 1, 1])

    # If it treats feature 1 as continuous, it would need depth=2 to separate {0, 2} from {1}
    # If treated as categorical, it can separate it at depth 1.
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y, categorical_features=[1])
    
    acc = clf.score(X, y)
    print("Decision Tree Categorical accuracy:", acc)
    assert acc == 1.0, "Decision Tree should perfectly separate this at depth 1 using categorical splits."

def test_categorical_rf():
    X = np.array([
        [1.2, 0],
        [1.5, 0],
        [2.3, 1],
        [2.1, 1],
        [3.4, 2],
        [3.6, 2],
    ], dtype=np.float64)
    y = np.array([1, 1, 0, 0, 1, 1])

    clf = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=42)
    clf.fit(X, y, categorical_features=[1])
    
    acc = clf.score(X, y)
    print("Random Forest Categorical accuracy:", acc)
    assert acc == 1.0, "Random Forest should perfectly separate this at depth 1 using categorical splits."

if __name__ == "__main__":
    test_categorical_tree()
    test_categorical_rf()
    print("All categorical split tests passed!")
