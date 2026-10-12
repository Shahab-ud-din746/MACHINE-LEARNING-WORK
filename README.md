# ==============================================================================
#  PROJECT    : Machine Learning Practical Work
#  PRACTICALS : 01 - Find-S Algorithm | 02 - ID3, C4.5 and CART Comparison
#  Student    : Shahabuddin
#  Subject    : Machine Learning
#  University : Hazara University
# ==============================================================================

# ------------------------------------------------------------------------------
#  SECTION 0 - Imports & Helper Functions
# ------------------------------------------------------------------------------
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

WIDTH = 72


def banner(title: str) -> None:
    """Print a centered banner."""
    print("\n" + "=" * WIDTH)
    print(title.center(WIDTH))
    print("=" * WIDTH)


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print rows as a neatly aligned text table."""
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(line)
    print(fmt.format(*headers))
    print(line)
    for r in rows:
        print(fmt.format(*r))
    print(line)


# ##############################################################################
#  PRACTICAL 01 : FIND-S ALGORITHM
# ##############################################################################
banner("PRACTICAL 01 | FIND-S ALGORITHM")

# ---- Step 1: Training dataset (EnjoySport) -----------------------------------
ATTRIBUTES: List[str] = ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast"]
TARGET: str = "EnjoySport"

TRAINING_DATA: List[List[str]] = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same",   "Yes"],
    ["Sunny", "Warm", "High",   "Strong", "Warm", "Same",   "Yes"],
    ["Rainy", "Cold", "High",   "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High",   "Strong", "Cool", "Change", "Yes"],
]

print("\nTRAINING DATASET")
print_table(["#"] + ATTRIBUTES + [TARGET],
            [[str(i + 1)] + row for i, row in enumerate(TRAINING_DATA)])
print(f"Total examples : {len(TRAINING_DATA)}")
print(f"Positive       : {sum(r[-1] == 'Yes' for r in TRAINING_DATA)}")
print(f"Negative       : {sum(r[-1] == 'No' for r in TRAINING_DATA)}")


# ---- Step 2: Find-S implementation -------------------------------------------
def find_s(data: List[List[str]], verbose: bool = True) -> Optional[List[str]]:
    """
    Find the most specific hypothesis consistent with all positive examples.

    Parameters
    ----------
    data    : rows whose last element is 'Yes' or 'No'.
    verbose : print the hypothesis after every example.

    Returns
    -------
    The final hypothesis as a list, or None if there is no positive example.
    """
    hypothesis: Optional[List[str]] = None

    for idx, row in enumerate(data, start=1):
        features, label = row[:-1], row[-1]

        if label != "Yes":                       # negative examples are ignored
            if verbose:
                print(f"Example {idx}: negative -> skipped")
            continue

        if hypothesis is None:                   # first positive example
            hypothesis = features.copy()
        else:                                    # generalize mismatched attributes
            hypothesis = [h if h == f else "?" for h, f in zip(hypothesis, features)]

        if verbose:
            print(f"Example {idx}: positive -> {hypothesis}")

    return hypothesis


def predict_find_s(hypothesis: List[str], instance: List[str]) -> str:
    """Classify a new instance using the learned hypothesis."""
    ok = all(h == "?" or h == x for h, x in zip(hypothesis, instance))
    return "Yes" if ok else "No"


# ---- Step 3: Run Find-S ------------------------------------------------------
print("\nFIND-S EXECUTION TRACE")
final_hypothesis = find_s(TRAINING_DATA)

print("-" * WIDTH)
if final_hypothesis is None:
    print("No positive example found in the dataset.")
else:
    print("FINAL HYPOTHESIS (most specific):")
    print_table(ATTRIBUTES, [final_hypothesis])

    # ---- Step 4: Test on new instances ---------------------------------------
    TEST_INSTANCES = [
        ["Sunny", "Warm", "High", "Strong", "Cool", "Same"],
        ["Rainy", "Warm", "High", "Strong", "Cool", "Same"],
    ]
    print("\nPREDICTIONS ON NEW INSTANCES")
    for inst in TEST_INSTANCES:
        print(f"{str(inst):<62} -> {predict_find_s(final_hypothesis, inst)}")


# ##############################################################################
#  PRACTICAL 02 : ID3, C4.5 AND CART DECISION TREES
# ##############################################################################
banner("PRACTICAL 02 | ID3 vs C4.5 vs CART")

# ---- Step 1: Dataset (PlayTennis) --------------------------------------------
df = pd.DataFrame([
    ["Sunny",    "Hot",  "High",   "Weak",   "No"],
    ["Sunny",    "Hot",  "High",   "Strong", "No"],
    ["Overcast", "Hot",  "High",   "Weak",   "Yes"],
    ["Rain",     "Mild", "High",   "Weak",   "Yes"],
    ["Rain",     "Cool", "Normal", "Weak",   "Yes"],
    ["Rain",     "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"],
    ["Sunny",    "Mild", "High",   "Weak",   "No"],
    ["Sunny",    "Cool", "Normal", "Weak",   "Yes"],
    ["Rain",     "Mild", "Normal", "Weak",   "Yes"],
    ["Sunny",    "Mild", "Normal", "Strong", "Yes"],
    ["Overcast", "Mild", "High",   "Strong", "Yes"],
    ["Overcast", "Hot",  "Normal", "Weak",   "Yes"],
    ["Rain",     "Mild", "High",   "Strong", "No"],
], columns=["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"])

FEATURES: List[str] = list(df.columns[:-1])
ROWS: List[Dict[str, str]] = df[FEATURES].to_dict("records")
LABELS: List[str] = df["PlayTennis"].tolist()

print("\nDATASET PREVIEW")
print(df.to_string(index=False))
print(f"\nShape: {df.shape} | Class distribution: {dict(Counter(LABELS))}")


# ---- Step 2: Splitting criteria ----------------------------------------------
def entropy(y: List[str]) -> float:
    """Shannon entropy: -sum(p * log2(p))."""
    n = len(y)
    return -sum((c / n) * math.log2(c / n) for c in Counter(y).values())


def gini(y: List[str]) -> float:
    """Gini impurity: 1 - sum(p^2)."""
    n = len(y)
    return 1 - sum((c / n) ** 2 for c in Counter(y).values())


def _group(rows: List[dict], y: List[str], f: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for r, label in zip(rows, y):
        groups.setdefault(r[f], []).append(label)
    return groups


def info_gain(rows, y, f) -> float:
    """ID3 criterion: Information Gain."""
    n = len(y)
    remainder = sum(len(g) / n * entropy(g) for g in _group(rows, y, f).values())
    return entropy(y) - remainder


def split_info(rows, y, f) -> float:
    n = len(y)
    return -sum(len(g) / n * math.log2(len(g) / n) for g in _group(rows, y, f).values())


def gain_ratio(rows, y, f) -> float:
    """C4.5 criterion: Gain Ratio = Gain / SplitInfo."""
    si = split_info(rows, y, f)
    return info_gain(rows, y, f) / si if si > 0 else 0.0


def best_gini_split(rows, y, f) -> Optional[Tuple[float, str]]:
    """CART criterion: best binary split (feature == value vs. rest)."""
    n, best = len(y), None
    for v in {r[f] for r in rows}:
        left = [l for r, l in zip(rows, y) if r[f] == v]
        right = [l for r, l in zip(rows, y) if r[f] != v]
        if not left or not right:
            continue
        score = len(left) / n * gini(left) + len(right) / n * gini(right)
        if best is None or score < best[0]:
            best = (score, v)
    return best


print("\nSPLIT SCORES AT THE ROOT NODE")
scores = pd.DataFrame({
    "Info Gain (ID3)": [info_gain(ROWS, LABELS, f) for f in FEATURES],
    "Gain Ratio (C4.5)": [gain_ratio(ROWS, LABELS, f) for f in FEATURES],
    "Weighted Gini (CART)": [best_gini_split(ROWS, LABELS, f)[0] for f in FEATURES],
}, index=FEATURES).round(4)
print(scores)


# ---- Step 3: Decision tree implementation ------------------------------------
@dataclass
class Node:
    """A node of the decision tree (leaf if `label` is set)."""
    label: Optional[str] = None
    feature: Optional[str] = None
    value: Optional[str] = None
    majority: Optional[str] = None
    children: Dict[str, "Node"] = field(default_factory=dict)


class DecisionTree:
    """Decision tree supporting criterion = 'id3', 'c45' or 'cart'."""

    def __init__(self, criterion: str = "id3") -> None:
        if criterion not in {"id3", "c45", "cart"}:
            raise ValueError("criterion must be 'id3', 'c45' or 'cart'")
        self.criterion = criterion
        self.root: Optional[Node] = None

    # ---------------------------- training ----------------------------
    def fit(self, rows, y, features) -> "DecisionTree":
        self.root = self._build(rows, y, features)
        return self

    def _build(self, rows, y, features) -> Node:
        counts = Counter(y)
        majority = counts.most_common(1)[0][0]
        if len(counts) == 1 or not features:
            return Node(label=majority)

        if self.criterion == "cart":
            return self._build_cart(rows, y, features, majority)

        if self.criterion == "id3":
            best = max(features, key=lambda a: info_gain(rows, y, a))
        else:  # C4.5: best gain ratio among attributes with >= average gain
            gains = {a: info_gain(rows, y, a) for a in features}
            avg = sum(gains.values()) / len(gains)
            candidates = [a for a in features if gains[a] >= avg - 1e-12]
            best = max(candidates, key=lambda a: gain_ratio(rows, y, a))

        node = Node(feature=best, majority=majority)
        remaining = [a for a in features if a != best]
        for v in {r[best] for r in rows}:
            idx = [i for i, r in enumerate(rows) if r[best] == v]
            node.children[v] = self._build([rows[i] for i in idx],
                                           [y[i] for i in idx], remaining)
        return node

    def _build_cart(self, rows, y, features, majority) -> Node:
        best = None
        for f in features:
            s = best_gini_split(rows, y, f)
            if s and (best is None or s[0] < best[0]):
                best = (s[0], f, s[1])
        if best is None:
            return Node(label=majority)

        _, f, v = best
        node = Node(feature=f, value=v, majority=majority)
        yes = [i for i, r in enumerate(rows) if r[f] == v]
        no = [i for i, r in enumerate(rows) if r[f] != v]
        for key, idx in (("yes", yes), ("no", no)):
            node.children[key] = self._build([rows[i] for i in idx],
                                             [y[i] for i in idx], features)
        return node

    # --------------------------- prediction ---------------------------
    def predict_one(self, row: dict) -> str:
        node = self.root
        while node.label is None:
            if self.criterion == "cart":
                key = "yes" if row[node.feature] == node.value else "no"
            else:
                key = row[node.feature]
            if key not in node.children:
                return node.majority
            node = node.children[key]
        return node.label

    def predict(self, rows: List[dict]) -> List[str]:
        return [self.predict_one(r) for r in rows]

    # ------------------------- tree statistics -------------------------
    def depth(self, node: Optional[Node] = None) -> int:
        node = node or self.root
        if node.label is not None:
            return 0
        return 1 + max(self.depth(c) for c in node.children.values())

    def count(self, node: Optional[Node] = None) -> Tuple[int, int]:
        """Return (total nodes, leaves)."""
        node = node or self.root
        if node.label is not None:
            return 1, 1
        nodes, leaves = 1, 0
        for c in node.children.values():
            cn, cl = self.count(c)
            nodes, leaves = nodes + cn, leaves + cl
        return nodes, leaves

    def show(self, node: Optional[Node] = None, indent: str = "") -> None:
        node = node or self.root
        if node.label is not None:
            print(f"{indent}-> {node.label}")
            return
        for key, child in node.children.items():
            if self.criterion == "cart":
                op = "==" if key == "yes" else "!="
                cond = f"{node.feature} {op} {node.value}"
            else:
                cond = f"{node.feature} = {key}"
            print(f"{indent}[{cond}]")
            self.show(child, indent + "    ")


# ---- Step 4: Train models and display trees ----------------------------------
MODELS: Dict[str, DecisionTree] = {
    "ID3": DecisionTree("id3"),
    "C4.5": DecisionTree("c45"),
    "CART": DecisionTree("cart"),
}

for name, model in MODELS.items():
    model.fit(ROWS, LABELS, FEATURES)
    banner(f"{name} TREE")
    model.show()


# ---- Step 5: Performance comparison ------------------------------------------
def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def loocv_accuracy(criterion: str) -> float:
    """Leave-One-Out Cross-Validation accuracy."""
    correct = 0
    for i in range(len(ROWS)):
        tree = DecisionTree(criterion).fit(ROWS[:i] + ROWS[i + 1:],
                                           LABELS[:i] + LABELS[i + 1:], FEATURES)
        correct += tree.predict_one(ROWS[i]) == LABELS[i]
    return correct / len(ROWS)


def fit_time_ms(criterion: str, repeats: int = 200) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        DecisionTree(criterion).fit(ROWS, LABELS, FEATURES)
    return (time.perf_counter() - start) / repeats * 1000


results = []
for name, model in MODELS.items():
    nodes, leaves = model.count()
    results.append({
        "Algorithm": name,
        "Train Accuracy": round(accuracy(LABELS, model.predict(ROWS)), 3),
        "LOOCV Accuracy": round(loocv_accuracy(model.criterion), 3),
        "Depth": model.depth(),
        "Nodes": nodes,
        "Leaves": leaves,
        "Fit Time (ms)": round(fit_time_ms(model.criterion), 3),
    })

comparison = pd.DataFrame(results).set_index("Algorithm")
banner("COMPARISON: ID3 vs C4.5 vs CART")
print(comparison)


# ---- Step 6: Visual comparison (works with or without matplotlib) ------------
METRICS = ["LOOCV Accuracy", "Depth", "Nodes", "Fit Time (ms)"]

try:
    import matplotlib.pyplot as plt

    colors = ["#00d4ff", "#7b61ff", "#ff7a59"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, col in zip(axes, METRICS):
        bars = ax.bar(comparison.index, comparison[col], color=colors)
        ax.set_title(col, fontweight="bold")
        ax.bar_label(bars, padding=2)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("ID3 vs C4.5 vs CART", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()

except ImportError:
    print("\nmatplotlib not installed, showing text charts instead.\n")
    for col in METRICS:
        print(col)
        top = comparison[col].max() or 1
        for name, val in comparison[col].items():
            print(f"  {name:<5} | {'█' * int(val / top * 30)} {val}")
        print()

banner("ALL PRACTICALS COMPLETED SUCCESSFULLY")
