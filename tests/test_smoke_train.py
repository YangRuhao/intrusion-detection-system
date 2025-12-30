from __future__ import annotations

import pandas as pd

from ids.data.split import detect_label_column, make_binary_label


def test_detect_label_column():
    df = pd.DataFrame({"a": [1], "Label": ["BENIGN"]})
    assert detect_label_column(df) == "Label"


def test_make_binary_label():
    y = pd.Series(["BENIGN", "DoS", "normal", "PortScan"])
    b = make_binary_label(y)
    assert b.tolist() == [0, 1, 0, 1]
