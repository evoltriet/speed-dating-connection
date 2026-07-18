from __future__ import annotations

import numpy as np

from speed_dating_connection.prepare import _balance


def test_balance_is_symmetric_and_bounded() -> None:
    values = _balance(np.array([5.0, 0.0, 4.0]), np.array([10.0, 0.0, 4.0]))

    assert values.tolist() == [0.5, 1.0, 1.0]
