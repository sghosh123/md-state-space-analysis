import numpy as np
import pytest
from mdstate.kinetics import transition_matrix, transition_implied_timescales


def test_transition_matrix_respects_trajectory_boundary():
    # If concatenated naively, the final state in traj1 would transition to first state in traj2.
    T, states = transition_matrix([np.array([0, 0, 1]), np.array([1, 1, 0])], lag_time=1)
    assert states.tolist() == [0, 1]
    assert T.shape == (2, 2)
    assert np.all(T.sum(axis=1) <= 1.000001)


def test_transition_its_has_nonstationary_modes():
    T = np.array([[0.9, 0.1], [0.2, 0.8]])
    d = transition_implied_timescales(T, lag_time=3)
    assert len(d) == 1
    assert d["implied_timescale_frames"].iloc[0] > 0


def test_transition_bad_lag():
    with pytest.raises(ValueError):
        transition_matrix([np.array([0, 1, 0])], lag_time=0)


def test_transition_timescales_reject_zero_rows():
    with pytest.raises(ValueError, match="sum to one"):
        transition_implied_timescales(np.array([[1.0, 0.0], [0.0, 0.0]]), lag_time=2)
