"""ECE 414/517 starter, adapted from CMU 15-780 Spring 2025 HW5.

Complete the five TODO functions. No neural networks are needed.
"""
import numpy as np


def create_discretization(min_state, max_state, num_discrete_states):
    """Return shape (4, n-1) internal thresholds defining n bins per dimension.

    Use linspace(low, high, n+1)[1:-1] independently for each dimension.
    """
    # TODO: Implement this!
    raise NotImplementedError


def get_discretized_state(state, buckets):
    """Return four integer bin indices, using right-bin equality.

    With n-1 thresholds, valid indices are 0,...,n-1, including out-of-range
    observations. np.digitize(..., right=False) implements this convention.
    """
    # TODO: Implement this!
    raise NotImplementedError


def create_q_table(num_discrete_states):
    """Return a zero-filled float array of shape (n,n,n,n,2)."""
    # TODO: Implement this!
    raise NotImplementedError


def epsilon_greedy(q_table, discrete_state, epsilon):
    """Return 0 or 1. Explore uniformly; greedy ties choose action 0.

    Use np.random so the driver's seed controls your sampling.
    """
    # TODO: Implement this!
    raise NotImplementedError


def q_update(q_table, discrete_state, action, reward, discrete_next_state,
             learning_rate, discount, terminated=False):
    """Update Q in place. Target is reward on genuine termination.

    Otherwise bootstrap from max Q(next_state). Time-limit truncation alone
    is not genuine termination under this assignment's task convention.
    """
    # TODO: Implement this!
    raise NotImplementedError
