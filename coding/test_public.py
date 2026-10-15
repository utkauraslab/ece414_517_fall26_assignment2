"""Illustrative tests. Expected to fail in the unfinished student starter."""
import numpy as np
import q_learning as q


def test_discretization():
    b = q.create_discretization([-2]*4, [2]*4, 4)
    np.testing.assert_allclose(b, np.tile([-1, 0, 1], (4, 1)))
    assert tuple(q.get_discretized_state([-9, -1, 0, 9], b)) == (0, 1, 2, 3)


def test_table():
    table = q.create_q_table(4)
    assert table.shape == (4, 4, 4, 4, 2)
    assert np.issubdtype(table.dtype, np.floating)
    assert np.all(table == 0)


def test_actions():
    table = np.zeros((2, 2, 2, 2, 2))
    s = (0, 0, 0, 0)
    assert q.epsilon_greedy(table, s, 0) == 0
    table[s + (1,)] = 2
    assert q.epsilon_greedy(table, s, 0) == 1
    np.random.seed(7)
    actions = [q.epsilon_greedy(table, s, 1) for _ in range(1000)]
    assert set(actions) == {0, 1}
    assert .4 < np.mean(actions) < .6


def test_updates():
    s, ns = (0, 0, 0, 0), (1, 1, 1, 1)
    for terminal, expected in [(False, 4.65), (True, 3.7)]:
        table = np.zeros((2, 2, 2, 2, 2))
        table[s + (0,)] = 4
        table[ns] = [5, 10]
        q.q_update(table, s, 0, 1, ns, .1, .95, terminated=terminal)
        assert np.isclose(table[s + (0,)], expected)
        assert table[s + (1,)] == 0
