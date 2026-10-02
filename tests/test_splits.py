import numpy as np
import pytest

from eegwords import splits
from eegwords.data import CACHE, Trials

pytestmark = pytest.mark.skipif(not (CACHE / "sub-01_imagine_meta.json").exists(),
                                reason="needs data/cache (scripts/run_grid.sh builds it)")


@pytest.fixture(scope="module")
def t():
    return Trials("01", "imagine")


def test_trial_count_and_classes(t):
    # The paper presents 6,681 sentences; the released preprocessed pickles keep 6,243 of them
    # (438 removed in preprocessing). Every kept sentence maps to a category.
    assert len(t) == 6243
    assert t.keep.all()
    assert len(np.unique(t.label[t.keep])) == 39


def test_interleaved_is_upstream_rule(t):
    (f,) = splits.make("interleaved", t)
    assert set(f.test) == set(t.index[t.keep & (t.index % 5 == 1)])


@pytest.mark.parametrize("name", splits.SPLITS)
def test_disjoint_and_complete(t, name):
    for f in splits.make(name, t, seed=0):
        a, b, c = set(f.train), set(f.val), set(f.test)
        assert not (a & b) and not (a & c) and not (b & c)
        assert a | b | c == set(t.index[t.keep])


@pytest.mark.parametrize("name", ["run_holdout", "session_holdout"])
def test_grouped_splits_share_no_run(t, name):
    for f in splits.make(name, t, seed=0):
        test_runs = set(t.run[f.test])
        assert not test_runs & set(t.run[f.train])
        assert not test_runs & set(t.run[f.val])
        assert not set(t.run[f.val]) & set(t.run[f.train])


def test_session_folds_are_whole_sessions(t):
    for f in splits.make("session_holdout", t):
        assert len(np.unique(t.session[f.test])) == 1
