import sys  # noqa: F401

import pytest  # noqa: F401


def xfail_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    if solver_class == "celer_cython":
        pytest.xfail(
            "Celer's install on CI raises: ValueError: numpy.ndarray size "
            "changed, may indicate binary incompatibility. Expected 96 from C "
            "header, got 88 from PyObject")
