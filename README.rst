Benchmark repository for Multi-Task Lasso
=========================================

|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
The Multi-Task Lasso consists in solving the following program:

$$ \\min_W \\, \\tfrac{1}{2} \\Vert Y - XW \\Vert^2_\\mathrm{F} + \\lambda \\Vert W\\Vert_{2, 1} $$

where $n$ (or n_samples) stands for the number of samples, $p$ (or n_features) stands for the number of features and
$T$ (or n_tasks) stands for the number of tasks.

$$ Y \\in \\mathbb{R}^{n \\times T}, \\, \\, X \\in \\mathbb{R}^{n \\times p} $$


Install
-------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/PABannier/benchmark_multi_task_lasso
   $ cd benchmark_multi_task_lasso/

To demonstrate the use of benchopt, one can run, from the `benchmark_multi_task_lasso` folder:

.. code-block::

   $ benchopt install . -s sklearn -s python-pgd --env
   $ benchopt run . --config example_config.yml --env

Alternatively, one can use the command line interface to select which problems, datasets and solvers are used:

.. code-block::

	$ benchopt run -s sklearn -d leukemia --max-runs 10 --n-repetitions 5


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

Troubleshooting
---------------

If you run into some errors when running the examples present in this Readme, try installing the development version of `benchopt`:

.. code-block::

  $ pip install -U git+https://github.com/benchopt/benchopt

If issues persist, you can also try running the benchmark in local mode with the `-l` option, e.g.:

.. code-block::

  $ benchopt run . -l -s sklearn -d leukemia --max-runs 10 --n-repetitions 10

Note that in this case, only solvers which dependencies are installed in the current env will be run.

.. |Build Status| image:: https://github.com/PABannier/benchmark_multi_task_lasso/workflows/Tests/badge.svg
   :target: https://github.com/PABannier/benchmark_multi_task_lasso/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
