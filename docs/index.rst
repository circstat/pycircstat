.. PyCircStat documentation master file, created by
   sphinx-quickstart on Tue Sep 23 13:08:52 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyCircStat's documentation!
======================================

All functions take arguments in radians (expect for ang2rad).
For a detailed description of arguments and outputs consult the help text of the respective functions.

The implementation follows in most cases 'Biostatistical Analysis' and all referenced equations and tables
are taken from this book, if not otherwise noted. In some cases, the other books were preferred for implementation
was more straightforward for solutions presented there.

If you have suggestions, bugs or feature requests or want to contribute code, please email us.



Contents:
=========

.. toctree::
   :maxdepth: 2

   descriptive.rst
   distributions.rst
   iterators.rst
   decorators.rst

Disclaimer:
===========

All functions in this toolbox were implemented with care and tested on the examples presented in
'Biostatistical Analysis' where possible. Nevertheless, they may contain errors or bugs, which may
affect the outcome of your analysis. We do not take responsibility for any harm coming from using
this toolbox, neither if it is caused by errors in the software nor if it is caused by its improper
application. Please email us any bugs you find.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [Fisher1995] Fisher, Nicholas I. Statistical analysis of circular data. Cambridge University Press, 1995.
.. [Jammalamadaka2001] Jammalamadaka, S. Rao, and Ambar Sengupta. Topics in circular statistics. Vol. 5. World Scientific, 2001.
.. [Zar2009] Zar, Jerrold H. Biostatistical analysis. Pearson Education India, 2009.