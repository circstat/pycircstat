Regression Models
=================

Example for circular-linear regression

.. code-block:: python

    # generate toy data
    alpha = np.random.rand(200)*np.pi*2
    a0 = np.random.rand()*2*np.pi
    A0 = np.abs(np.random.randn())
    m0 = np.random.randn()*10
    x = m0 + A0*np.cos(alpha - a0)

    # generate regressor
    reg = CircularLinearRegression()

    # train regressor
    reg.train(alpha, x)

    # predict
    x2 = reg(alpha)

    # look at coefficients
    print(reg[:])

.. automodule:: pycircstat.regression
    :members:
