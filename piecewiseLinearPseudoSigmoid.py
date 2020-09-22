"""
Piecewise linear 'sigmoid' used for speed when squashing neural inputs in difference eqns
The piecewise linear sigmoid function has minimum -span / 2 and maximum span / 2,
and has gradient = slop at x = 0.
"""
import numpy as np


def piecewiseLinearPseudoSigmoid(x, span, slope):
    y = x * slope
    y = np.maximum(y, -span / 2)
    y = np.minimum(y, span / 2)
    return y



# Plot an example of the piecewise linear sigmoid function
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.arange(-8, 8, 0.5)
# y = piecewiseLinearPseudoSigmoid(x, 4, 1)
# plt.plot(x,y)
# plt.show()





