"""
Function to do speed tests easily.
"""
import numpy as np
import pandas as pd
from timeit import default_timer


def speedtest(speed_inputs, speed_input_labels, funcs):
    """
    Runs speed tests, and asserts outputs are all the same. Runs the first test before timing anything to make sure
        numba functions are initialized properly.

    :param speed_inputs: list of tuples of (args, kwargs) where args is the *args and kwargs is the **kwargs
    :param speed_input_labels: names to use for each speed_input test
    :param funcs: the functions to test
    """
    func_times, outputs = [], []

    # Initialize numba
    for func in funcs:
        func(*speed_inputs[0][0], **speed_inputs[0][1])

    # Run timing tests
    for func in funcs:
        times = []
        these_outputs = []

        for args, kwargs in speed_inputs:
            t = default_timer()
            these_outputs.append(func(*args, **kwargs))
            times.append(default_timer() - t)

        func_times.append(times)
        outputs.append(these_outputs)

    for i in range(1, len(outputs)):
        for j in range(len(outputs[0])):
            np.testing.assert_array_almost_equal(outputs[i][j], outputs[i - 1][j])

    print("Timings:\n")
    print(pd.DataFrame(func_times, columns=speed_input_labels, index=[f.__name__ for f in funcs]).to_markdown(floatfmt='0.4f'))
