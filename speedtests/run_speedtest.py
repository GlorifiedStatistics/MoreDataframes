"""
File to actually run speed tests
"""
from speedtest_binning import speedtest_chi_square_all
from speedtest_minmax import speedtest_min_max


if __name__ == '__main__':
    speedtest_min_max()
