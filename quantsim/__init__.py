"""quantsim — Monte Carlo market simulator with terminal-native risk analytics."""
from quantsim.simulate import ascii_histogram, max_drawdown, simulate_gbm, summarize

__version__ = "0.1.0"
__all__ = ["simulate_gbm", "summarize", "max_drawdown", "ascii_histogram", "__version__"]
