"""The logger of Larmor, silent unless the application configures logging."""

from logging import Logger, NullHandler, getLogger

# The logger every module of Larmor writes to.
LarmorLogger: Logger = getLogger("larmor")
LarmorLogger.addHandler(NullHandler())
