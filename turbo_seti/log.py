import logbook
import sys

logger_group = logbook.LoggerGroup()
logger_group.level = logbook.DEBUG

logbook.StreamHandler(sys.stdout).push_application()
