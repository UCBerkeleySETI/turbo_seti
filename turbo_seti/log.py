import logbook
import sys

logger_group = logbook.LoggerGroup()
logger_group.level = logbook.INFO

logbook.StreamHandler(sys.stdout).push_application()
