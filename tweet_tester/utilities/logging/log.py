from django.conf import settings
from enum import Enum


class LogLevel(Enum):
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'


def log(o, t=None):
    if t == None:
        t = LogLevel.INFO
    if settings.DEBUG == True:
        print('[' + t.value + ']:', o)
