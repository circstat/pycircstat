from __future__ import absolute_import
from collections import namedtuple

CI = namedtuple('confidence_interval', ['lower', 'upper'])

from .descriptive import *
from .tests import *
from .utils import *
from . import distributions
from . import data
from . import clustering
from . import event_series