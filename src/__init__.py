# This can often be left empty for simple projects
# You can optionally use it to define what should be imported when someone does `from src import *`

from . import data
from . import features
from . import models
from . import visualization

# If you want to make certain functions easily accessible, you can import them here
from .models.predict_model import predict_oil_production
from .visualization.visualize import plot_oil_production

# You can also define a version for your package
__version__ = '0.1.0'