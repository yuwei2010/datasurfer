"""Top-level package for Data Structure."""

__author__ = """Wei Yu (PS-EM/ESY2)"""
__email__ = 'wei.yu2@de.bosch.com'



from .datapool import DataPool
from .datalake import DataLake

__all__   = ['DataPool', 'DataLake']
__slots__ = ['DataPool', 'DataLake']