from setuptools import setup
import sys
sys.path.insert(0, ".")
from EasyMLR import __version__

setup(
    name='EasyMLR',
    version=__version__,
    author='Greg Pelletier',
    py_modules=['EasyMLR'], 
    install_requires=[
        'numpy','pandas','statsmodels','seaborn',
        'scikit-learn','tabulate','matplotlib',
        'xgboost','lightgbm','mlxtend','optuna'],
)

