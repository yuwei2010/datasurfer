import numpy as np
from functools import wraps
from datasurfer.datautils import parse_data

#%%
def output_switcher(func):   
    """
    Decorator function that allows switching the output of a function.
    
    Args:
        func (function): The function to be decorated.
        
    Returns:
        function: The decorated function.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        
        return_model = kwargs.pop('return_model', False)
        
        mdl, foo = func(self, *args, **kwargs)
        
        if return_model:
            return mdl
        else:
            return foo()
        
    return wrapper
        
        

#%%
class MLearn(object):
    """
    The MLearn class provides functionality for machine learning tasks.

    Parameters:
    dp: object, optional
        The data preprocessing object.

    Attributes:
    dp: object
        The data preprocessing object.

    Methods:
    display_decision: Display the decision boundary of a machine learning model.
    detect_outliers: Detect outliers in the given data.
    """

    def __init__(self, dp=None):
        
        self.dp = dp

    @parse_data    
    def display_decision(self, *vals, model, **kwargs):
        """
        Display the decision boundary of a machine learning model.

        Parameters:
        *vals: tuple
            Input values for the decision boundary display. Only two inputs are supported.
        model: object
            The trained machine learning model.
        **kwargs: dict
            Additional keyword arguments to be passed to the DecisionBoundaryDisplay class.

        Returns:
        ax: matplotlib.axes.Axes
            The matplotlib axes object containing the decision boundary plot.
        """
        assert len(vals) == 2, "Only two inputs are supported for decision boundary display."
        from sklearn.inspection import DecisionBoundaryDisplay

        lbls = kwargs.pop('labels', None)
        disp = DecisionBoundaryDisplay.from_estimator(model, np.vstack(vals).T, **kwargs)

        ax = disp.ax_
        ax.set_xlabel(lbls[0])
        ax.set_ylabel(lbls[1])

        return ax

    @output_switcher 
    @parse_data    
    def detect_outliers(self, *vals, **kwargs):
        """
        Detect outliers in the given data.

        Parameters:
        *vals : array-like
            Input data values.

        **kwargs : dict
            Additional keyword arguments to be passed to the IsolationForest model.

        Returns:
        bool
            An array of boolean values indicating whether each data point is an outlier or not.
        """

        from sklearn.ensemble import IsolationForest
        kwargs.pop('labels', None)
        
        X = np.vstack(vals).T
        clf = IsolationForest(**kwargs)
        
        clf.fit(X)
        fstar = lambda :clf.predict(X) == -1

        return clf, fstar
    
    
    
        
    



