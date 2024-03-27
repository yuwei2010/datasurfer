
from datasurfer.datautils import arghisto, parse_data

#%%

class Stats(object):
    """
    A class for performing statistical computations.

    Attributes:
    dp: The data object.

    """

    def __init__(self, dp) -> None:
        self.dp = dp

    @parse_data
    def arghisto(self, val, *, bins, **kwargs):
        """
        Compute the histogram of a given value.

        Parameters:
        val (array-like): The input values.
        bins (int): The number of bins to use for the histogram.

        Returns:
        numpy.ndarray: The computed histogram.

        """
        return arghisto(val, bins)
    
    
        
 
        
    
# %%
