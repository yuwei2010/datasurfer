
from datasurfer import DataInterface
from datasurfer.datautils import str2df


class STRING_OBJECT(DataInterface):
    """
    Represents a string object.

    Args:
        path (str): The path to the string object.
        config (dict): Configuration settings for the string object.
        name (str): The name of the string object.
        comment (str): Additional comment for the string object.

    Attributes:
        path (str): The path to the string object.
        config (dict): Configuration settings for the string object.
        name (str): The name of the string object.
        comment (str): Additional comment for the string object.
        df (pandas.DataFrame): The data as a pandas DataFrame.
    """

    def __init__(self, s, name, comment=None, **kwargs):
        """
        Initializes a new instance of the STRING_OBJECT class.

        Args:
            path (str): The path to the string object.
            config (dict): Configuration settings for the string object.
            name (str): The name of the string object.
            comment (str): Additional comment for the string object.
        """
        super().__init__(path=None, config=None, name=name, comment=comment)
        
        self._df = str2df(s, **kwargs)

    def get_df(self):

        raise NotImplementedError('No need to implement this method for a STRING_OBJECT.')