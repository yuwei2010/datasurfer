
import json
import pandas as pd
from datasurfer.datainterface import DataInterface, translate_config


class JSON_OBJECT(DataInterface):
    """
    Represents a JSON object.

    Args:
        path (str): The path to the JSON file.
        config (dict): Configuration options for the JSON object.
        name (str): The name of the JSON object.
        comment (str): Additional comments about the JSON object.

    Attributes:
        path (str): The path to the JSON file.
        config (dict): Configuration options for the JSON object.
        name (str): The name of the JSON object.
        comment (str): Additional comments about the JSON object.
    """

    def __init__(self, path=None, config=None, name=None, comment=None):
        super().__init__(path, config=config, name=name, comment=comment)
        
    @translate_config()
    def get_df(self):
        """
        Returns a pandas DataFrame representation of the JSON object.

        Returns:
            pandas.DataFrame: The DataFrame representation of the JSON object.
        """
        dat = json.load(open(self.path, 'r'))
        df = pd.DataFrame(dat).transpose()
        return df
        
if __name__ == '__main__':
    
    pass