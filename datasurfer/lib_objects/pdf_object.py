
import PyPDF2
import pandas as pd
from datasurfer import DataInterface


class PDFTextObject(DataInterface):
    '''
    Demos:

    https://github.com/nikhilkumarsingh/geeksforgeeks/tree/master/PyPDF2_tutorial

    '''    
    exts = ['.pdf']
    def __init__(self, path, name=None, comment=None, page_range=None):
        super().__init__(path, comment=comment, name=name)
        self.page_range = page_range


    def get_df(self):
        pagetexts = []
        page_range = self.page_range or pdf_reader.numPage
        pdf_file = open(self.path, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(*page_range):
            page = pdf_reader.getPage(page_num)
            page_text = page.extract_text()
            pagetexts.append(page_text)

        pdf_file.close()
        
        df = pd.DataFrame(pagetexts, columns=['text'])
        #df['transtext'] = pd.NA
        return df        


   


