
import PyPDF2
from datasurfer import DataInterface


class PDFObject(DataInterface):
    '''
    Demos:

    https://github.com/nikhilkumarsingh/geeksforgeeks/tree/master/PyPDF2_tutorial

    '''    
    def __init__(self, path):
        self.path = path
        self.text = ''
        self.pages = []
        # self.extract_text()

    def extract_text(self):

        pdf_file = open(self.path, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            self.pages.append(page)
            self.text += page.extract_text()
        pdf_file.close()

    def get_text(self):
        return self.text

    def get_pages(self):
        return self.pages

    def get_page(self, page_num):
        return self.pages[page_num]

    def get_num_pages(self):
        return len(self.pages)

    def get_text_from_page(self, page_num):
        return self.pages[page_num].extract_text()

    def get_text_from_pages(self, page_nums):
        text = ''
        for page_num in page_nums:
            text += self.pages[page_num].extract_text()
        return text

    def get_text_from_range(self, start, end):
        text = ''
        for page_num in range(start, end):
            text += self.pages[page_num].extract_text()
        return text

    def get_text_from_all_pages(self):
        return self.get_text_from_range(0, self.get_num_pages())

    def get_text_from_pages(self, page_nums):
        text = ''
        for page_num in page_nums:
            text += self.pages[page_num].extract_text()
        return text

    def get_text_from_range(self, start, end):
        text = ''
        for page_num in range(start, end):
            text += self.pages[page_num].extract_text()
        return text

    def get_text_from_all_pages(self):
        return self.get_text_from_range(0, self.get_num_pages())

    def get_text_from_pages(self, page_nums):
        text = ''
        for page_num in page_nums:
            text += self.pages[page_num].extract_text()
        return text

    def get_text_from_range(self, start, end):
        text = ''
        for page_num in range(start, end):
            text += self.pages[page_num].extract_text()
        return text

    def get_text_from_all_pages(self):
        return self.get_text_from_range(0, self.get_num_pages())

   


