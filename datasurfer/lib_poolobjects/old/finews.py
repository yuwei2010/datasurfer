# -*- coding: utf-8 -*-

import io
import requests
from xml.etree.ElementTree import parse

URL_YAHOO = {
        'de' : 'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=DE&lang=de-DE',
        'us' : 'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=us&lang=en-us',
        'uk' : 'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=uk&lang=en-uk',
        
        }


#%%---------------------------------------------------------------------------#
class FiNews(list):
    
    def __init__(self, symbol:str, which:str='de'):
        
        url = URL_YAHOO.get(which).format(symbol=symbol)
        
        respon = requests.get(url)
        
        doc = parse(io.StringIO(respon.text))
        
        items = doc.iterfind('channel/item')
        
        keys = ['description', 'guid', 'link', 'pubDate', 'title']
        
        items = [dict((key, item.findtext(key)) for key in keys) for 
              item in items]
        
        for item in items:
            item['title'] = item['title'].replace('&#39;', "'")
            item['description'] = item['description'].replace('&#39;', "'")
                
        super().__init__(items)
        

#%%---------------------------------------------------------------------------#

if __name__ == '__main__':
    
    pass