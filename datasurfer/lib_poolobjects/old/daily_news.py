# -*- coding: utf-8 -*-
import sys
import io

sys.path.insert(0, r'D:\01_Python')
import os


from shares import DAX, MDAX, SDAX, GREEN, GOLD


from datetime import datetime



from finews import FiNews






redo = False

root = os.path.join(r'Analyse_Data', datetime.today().strftime('%Y%m%d') )

if not os.path.lexists(root):
    os.mkdir(root)



grps = [
        (0, GOLD, 'GOLD'),
        (1, DAX, 'DAX'), 
        (2, MDAX, 'MDAX'),
        (3, SDAX, 'SDAX'),
        (4, GREEN, 'GREEN'),
        ]

for idx, grp, gname in grps:
    
    subroot = os.path.join(root, gname)
    
    if not os.path.lexists(subroot):
        
        os.mkdir(subroot)
    
    for key, name in grp.items():
        
        print(idx, key)
        
        
        newsfile = os.path.join(subroot, key)

        if redo or not os.path.lexists(newsfile):
            with io.open(os.path.join(subroot, key), 'w', encoding='utf-8') as fobj:
                
                for which in ['de', 'us', 'uk']:
                    newss = FiNews(key, which)
                    for news in newss:
                        
                        title = news['title']
    #                    descrip = news['description'].replace('\n', ' ')
                        pdate = news['pubDate']
                        link = news['link']
                        
                        fobj.write(pdate+'\n')
                        try:
#                            for en, cn in tr.translate(title, 'de', 'zh-cn', split=True):
                            
#                                cn = bd.translate(title, dst='zh')
                                
                                fobj.write(title+'\n')
                                
                                
#                                fobj.write(cn+'\n')
#                                time.sleep(1)
                                
                        except Exception as err:
                            
                            print(title, err)
                            fobj.write(title+'\n')
                            
                        
                        fobj.write(link+'\n')
        
                            
                        fobj.write('\n')
                    fobj.write(('-'*60)+'\n')
        


