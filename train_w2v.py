from gensim.models import Word2Vec
from datetime import datetime

#30101194 rows
FIN = './unlabeled_statuses/unlabeled_statuses_sorted_preprocessed.txt'
MODEL = './unlabeled_statuses/w2vmodel_iter5.txt'

# def read_sents(f):
#     t = datetime.now()
#     for i,line in enumerate(open(f)):
#         if i%1e6==0:
#             print('%dm  %s'%(i/1e6,datetime.now()-t))
#         # if i==1e6:
#         #     raise StopIteration
#         #     break
        
#         lines = line.split(',')
#         cur_uid = lines[0]
#         sent = ','.join(lines[5:])
#         if i == 0:
#             prev_uid = cur_uid
#             sents = []
            
#         if cur_uid == prev_uid:
#             sents.append(sent)
#         else:
#             sents = ' '.join(sents)
#             #regex
# #            sents = re.sub('\n|http.*|@.*|//','',sents)
#             sents = ' '.join(re.findall(r'[\u4e00-\u9fff]+',sents))
#             sents = re.sub('\s+','',sents)
        
#             seg_list = jieba.cut(sents, cut_all=False)
#             yield list(seg_list)
#             sents = []
#             prev_uid = cur_uid

# class gene2iter(object):
#     def __init__(self,ge):
#         self.ge = ge
#     def __iter__(self):
#         for sent in self.ge:
#             yield sent

# class Sents(object):
#     def __init__(self,f):
#         self.f = f
#     def __iter__(self):
#         t = datetime.now()
#         length=30
#         for i,line in enumerate(open(self.f)):
#             if i%1e3==0 and i!=0:
#                 print('%dk  %s'%(i/1e3,datetime.now()-t))
#             lines = line.split(',')
#             cur_uid = lines[0]
#             sent = lines[1]
#             words = sent.split()
#             size = len(words)
#             n = size//length
#             for i in range(n):
#                 yield words[i*length:(i+1)*length]

class Sents(object):
    def __init__(self,f):
        self.f = f
    def __iter__(self):
        t = datetime.now()
        for i,line in enumerate(open(self.f)):
            if i%1e3==0 and i!=0:
                print('%dk  %s'%(i/1e3,datetime.now()-t))
            lines = line.split(',')
            cur_uid = lines[0]
            sent = lines[1]
            words = sent.split()
            yield words
            del words

sents = Sents(FIN)
model = Word2Vec(sents, size=100, window=5, min_count=5, workers=8,iter=2)
model.save_word2vec_format(MODEL)