import pandas as pd
import numpy as np
import re
import xgboost as xgb
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

df_tr = pd.read_csv('./train/train_labels.txt',sep=u'|',header=None).dropna(1)
df_tr.columns = ['uid','sex','age','loc']
df_te = pd.read_csv('./test/test_nolabels.txt',sep=u'|',header=None).dropna(1)
df_te.columns = ['uid']
df_all = pd.concat([df_tr,df_te],axis=0)

df_tr_info = pd.read_csv('./train/train_info.txt',sep=u'|',header=None).dropna(1)
df_tr_info.columns = ['uid','name','image']
df_tr_info = df_tr_info.drop_duplicates()
df_te_info = pd.read_csv('./test/test_info.txt',sep=u'|',header=None).dropna(1)
df_te_info.columns = ['uid','name','image']
df_te_info = df_te_info.drop_duplicates()
df_info = pd.concat([df_tr_info,df_te_info],axis=0)

links = []
for i, line in enumerate(open('./train/train_links.txt')):
    line = line.split()
    row = {'uid':int(line[0]),'fans_cnt':len(line)-1,'fans':' '.join(line[1:])}
    links.append(row)
df_tr_links = pd.DataFrame(links)
df_tr_links = df_tr_links.drop_duplicates()

links = []
for i, line in enumerate(open('./test/test_links.txt')):
    line = line.split()
    row = {'uid':int(line[0]),'fans_cnt':len(line)-1,'fans':' '.join(line[1:])}
    links.append(row)
df_te_links = pd.DataFrame(links)
df_te_links = df_te_links.drop_duplicates()

df_links = pd.concat([df_tr_links,df_te_links],axis=0)

status = []
for i, line in enumerate(open('./train/train_status.txt')):
    
    l = re.search(',',line).span()[0]
    r = re.search(',',line).span()[1]
    row = {'uid':int(line[:l]),'sta':line[r:]}
    status.append(row)
df_tr_status = pd.DataFrame(status)

status = []
for i, line in enumerate(open('./test/test_status.txt')):
    
    l = re.search(',',line).span()[0]
    r = re.search(',',line).span()[1]
    row = {'uid':int(line[:l]),'sta':line[r:]}
    status.append(row)
df_te_status = pd.DataFrame(status)

df_status = pd.concat([df_tr_status,df_te_status],axis=0)

df_mge = pd.merge(df_all,df_info,on='uid',how='left')
df_mge = pd.merge(df_mge,df_links,on='uid',how='left')
df_mge.index = range(len(df_mge))
##################################################################################

df_status['ret'] = df_status.sta.map(lambda s:int(s.split(',')[0]))
df_status['rev'] = df_status.sta.map(lambda s:int(s.split(',')[1]))
df_status['src'] = df_status.sta.map(lambda s:s.split(',')[2])
df_status['time'] = df_status.sta.map(lambda s:s.split(',')[3])
df_status['content'] = df_status.sta.map(lambda s:','.join(s.split(',')[4:]))
bag_twts = df_status.groupby('uid')['content'].agg(lambda lst:' '.join(lst))
df_mge['bag_twts'] = df_mge.uid.map(bag_twts)
df_mge['twts_cnt'] = df_mge.uid.map(df_status.groupby('uid').size())
df_mge['twts_ret_mean'] = df_mge.uid.map(df_status.groupby('uid')['ret'].agg('mean'))
df_mge['twts_rev_mean'] = df_mge.uid.map(df_status.groupby('uid')['rev'].agg('mean'))

d = {'上海': '华东',
 '云南': '西南',
 '内蒙古': '华北',
 '北京': '华北',
 '台湾': '华东',
 '吉林': '东北',
 '四川': '西南',
 '天津': '华北',
 '宁夏': '西北',
 '安徽': '华东',
 '山东': '华东',
 '山西': '华北',
 '广东': '华南',
 '广西': '华南',
 '新疆': '西北',
 '江苏': '华东',
 '江西': '华东',
 '河北': '华北',
 '河南': '华中',
 '浙江': '华东',
 '海南': '华南',
 '湖北': '华中',
 '湖南': '华中',
 '澳门': '华南',
 '甘肃': '西北',
 '福建': '华东',
 '西藏': '西南',
 '贵州': '西南',
 '辽宁': '东北',
 '重庆': '西南',
 '陕西': '西北',
 '青海': '西北',
 '香港': '华南',
 '黑龙江': '东北'}

def bin_loc(s):
    if pd.isnull(s):
        return s
    s = s.split(' ')[0]
    if s == 'None':
        return '华北'
    if s == '海外':
        return s
    return d[s]

def bin_age(age):
    if pd.isnull(age):
        return age
    if age <=1979:
        return "-1979"
    elif age<=1989:
        return "1980-1989"
    else:
        return "1990+"

df_mge['bin_loc'] = df_mge['loc'].map(bin_loc)
df_mge['bin_age'] = df_mge['age'].map(bin_age)

src_lst = df_status.groupby('uid')['src'].agg(lambda lst:' '.join(lst))
df_mge['src_content'] = df_mge['uid'].map(src_lst) 

keys = '|'.join(d.keys())
df_mge['src_prov'] = df_mge['src_content'].map(lambda s:' '.join(re.findall(keys,s)))
df_mge['cont_prov'] = df_mge['bag_twts'].map(lambda s:' '.join(re.findall(keys,s)))

d = defaultdict(lambda :'空',d)
tokenizer = lambda line: [d[w] for w in line.split(' ')]
tfv = TfidfVectorizer(tokenizer=tokenizer,norm=False, use_idf=False, smooth_idf=False, sublinear_tf=False)
X_all_sp = tfv.fit_transform(df_mge['cont_prov'])
prov_cnt = X_all_sp.toarray()
for i in range(prov_cnt.shape[1]):
    df_mge['prov_cnt_%d'%i] = prov_cnt[:,i]

twts_len = df_status.groupby('uid')['content'].agg(lambda lst:np.mean([len(s.split(' ')) for s in lst]))
df_mge['twts_len_0'] = df_mge['uid'].map(twts_len)
twts_len = df_status.groupby('uid')['content'].agg(lambda lst:np.min([len(s.split(' ')) for s in lst]))
df_mge['twts_len_1'] = df_mge['uid'].map(twts_len)
twts_len = df_status.groupby('uid')['content'].agg(lambda lst:np.max([len(s.split(' ')) for s in lst]))
df_mge['twts_len_2'] = df_mge['uid'].map(twts_len)

df_mge['name_len_1'] = df_mge.name.map(lambda s:s if pd.isnull(s) else len(re.sub(r'[\u4e00-\u9fff]+','',s)))
df_mge['name_len_2'] = df_mge.name.map(lambda s:s if pd.isnull(s) else len(s))

df_stack = pd.read_csv('./data/stack.csv')
df_mge = pd.concat([df_mge,df_stack],axis=1)

#########################################################################################
cols = '|'.join(['twts_len','name_len','prov_cnt','fans_cnt',
                'age_','sex_','loc_',
               'twts_ret_mean','twts_cnt','twts_rev_mean'])
cols = [c for c in df_mge.columns if re.match(cols,c)]

age_le = LabelEncoder()
ys = {}
ys['age'] = age_le.fit_transform(df_mge.iloc[:3200]['bin_age'])

loc_le = LabelEncoder()
ys['loc'] = loc_le.fit_transform(df_mge.iloc[:3200]['bin_loc'])

sex_le = LabelEncoder()
ys['sex'] = sex_le.fit_transform(df_mge.iloc[:3200]['sex'])


task = ['tr']


TR = 3200
TE = 980
X_all = df_mge[cols]
X = X_all[:TR]
prds = []
#################################################
label = 'age'
print('='*20)
print(label)
print('='*20)
y = ys[label]

ss = 0.4
mc = 2.5
md = 4
gm = 2.5

n_trees = 500
params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    "eval_metric": "mlogloss",
    "num_class":3,
    'max_depth':md,
    'min_child_weight':mc,
    'subsample':ss,
    'colsample_bytree':1,
    'gamma':gm,
    "eta": 0.01,
    "lambda":1,
    'alpha':0,
    "silent": 1,
}
if 'tr' in task:
    for tr,va in StratifiedShuffleSplit(y,n_iter=1,test_size=0.2,random_state=1):
        X_tr = X.iloc[tr]
        y_tr = y[tr]
        X_va = X.iloc[va]
        y_va = y[va]
    dtrain = xgb.DMatrix(X_tr, y_tr)
    dvalid = xgb.DMatrix(X_va, y_va)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, 2000, evals=watchlist,
                    early_stopping_rounds=25, verbose_eval=20)
if 'sub' in task:
    dtrain = xgb.DMatrix(X, y)
    dtest = xgb.DMatrix(X_all[TR:])
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, n_trees, evals=watchlist,
                    early_stopping_rounds=25, verbose_eval=100)
    prds.append(bst.predict(dtest))
#################################################
label = 'sex'
print('='*20)
print(label)
print('='*20)
y = ys[label]

ss = 1.0
mc = 1.5
md = 4
gm = 4

n_trees = 429
params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "logloss",
    'max_depth':md,
    'min_child_weight':mc,
    'subsample':ss,
    'colsample_bytree':1,
    'gamma':gm,
    "eta": 0.01,
    "lambda":3,
    'alpha':0,
    "silent": 1,
}
if 'tr' in task:
    for tr,va in StratifiedShuffleSplit(y,n_iter=1,test_size=0.2,random_state=1):
        X_tr = X.iloc[tr]
        y_tr = y[tr]
        X_va = X.iloc[va]
        y_va = y[va]
    dtrain = xgb.DMatrix(X_tr, y_tr)
    dvalid = xgb.DMatrix(X_va, y_va)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, 2000, evals=watchlist,
                    early_stopping_rounds=25, verbose_eval=20)
if 'sub' in task:
    dtrain = xgb.DMatrix(X, y)
    dtest = xgb.DMatrix(X_all[TR:])
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, n_trees, evals=watchlist,
                    early_stopping_rounds=25, verbose_eval=100)
    _prd = bst.predict(dtest)
    prd = np.zeros((len(_prd),2))
    prd[:,1] = _prd
    prd[:,0] = 1 - prd[:,1]
    prds.append(prd)
#################################################
label = 'loc'
print('='*20)
print(label)
print('='*20)
y = ys[label]

ss = 0.4
mc = 2.5
md = 5
gm = 2.5

n_trees = 616
params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    "eval_metric": "mlogloss",
    "num_class":8,
    'max_depth':md,
    'min_child_weight':mc,
    'subsample':ss,
    'colsample_bytree':1,
    'gamma':gm,
    "eta": 0.01,
    "lambda":1,
    'alpha':0,
    "silent": 1,
}
if 'tr' in task:
    for tr,va in StratifiedShuffleSplit(y,n_iter=1,test_size=0.2,random_state=1):
        X_tr = X.iloc[tr]
        y_tr = y[tr]
        X_va = X.iloc[va]
        y_va = y[va]
    dtrain = xgb.DMatrix(X_tr, y_tr)
    dvalid = xgb.DMatrix(X_va, y_va)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain, 2000, evals=watchlist,
                    early_stopping_rounds=25, verbose_eval=20)
if 'sub' in task:
    dtrain = xgb.DMatrix(X, y)
    dtest = xgb.DMatrix(X_all[TR:])
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, n_trees, evals=watchlist,
                    early_stopping_rounds=25, verbose_eval=100)
    prds.append(bst.predict(dtest))

##############################################
if 'sub' in task:
    df_sub = pd.DataFrame()
    df_sub['uid'] = df_mge.iloc[TR:]['uid']
    n = len(df_sub)
    df_sub['age'] = age_le.inverse_transform(prds[0].argmax(axis=1))
    df_sub['gender'] = sex_le.inverse_transform(prds[1].argmax(axis=1))
    df_sub['province'] = loc_le.inverse_transform(prds[2].argmax(axis=1))
    df_sub.to_csv('./predictions/temp.csv',index=None)
    print('sub finish!')

#age [499]  train-mlogloss:0.682963 eval-mlogloss:0.80601
#sex [429]  train-logloss:0.278002  eval-logloss:0.340442
#loc [616]  train-mlogloss:0.793604 eval-mlogloss:1.18464

#LB 0.66378