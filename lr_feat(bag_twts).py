import pandas as pd
import numpy as np
import re
import xgboost as xgb

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
    
def mlogloss(yhat,y):
    return np.mean([-np.log(yhat[i,_y]) for i,_y in enumerate(y)])
def macc(yhat,y):
    return np.mean(yhat.argmax(axis=1) == y)

def lb_score(yhat_age,y_age,yhat_sex,y_sex,yhat_loc,y_loc):
    print('age mlogloss:',mlogloss(yhat_age,y_age))
    a1 = macc(yhat_age,y_age)
    print('age macc:',a1)
    
    print('sex mlogloss:',mlogloss(yhat_sex,y_sex))
    a2 = macc(yhat_sex,y_sex)
    print('sex macc:',a2)
    
    print('loc mlogloss:',mlogloss(yhat_loc,y_loc))
    a3 = macc(yhat_loc,y_loc)
    print('loc macc:',a3)
    
    print('LB score:',0.3*a1+0.2*a2+0.5*a3)

df_mge['loc_bin'] = df_mge['loc'].map(bin_loc)
df_mge['age_bin'] = df_mge['age'].map(bin_age)

tokenizer = lambda s:s.split(' ')
tfv = TfidfVectorizer(tokenizer=tokenizer)
X_all_sp = tfv.fit_transform(df_mge.bag_twts)
X_sp = X_all_sp[:3200]

age_le = LabelEncoder()
y_age = age_le.fit_transform(df_mge.iloc[:3200]['age_bin'])

loc_le = LabelEncoder()
y_loc = loc_le.fit_transform(df_mge.iloc[:3200]['loc_bin'])

sex_le = LabelEncoder()
y_sex = sex_le.fit_transform(df_mge.iloc[:3200]['sex'])

prds = []

lb = 'age'
y = y_age
C=5
for tr,va in StratifiedShuffleSplit(y,n_iter=1,test_size=0.2,random_state=1):
    X_tr = X_sp[tr]
    y_tr = y[tr]
    X_va = X_sp[va]
    y_va = y[va]

lr = LogisticRegression(C=C)
lr.fit(X_tr,y_tr)
print(lb,C,mlogloss(lr.predict_proba(X_tr),y_tr),mlogloss(lr.predict_proba(X_va),y_va))
print(lb,C,macc(lr.predict_proba(X_tr),y_tr),macc(lr.predict_proba(X_va),y_va))
prds.append(lr.predict_proba(X_va))
prds.append(y_va)

lb = 'sex'
y = y_sex
C=16
for tr,va in StratifiedShuffleSplit(y,n_iter=1,test_size=0.2,random_state=1):
    X_tr = X_sp[tr]
    y_tr = y[tr]
    X_va = X_sp[va]
    y_va = y[va]

lr = LogisticRegression(C=C)
lr.fit(X_tr,y_tr)
print(lb,C,mlogloss(lr.predict_proba(X_tr),y_tr),mlogloss(lr.predict_proba(X_va),y_va))
print(lb,C,macc(lr.predict_proba(X_tr),y_tr),macc(lr.predict_proba(X_va),y_va))
prds.append(lr.predict_proba(X_va))
prds.append(y_va)

lb = 'loc'
y = y_loc
C=16
for tr,va in StratifiedShuffleSplit(y,n_iter=1,test_size=0.2,random_state=1):
    X_tr = X_sp[tr]
    y_tr = y[tr]
    X_va = X_sp[va]
    y_va = y[va]

lr = LogisticRegression(C=C)
lr.fit(X_tr,y_tr)
print(lb,C,mlogloss(lr.predict_proba(X_tr),y_tr),mlogloss(lr.predict_proba(X_va),y_va))
print(lb,C,macc(lr.predict_proba(X_tr),y_tr),macc(lr.predict_proba(X_va),y_va))
prds.append(lr.predict_proba(X_va))
prds.append(y_va)

lb_score(prds[0],prds[1],prds[2],prds[3],prds[4],prds[5])

prds = []
n = 3200

y = y_age
C=5
lr = LogisticRegression(C=C)
lr.fit(X_all_sp[:n],y)
prds.append(lr.predict_proba(X_all_sp[n:]))

y = y_sex
C=16
lr = LogisticRegression(C=C)
lr.fit(X_all_sp[:n],y)
prds.append(lr.predict_proba(X_all_sp[n:]))

y = y_loc
C=16
lr = LogisticRegression(C=C)
lr.fit(X_all_sp[:n],y)
prds.append(lr.predict_proba(X_all_sp[n:]))

df_sub = pd.DataFrame()
df_sub['uid'] = df_mge.iloc[3200:]['uid']
n = len(df_sub)
df_sub['age'] = age_le.inverse_transform(prds[0].argmax(axis=1))
df_sub['gender'] = sex_le.inverse_transform(prds[1].argmax(axis=1))
df_sub['province'] = loc_le.inverse_transform(prds[2].argmax(axis=1))
df_sub.to_csv('./predictions/temp.csv',index=None)

# age mlogloss: 0.854034688384
# age macc: 0.5890625
# sex mlogloss: 0.396567507695
# sex macc: 0.8203125
# loc mlogloss: 1.51318278587
# loc macc: 0.433489827856
# LB score: 0.557526163928

# LB score:0.57531