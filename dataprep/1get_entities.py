import pandas as pd
import json
import os
from utils.query import qr
from tqdm import tqdm

csv_path = 'code2query.csv'
prev_csv_path = 'code2query_old.csv'
json_path = 'final.json'
query_path ='example_entities/queries/'
entity_path ='example_entities/entities/'

df = pd.read_csv(csv_path,)
df.fillna('',inplace=True)

df_prev = pd.read_csv(prev_csv_path,)
df_prev.fillna('',inplace=True)


tags = json.load(open(json_path,'r'))



def traverse_json(j,prefix=None):
    # print('Called function with prefix',prefix)
    if isinstance(j,str):
        yield j if prefix is None else prefix+[j]
    elif isinstance(j,list):
        for item in j:
            yield from traverse_json(item,prefix)
    elif isinstance(j,dict):
        for key,value in j.items():
            prefix_ = prefix+[key] if prefix is not None else [key]
            yield from traverse_json(value,prefix_)
    elif j is None:
        yield prefix
    else:
        raise TypeError
    

list_tags = list(traverse_json(tags))






def query_helper(index,file,entity):
    if '+' in entity:
        ents = entity.split('+')
    else:
        ents=[entity]
    
    query = open(file,'r').read()
    res=[]
    for ent in ents:
        res += qr.query_runner(index,query,ent)
    return res
    

try:
    for i in tqdm(range(len(list_tags))):
        path = entity_path+'/'.join(list_tags[i])+'.txt'
        if df['entity'].values[i]=='' or df['file'].values[i]=='':
            res=[]
            print('None for', i)
        elif df['entity'].values[i]==df_prev['entity'].values[i] and df['file'].values[i]==df_prev['file'].values[i]:
            continue
        else:
            res = query_helper(i,query_path+df['file'].values[i],df['entity'].values[i])
            print(i)
            print(path)

        os.makedirs(entity_path+'/'.join(list_tags[i][:-1]),exist_ok=True)
        with open(path,'w') as f:
            for i in res:
                f.write(i)
                f.write('\n')
except Exception as e:
    json.dump(qr.dlq,open('dlq.json','w'))
    raise

json.dump(qr.dlq,open('dlq.json','w'))




