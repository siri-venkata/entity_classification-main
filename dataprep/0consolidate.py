import pandas as pd
import json
import numpy as np
import types

csv_path = 'code2query.csv'
json_path = 'final.json'
query_path ='example_entities/queries/'

df = pd.read_csv(csv_path,)
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

def check_csv_json_comptibility(list_tags,df):
    a=[]
    for i in range(len(list_tags)-1):
        if list_tags[i][-1] == df['tag'].values[i]:
            # print(df['tag'].values[i])
            a.append(0)
        else:
            a.append(1)
            print(i)
            print(list_tags[i])
            print(df['tag'].values[i])
            raise ValueError
    if sum(a)==0:
        print('All tags are compatible')
    return a        


if __name__=="__main__":
    a = check_csv_json_comptibility(list_tags,df)
    if sum(a):
        i = np.cumsum(a[::-1]).tolist().index(1)
        bad_till = len(list_tags)-i-1
        print('Bad tags till index',bad_till)
        print('Total Length',len(a))

