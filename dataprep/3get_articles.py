import os
import json
from pprint import pprint

from more_itertools import chunked
from tqdm import tqdm
import requests
from copy import deepcopy


entity_path = 'example_entities/small_entity_counts.json'
article_path = 'example_entities/articles/'
prev_titles = json.load(open(article_path+'titles.json','r'))
try:
    missing_ids = set(json.load(open(article_path+'missing_ids.json','r')))
except:
    missing_ids = set()

entities = json.load(open(entity_path,'r'))



matching_titles = set(entities.keys())&set(prev_titles.keys())


for i in prev_titles:
    entities.pop(i,None)

for i in missing_ids:
    entities.pop(i,None)

# print(len(entities))
# print(len(prev_titles))

# print(len(set(entities.keys())-set(prev_titles.keys())))
# print(len(set(prev_titles.keys())-set(entities.keys())))

# exit()

languages = "aswiki|bnwiki|guwiki|hiwiki|knwiki|mlwiki|mrwiki|orwiki|pawiki|tawiki|tewiki|enwiki"
reqUrl = "https://www.wikidata.org/w/api.php"
payload={
	"action": "wbgetentities",
	"format": "json",
	"ids": "Q42|Q146",
	"props": "sitelinks",
    "sitefilter":languages,
	"formatversion": "2"
}



def process_response(response):
    if response.status_code!=200:
        raise ValueError
    try:
        data = eval(response.text)
        langs = languages.split('|')
        clean_dict = dict()
        for ent,i in data['entities'].items():
            try:
                for l,j in i['sitelinks'].items():
                    temp  = clean_dict.get(ent,{})
                    temp.update({l:j['title']})
                    clean_dict[ent] = temp
            except KeyError:
                clean_dict[ent]={}


        return clean_dict
    except:
        print(response.text)
        raise

def make_request(ents):
    ids = "|".join(ents)
    payload["ids"]=ids
    response = requests.get(reqUrl,params=payload)
    clean_dict = process_response(response)
    missing_ids = set(ents)-set(clean_dict.keys())
    return clean_dict,missing_ids

counter=0
batch_size=49
for i in chunked(entities,batch_size):
    counter+=1

titles=deepcopy({i:prev_titles[i] for i in prev_titles if i in matching_titles})
for i in tqdm(chunked(entities,batch_size),total=counter):
    clean_dict,m = make_request(i)
    titles.update(clean_dict)
    missing_ids.update(m)

# print(len(titles))
# print(len(prev_titles))

# print(len(set(titles.keys())-set(prev_titles.keys())))
# print(len(set(prev_titles.keys())-set(titles.keys())))
# json.dump(titles,open(article_path+'new_titles.json','w'))

json.dump(titles,open(article_path+'titles.json','w'))
json.dump(list(missing_ids),open(article_path+'missing_titles.json','w'))