import os
import json
from pprint import pprint
from more_itertools import chunked
from collections import defaultdict
import requests


titles_path ='example_entities/articles/titles.json'
article_path = 'example_entities/articles/'
languages = "aswiki|bnwiki|guwiki|hiwiki|knwiki|mlwiki|mrwiki|orwiki|pawiki|tawiki|tewiki|enwiki"

titles = json.load(open(titles_path,'r'))


lang_titles={}
lang_ids={}
for lang in languages.split('|'):
    title_lookup=defaultdict(list)
    id_lookup=defaultdict(list)
    for key,val in titles.items():
        try:
            title_lookup[key].append(val[lang])
            id_lookup[val[lang]].append(key)
        except:
            pass
    lang_titles[lang] = title_lookup
    lang_ids[lang] = id_lookup

json.dump(lang_titles,open(titles_path.replace('titles','lang_titles'),'w'))
json.dump(lang_ids,open(titles_path.replace('titles','lang_ids'),'w'))

print('Duplicates in Titles')
for lang in lang_titles:
    for key,val in lang_titles[lang].items():
        if len(val)>1:
            print(key,' ',val)
    print('For lang ',lang,'total titles ',len(lang_titles[lang]))

print('Duplicates in ids')
for lang in lang_ids:
    for key,val in lang_ids[lang].items():
        if len(val)>1:
            print(key,' ',val)
    print('For lang ',lang,'total ids ',len(lang_ids[lang]))

