import os
import json
from pprint import pprint
from more_itertools import chunked
from tqdm import tqdm
import requests

titles_path ='example_entities/articles/titles.json'
article_path = 'example_entities/articles/'
languages = "aswiki|bnwiki|guwiki|hiwiki|knwiki|mlwiki|mrwiki|orwiki|pawiki|tawiki|tewiki|enwiki"

titles = json.load(open(titles_path,'r'))

counters= {i:0 for i in languages.split('|')}

for key,values in titles.items():
    for lang,_ in values.items():
        counters[lang]+=1

pprint(counters)
print('Total articles ',sum(counters.values()))


repack_titles = {i:[title for key,val in titles.items() for lang,title in val.items() if lang==i] for i in languages.split('|')}

dlq =[]
def parse_response(response,lang):
    articles ={}
    try:
        data = response.json()
        for page in data["query"]["pages"]:
            articles[page["title"]]=page["revisions"][0]["content"]
    except KeyError:
        if page["missing"]:
            dlq.append([page["title"],lang])
    return articles

payload={
	"action": "query",
	"format": "json",
	"prop": "revisions",
	"titles": "",
	"formatversion": "2",
	"rvprop": "content",
}
def run_query(url,titles):
    payload["titles"]="|".join([i.replace('|','\|') for i in titles])
    return requests.get(url =url,params=payload)

def get_content(lang,titles):
    all_articles = json.load(open(article_path+lang+'wiki.json','r'))
    url = "https://en.wikipedia.org/w/api.php"
    url = url.replace("en",lang)
    batch_size=49
    titles = [i for i in titles if i not in all_articles]
    q = list(chunked(titles,batch_size))
    timeout_counter=0
    while len(q):
        curr = q.pop()
        response = run_query(url,curr)
        if response.status_code ==414:
            timeout_counter+=1
            if len(curr)<=1:
                raise ValueError
            q.append(curr[:len(curr)//2])
            q.append(curr[len(curr)//2:])
        elif response.status_code==200:
            response = parse_response(response,lang)
            all_articles.update(response)
        else:
            raise ValueError
        print(str(len(q))+'                      ',end='\r')
    print('Timed out ',timeout_counter,' times')
    return all_articles

for lang,titles in tqdm(repack_titles.items()):
    content = get_content(lang[:2],titles)
    json.dump(content,open(article_path+lang+'.json','w'))

json.dump(dlq,open(article_path+'dlq.json','w'))
# get_content('en',['Football','Iceland','Pet door'])