import os
import json
from collections import Counter
from pprint import pprint
from copy import deepcopy
from more_itertools import sort_together

entity_path = 'example_entities/entities/'
prev_entity_counts = json.load(open('example_entities/entity_counts.json','r'))
entity_rep_counter = Counter()
type_rep_counter = dict()
inverted_index = dict()

def dict_formation_helper(path,count):
    global type_rep_counter
    nodes = path.split('/')
    nodes=nodes[2:]
    curr = deepcopy(type_rep_counter)

    for index,i in enumerate(nodes):
        if i in curr:
             curr = curr[i]
        else:
            break


    prev=count
    for i in nodes[-1:-(len(nodes)-index)-1:-1]:
        temp_dict = dict()
        temp_dict[i]=prev
        prev = temp_dict

    curr = prev

    temp = type_rep_counter

    if index==0:
        type_rep_counter[nodes[0]]=curr[nodes[0]]
    else:
        for i in nodes[:index-1]:
            temp= temp[i]
        temp = temp.get(nodes[index-1],{})
        temp.update(curr)







count_dict={}
counter=0
for i,j,k in os.walk(entity_path):
    for file in k:
        if not file.endswith('.txt'):
            continue
        counter+=1
        ents = [ent for ent in open(i+'/'+file,'r').read().split('\n') if ent.startswith('Q')]
        ents = list(set(ents))
        if len(ents)<2:
            continue

        if len(ents)>1000:
            counts = [prev_entity_counts[ent] if ent in prev_entity_counts else -1 for ent in ents]
            s_c,s_e = sort_together([counts,ents])
            ents = s_e[:500]+s_e[-500:]

        #Update the file count
        dict_formation_helper(i+'/'+file,len(ents))


        #Deduplicate files
        # with open(i+'/'+file,'w') as f:
        #     for ent in ents:
        #         f.write(ent)
        #         f.write('\n')

        #Update entity counts
        entity_rep_counter.update(ents)

        #Update inverted index
        nodes = i.split('/')[2:]+[file]
        for ent in ents:
            ent_row = inverted_index.get(ent,[])
            ent_row.append(nodes)
            inverted_index[ent]=ent_row




print('Total number of entities ',len(entity_rep_counter))
print('Total number of files ',counter)
json.dump(type_rep_counter,open(entity_path.split('/')[0]+'/'+'small_type_counts.json','w')) 
json.dump(entity_rep_counter,open(entity_path.split('/')[0]+'/'+'small_entity_counts.json','w'))
json.dump(inverted_index,open(entity_path.split('/')[0]+'/'+'small_inverted_index.json','w'))


