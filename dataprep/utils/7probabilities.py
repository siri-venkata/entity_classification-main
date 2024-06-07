import os
import json
from collections import Counter
from pprint import pprint
from copy import deepcopy
import numpy as np
from tqdm import tqdm

entity_path = 'example_entities/entities/'
type_path  = 'example_entities/type_counts.json'

entity_rep_counter = Counter()
type_rep_counter = dict()
inverted_index = dict()


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def flatten_dict(d):
    if type(d)==dict:   
        res = [[i]+flatten_dict(d[i]) for i in d]
        return [j for i in res for j in i]
    return [str(d)]

def flatten_list(l):
    res = [flatten_list(i) if type(i)==list else [i] for i in l]
    return [j for i in res for j in i]

def count_labels():
    labels = read_json(type_path)
    num_labels = len(set([i.replace('.txt','') for i in flatten_dict(labels) if not i.isnumeric()]))
    print('Number of labels: ',num_labels)
    return labels,num_labels

def get_label_maps():
    labels,num_labels = count_labels()
    id2label = {}
    label2id = {}

    A = np.zeros((num_labels,num_labels))
    M = np.eye(num_labels)
    def traverse(d,ancestors = None,count=0):
        if type(d)==int: return count
        for i in d:
            unlabelled = 1 if i.replace('.txt','') not in label2id else 0
            node_id = count if unlabelled else label2id[i.replace('.txt','')]
            id2label[node_id] = i.replace('.txt','')
            label2id[i.replace('.txt','')] = node_id

            if ancestors:
                A[ancestors[-1]][node_id]=1
                for j in ancestors:
                    try:
                        M[j][node_id] = 1
                    except:
                        import pdb;pdb.set_trace()
                count = traverse(d[i],ancestors+[node_id],count+1 if unlabelled else count)
            else:
                count = traverse(d[i],[node_id],count+1 if unlabelled else count)
        return count
    highest = traverse(labels)

    assert num_labels==len(id2label)
    assert max(id2label.keys())+1==highest
    
    return A,M,labels,label2id,id2label,num_labels 


A,M,labels,label2id,id2label,num_labels = get_label_maps()
print(num_labels,len(label2id))


def verify_if_DAG(A):
    import networkx as nx
    G = nx.from_numpy_matrix(A,create_using=nx.DiGraph)
    try:
        assert nx.is_directed_acyclic_graph(G)
    except:
        return [[id2label[node] for node in cycle] for cycle in nx.simple_cycles(G)]
    return True

print('Verifying if DAG',verify_if_DAG(A))



def match_labels(leaf_node):
    '''
    Used to find a path till root class from the leaf node
    '''
    parents = [index for index,i in enumerate(A[:,leaf_node]) if i==1]
    return [leaf_node,[match_labels(parent) for parent in parents]]



def get_class_dict():
    count_dict={}
    counter=0
    class_dict = {}
    file_count=0
    for i,j,k in os.walk(entity_path):
        for file in k:
            file_count+=1
            print(file_count,'                                      ',end='\r')
            if not file.endswith('.txt'):
                continue
            counter+=1
            ents = [ent for ent in open(i+'/'+file,'r').read().split('\n') if ent.startswith('Q')]
            ents = list(set(ents))
            if len(ents)<2:
                continue
            try:
                parent_classes = flatten_list(match_labels(label2id[file.replace('.txt','')]))
                parent_classes = [id2label[i] for i in parent_classes]

                for class_ in parent_classes:
                    class_dict[class_] = list(set(class_dict.get(class_,[])+ents))
            except Exception as e:
                print(e)
                print(i,file)
                import pdb;pdb.set_trace()
    return class_dict

if os.path.exists('example_entities/labels/class_dict.json'):
    class_dict = read_json('example_entities/labels/class_dict.json')
else:
    class_dict = get_class_dict()
    with open('example_entities/labels/class_dict.json','w') as f:
        json.dump(class_dict,f)


print(len(class_dict))

def get_conditional_probs(class_dict):
    conditional_probs = np.zeros((num_labels,num_labels))
    intersections=np.zeros((num_labels,num_labels))
    sets={i:set(class_dict[i]) for i in tqdm(class_dict,leave=False)}
    for i in tqdm(class_dict):
        I = label2id[i]
        denom = 0.001+len(class_dict[i])
        
        for j in tqdm(class_dict,leave=False):
            J = label2id[j]
            if intersections[I][J]==0:
                intersections[I][J] = len(sets[i] & sets[j])#len(set(class_dict[i]).intersection(set(class_dict[j])))
            conditional_probs[I][J] = intersections[I][J]/denom

    conditional_dict = {i:{j:conditional_probs[label2id[i]][label2id[j]] for j in class_dict} for i in class_dict}
    return conditional_dict

if os.path.exists('example_entities/labels/conditional_dict.json'):
    conditional_dict = read_json('example_entities/labels/conditional_dict.json')
else:
    conditional_dict = get_conditional_probs(class_dict)
    with open('example_entities/labels/conditional_dict.json','w') as f:
        json.dump(conditional_dict,f)

def shortest_paths(A,no_path=2000):
    print('Finding shortest paths')
    import networkx as nx
    G = nx.from_numpy_matrix(A,create_using=nx.DiGraph)
    #Convert to undirected graph
    G = G.to_undirected()
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    sp={}
    for i in shortest_paths:
        sp[id2label[i]] = {id2label[j]:shortest_paths[i][j] if j in shortest_paths[i] else no_path for j in range(A.shape[0])}
        # sp[id2label[i]] = {id2label[j]:sp[i][j] for j in sp[i]}
    return sp


if os.path.exists('example_entities/labels/shortest_paths.json'):
    shortest_paths = read_json('example_entities/labels/shortest_paths.json')
else:
    shortest_paths = shortest_paths(A)
    with open('example_entities/labels/shortest_paths.json','w') as f:
        json.dump(shortest_paths,f)






