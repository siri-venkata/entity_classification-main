import json
import os
import pandas as pd
from tqdm import tqdm


home_path = '/scratch/'

model_names = ['naive_cbc','naive_rac','naive_dac','naive_radac',]
#        'hierarchial_cbc','hierarchial_rac','hierarchial_dac','hierarchial_radac',
#        'graph_cbc','graph_rac','graph_dac','graph_radac']

class HF1Computer():
    def __init__(self):
        self.TP = 0
        self.prediction_count = 0
        self.label_count = 0

    def update_counts(self,predictions,labels):
        self.TP+=len(set(predictions).intersection(set(labels)))
        self.prediction_count+=len(set(predictions))
        self.label_count+=len(set(labels))

    def add(self,predictions,labels):
        for prediction,label in tqdm(zip(predictions,labels)): self.update_counts(prediction,label)

    def reset(self):
        self.TP = 0
        self.prediction_count = 0
        self.label_count = 0

    def hierarchial_precision_(self):
        return self.TP/self.prediction_count if self.prediction_count>0 else 0


    def hierarchial_recall_(self):
        return self.TP/self.label_count if self.label_count>0 else 0
    def hierarchial_f1(self):

        HP = self.hierarchial_precision_()
        HR = self.hierarchial_recall_()
        HF1 = 2*HP*HR/(HP+HR) if HP+HR>0 else 0

        if self.label_count==0:
            raise ValueError('Calculated metrics after providing results atleast once')
        return {'HP':HP,'HR':HR,'HF1':HF1,}



def try_and_load_results(filename):
    try:
        df_eval = pd.read_csv(filename+'/output/predictionsevalbest.csv')
        df_train = pd.read_csv(filename+'/output/predictionstrain.csv')
        return df_train,df_eval
    except:
        try:
            df_eval = pd.read_csv(filename+'/predictionsevalbest.csv')
            df_train = pd.read_csv(filename+'/predictionstrain.csv')
            return df_train,df_eval
        except:
            print(f'Couldn\'t load {filename}')
            return None,None

def evaluate_file(model_name,df):
    print(f'============================{model_name}============================')
    model_name = model_name.split()[0]
    languages = "aswiki|bnwiki|guwiki|hiwiki|knwiki|mlwiki|mrwiki|orwiki|pawiki|tawiki|tewiki|enwiki".replace('wiki','').split('|')
    rows = []
    for langid,lang in enumerate(languages):
        subset = df[df['lang']==langid]
        hfcompute = HF1Computer()
        hfcompute.add(subset['predictions'].values,subset['ground truth'].values)
        print(f'====={lang} {subset.shape}========')
        try:
            res={'model_type':model_name.split('_')[0],'loss_type':model_name.split('_')[1],'lang':lang}
            res.update(hfcompute.hierarchial_f1())
            rows.append(res)
        except:
            pass
    rows =  pd.DataFrame(rows)
    new_row = {'model_type':model_name.split('_')[0],'loss_type':model_name.split('_')[1],'lang':'Average',
            'HP':rows['HP'].mean(),'HR':rows['HR'].mean(),'HF1':rows['HF1'].mean()}
    rows = pd.concat([rows, pd.DataFrame([new_row])], ignore_index=True)
    return rows

train_results =[]
eval_results = []
for model_name in model_names:
    df_train,df_eval = try_and_load_results(home_path+model_name)
    if df_train is None  or df_eval is None: continue
    #print(model_name,df_train.shape,df_eval.shape)
    train_results.append(evaluate_file(model_name+' train',df_train))
    eval_results.append(evaluate_file(model_name+' eval',df_eval))
    #train_results.to_csv(model_name+'_train_results,csv',index=False)
    #eval_results.to_csv(model_name+'_eval_results.csv',index=False)
    #import pdb;pdb.set_trace()

train_df = pd.concat(train_results)
train_df.to_csv('train_results.csv',index=False)

eval_df = pd.concat(eval_results)
eval_df.to_csv('eval_results.csv',index=False)
                                                                                              
