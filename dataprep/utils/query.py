import sys
import time
from SPARQLWrapper import SPARQLWrapper, JSON

endpoint_url = "https://query.wikidata.org/sparql"

query = """
SELECT ?item ?itemLabel 
WHERE 
{
  ?item wdt:P31 wd:Q146

  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } # Helps get the label in your language, if not, then en language
}

"""

bad_query = """
SELECT ?item ?itemLabel 
WHERE 
{
  ?item wdt:P31 wd:Q3305213

  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } # Helps get the label in your language, if not, then en language
}

"""





class Qrunner():
    def __init__(self):
        self.endpoint_url = "https://query.wikidata.org/sparql"
        self.user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        self.dlq=[]

    def get_results(self,query):
        sparql = SPARQLWrapper(self.endpoint_url, agent=self.user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results =  sparql.query().convert()
        return [result['item']['value'].split('/')[-1] for result in results["results"]["bindings"]]
    
    def query_runner(self,index,query,ent):
        query = query.replace('__/\__',ent)
        counter= 0
        while counter<3:
            try:
                res = self.get_results(query)
                return res
            except Exception as e:
                # print('Failed for index ',index,' and entity ',ent)
                counter+=1
                time.sleep(5)
                a = str(type(e))
                # print('Retrying')
        # print('Moving to dlq')
        self.dlq.append([index,query,ent,a])
        return []
                




qr = Qrunner()

if __name__=="__main__":  
    results = qr.query_runner(0,query,'d')
    print(results)
    print(qr.dlq)

    
