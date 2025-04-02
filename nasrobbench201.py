# -*- coding: utf-8 -*-
import os
import json
from collections import defaultdict

class NasRobBench201:
    def __init__(self, metapath,datasetpath):
        
        with open(metapath) as f:
            self.meta = json.load(f)
            self.map_str_to_id = {m["nb201-string"]:k for k,m in self.meta["ids"].items()}
            self.non_isomorph_ids = [i for i, d in self.meta["ids"].items() if d["isomorph"]==i]

        with open(datasetpath) as f:
            self.res = json.load(f)

    def get_uid(self, i):
        """
        Returns the evaluated architecture id (if given id is isomorph to another network)
        
        Parameters
        ----------
        i : str/int
            Architecture id.
        """
        return self.meta["ids"][str(i)]["isomorph"]
    
    def get_result(self,dataset,architecture_id,metric):
        architecture_id = self.get_uid(architecture_id)
        return self.res[dataset][architecture_id][metric]['1']
    
    def extract_info_benchmark(self, dest_folder):
        os.makedirs(dest_folder, exist_ok= True) 
        for dt, evals in self.res.items(): 
            filename = os.path.join(dest_folder, f"{dt}.json")
            map_id_to_str = {val:key for key, val in self.map_str_to_id.items()}
            new_dict = {}
            for key, val in evals.items(): 
                new_key = map_id_to_str.get(key)
                new_dict[new_key] = val
            
            # write result to json 
            with open(filename, "w") as f: 
                json.dump(new_dict, f, indent= 4)

        return True  
    
    def group_same_isomorph(self, dest_folder): 
        os.makedirs(dest_folder, exist_ok= True)
        filename = os.path.join(dest_folder, f"isomorph.json")
        
        group_data = defaultdict(list)
        for idx, evals in self.meta.get('ids').items(): 
            group_data[evals['isomorph']].append(evals['nb201-string'])
        
        # save result 
        with open(filename, 'w', encoding= 'utf-8') as f: 
            json.dump(group_data, f, indent= 4, ensure_ascii= False)
        
        return True 

    
if __name__ == "__main__": 
    meta_path = "dataset/meta.json"
    dataset_path = "dataset/NAS-RobBench-201_all.json"
    processor = NasRobBench201(meta_path, dataset_path)
    result = processor.extract_info_benchmark("dataset/")
    result = processor.group_same_isomorph("dataset/")