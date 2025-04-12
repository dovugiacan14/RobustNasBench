import pickle 
import json 

# path  = "results/SO-NAS201-1/SO-NAS201-1_GA_epe_nas_0804173100/1/best_architecture_each_gen.p"
# with open(path, "rb") as file: 
#     data = pickle.load(file)

# print(data)

file = "config/cifar10.json"
with open(file, "r") as f:
    conf = json.load(f)
print(conf)



