import numpy as np
import pandas as pd
import csv

folder = "0000_0003_20221123_0000_genre_test"
path = "./KK_new/triplets/"+folder
filmarks_kg_RAC_tsv_path="."

kg_ent = {}
ent1 = []
rels = []
ent2 = []

with open(filmarks_kg_RAC_tsv_path+"/filmarks_kg_RAC.tsv") as f:
    for line in f:
        (key, rel_, val2) = line.split("\t")
        kg_ent[(key)] = val2
        val = kg_ent[(key)].split("\n")[-2]
        kg_ent[(key)] = (rel_,val)
        ent1.append(key)
        ent2.append(val2.split("\n")[-2])
        rels.append(rel_)

print(len(ent1))
print(len(ent2))
print(len(rels))
print(len(set(ent1)))
print(len(set(ent2)))
print(len(set(rels)))
print(len(set(ent1))+len(set(ent2)))



count = 0
with open(path+"/ent_ids_1") as f:
    for line in f:
        count+=1
print(count)

## make ent_ids_2
all_item_ = []
all_item = []
ids = []
for i in set(ent1):
    s = i.replace('"','')
    all_item_.append(s)
print(len(all_item_))
for i in set(ent2):
    s = i.replace('"','')
    all_item_.append(s)
print(len(all_item_))

print("======")
print(len(set(all_item_)))
for i in set(all_item_):
    all_item.append(i)
    ids.append(count)
    count+=1
print(len(all_item))
print(len(set(all_item)))
print(len(ids))


ent_ids_1 = pd.DataFrame()
ent_ids_1['id'] = ids
ent_ids_1['ent_name'] = all_item
print(len(ent_ids_1['id']))
print(len(ent_ids_1['ent_name']))
print(len(set(ent_ids_1['ent_name'])))
print(len(ent_ids_1))

ent_ids_1.to_csv(path+"/ent_ids_2", header=None, index=None, sep='\t')


id2  = []
name2 = []
f = open(path+'/ent_ids_2', 'r')
for line in f.readlines():
    # print(line)
    s = line.split("\t")
    
    id2.append(s[0])
    name2.append(s[1].split('\n')[-2])

ent_ids_1 = pd.DataFrame()
ent_ids_1['id'] = id2
ent_ids_1['ent_name'] = name2
print(len(ent_ids_1['id']))
print(len(ent_ids_1['ent_name']))

ids  = []
all_item = []
f = open(path+'/rel_ids', 'r')
for line in f.readlines():
    # print(line)
    s = line.split("\t")
    
    ids.append(s[0])
    all_item.append(s[1].split("\n")[-2])

rel_ids = pd.DataFrame()
rel_ids['id'] = ids
rel_ids['ent_name'] = all_item

rel_dict = {}
for i in range(len(rel_ids)):
    rel_dict[rel_ids['ent_name'][i]] = rel_ids['id'][i]
rel_dict

ent_dict = {}
for i in range(len(ent_ids_1)):
    ent_dict[ent_ids_1['ent_name'][i]] = ent_ids_1['id'][i]
print(len(ent_dict))



kg_ent = {}
ent1 = []
rels = []
ent2 = []

with open(filmarks_kg_RAC_tsv_path+"/filmarks_kg_RAC.tsv") as f:
    for line in f:
        (key, rel, val2) = line.split("\t")
        key_ = key.replace('"','')
        rel_ = rel.replace('"','')
        val2_ = val2.replace('"','')
        
        
        kg_ent[(key_)] = val2_
        val = kg_ent[(key_)].split("\n")[-2]
        kg_ent[(key_)] = (rel_,val)
        ent1.append(ent_dict[key_])
        ent2.append(ent_dict[val])
        rels.append(rel_dict[rel_])

print(len(ent1))
print(len(ent2))
print(len(rels))

new = pd.DataFrame()
new['ent1'] = ent1
new['rels'] = rels
new['ent2'] = ent2

new.to_csv(path+"/triples_2", header=None, index=None, sep='\t')