import numpy as np
import pandas as pd
import csv

folder = "0000_0003_20221123_0000_genre_test"
path = "./KK_new/triplets/"+folder
items_tsv_path = "KK_new"

kg_ent = {}
kg_attr = {}
ent1 = []
rels = []
ent2 = []

with open(path+"/kg.tsv") as f:
    for line in f:
        (key, rel_, val2) = line.split("\t")
        kg_ent[(key)] = val2
        val = kg_ent[(key)].split("/")[-1].split("\n")[-2]
        if ":" in val:
            val_ = val.split(":")[1]
            val=val_
        kg_ent[(key)] = (rel_,val)
        ent1.append(key)
        ent2.append(val)
        rels.append(rel_)

print(len(ent1))
print(len(ent2))
print(len(rels))

df = pd.DataFrame()
df['ent1'] = ent1
df['ent2'] = ent2
df['rels'] = rels



kg1 = {}
with open(items_tsv_path+"/items.tsv") as f:
    for line in f:
       (key, val) = line.split("\t")
       kg1[(key)] = val
       val = kg1[(key)].split("\n")[-2]
       kg1[(key)] = (val)

ent1_new_name = []
for i in ent1:
    ent1_new_name.append(kg1[i])
print(len(ent1))
print(len(ent1_new_name))


new = pd.DataFrame()
new['ent1'] = ent1_new_name
new['rels'] = rels
new['ent2'] = ent2

new.to_csv(path+"/newKG.tsv", header=None, index=None, sep='\t')


kg_ent = {}
ent1 = []
rels = []
ent2 = []

with open(path+"/newKG.tsv") as f:
    for line in f:
        (key, rel_, val2) = line.split("\t")
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


## make ent_ids_1
all_item_ = []
all_item = []
ids = []
count = 0
for i in set(ent1):
    if i != "":
        all_item_.append(i)
print(len(all_item_))
for i in set(ent2):
    if i != "":
        all_item_.append(i)
print(len(all_item_))

print("======")
print(len(set(all_item_)))
for i in set(all_item_):
    if i != "":
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
print(len(ent_ids_1))

ent_ids_1.to_csv(path+"/ent_ids_1", header=None, index=None, sep='\t')


## make rel_ids
all_item = []
ids = []
count = 0
for i in set(rels):
    all_item.append(i)
    ids.append(count)
    count+=1
print(len(all_item))


rel_ids = pd.DataFrame()
rel_ids['id'] = ids
rel_ids['ent_name'] = all_item

rel_ids.to_csv(path+"/rel_ids", header=None, index=None, sep='\t')


rel_dict = {}
for i in range(len(rel_ids)):
    rel_dict[rel_ids['ent_name'][i]] = rel_ids['id'][i]
rel_dict


ent_dict = {}
for i in range(len(ent_ids_1)):
    ent_dict[ent_ids_1['ent_name'][i]] = ent_ids_1['id'][i]
print(len(ent_dict))



path = "./KK_new/triplets/"+folder

kg_ent = {}
ent1 = []
rels = []
ent2 = []

with open(path+"/newKG.tsv") as f:
    for line in f:
        (key, rel_, val2) = line.split("\t")
        kg_ent[(key)] = val2
        val = kg_ent[(key)].split("\n")[-2]
        kg_ent[(key)] = (rel_,val)
        if key and val != "":
            ent1.append(ent_dict[key])
            ent2.append(ent_dict[val])
            rels.append(rel_dict[rel_])

print(len(ent1))
print(len(ent2))
print(len(rels))

new = pd.DataFrame()
new['ent1'] = ent1
new['rels'] = rels
new['ent2'] = ent2

new.to_csv(path+"/triples_1", header=None, index=None, sep='\t')