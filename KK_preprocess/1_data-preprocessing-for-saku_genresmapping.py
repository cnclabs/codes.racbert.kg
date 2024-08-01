import numpy as np
import pandas as pd
import csv

folder = "0000_0003_20221123_0000_genre_test"
path = "./KK_new/triplets/"+folder
items_tsv_path = "KK_new"
genres_mapping_path = "./KK_new/genres_mapping.tsv"

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


genre_dict = {}

saku_genre_list = []
fill_genre_list = []

with open(genres_mapping_path) as f:
    for line in f:
        (key, val) = line.split("\t")
        val_ = val.split("[")[-1].split("]")[0].split(",")
        for i in val_:
            i_ = i.split(":")[-1]
            genre_dict[str(i_)] = str(key)

ent2_new_name = []
for i in ent2:
    if i in list(genre_dict.keys()):
        ent2_new_name.append(genre_dict[i])
    else:
        ent2_new_name.append(i)

new = pd.DataFrame()
new['ent1'] = ent1_new_name
new['rels'] = rels
new['ent2'] = ent2_new_name

new.to_csv(path+"/newKG.tsv", header=None, index=None, sep='\t')