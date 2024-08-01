import numpy as np
import pandas as pd
import csv

folder = "0000_0003_20221123_0000_genre_test"
path = "./KK_new/triplets/"+folder
items_tsv_path = "./KK_new"


tvvod = {}
id_tvvod = []
ent_tvvod = []
with open(items_tsv_path+"/items.tsv") as f:
    for line in f:
        (key, val_) = line.split("\t")
        val = val_.split("\n")[-2]
        tvvod[str(val)] = key
        id_tvvod.append(key)
        ent_tvvod.append(val)

print(len(ent_tvvod))
print(len(set(ent_tvvod)))
print(len(tvvod))


kg_ent_saku = {}
id_saku = []
ent_saku = []

with open(path+"/ent_ids_1") as f:
    for line in f:
        (key, val_) = line.split("\t")
        val = val_.split("\n")[-2]
        kg_ent_saku[str(val)] = key
        id_saku.append(key)
        ent_saku.append(val)

print(len(id_saku))
print(len(ent_saku))
print(len(set(id_saku)))
print(len(set(ent_saku)))

a = set(ent_saku)
b = set(ent_tvvod)

print(len(a))
print(len(b))
c = list(set(a) & set(b))
print(len(c))



kg_ent_saku_ = {}
id_saku_ = []
ent_saku_ = []

for i in c:
    kg_ent_saku_[str(i)] = kg_ent_saku[i]
    id_saku_.append(kg_ent_saku_[str(i)])
    ent_saku_.append(i)#.split("\n")[-2])

print(len(id_saku_))
print(len(ent_saku_))
print(len(set(id_saku_)))
print(len(set(ent_saku_)))




kg_ent_filmarks = {}
id_filmarks = []
ent_filmarks = []

with open(path+"/ent_ids_2") as f:
    for line in f:
        (key, val_) = line.split("\t")
        val = val_.split("\n")[-2]
        kg_ent_filmarks[str(val)] = key
        id_filmarks.append(key)
        ent_filmarks.append(val)

print(len(id_filmarks))
print(len(ent_filmarks))
print(len(set(id_filmarks)))
print(len(set(ent_filmarks)))

a = set(ent_filmarks)
b = set(ent_tvvod)

print(len(a))
print(len(b))
c = list(set(a) & set(b))
print(len(c))


kg_ent_filmarks_ = {}
id_filmarks_ = []
ent_filmarks_ = []

for i in c:
    kg_ent_filmarks_[str(i)] = kg_ent_filmarks[i]
    id_filmarks_.append(kg_ent_filmarks_[str(i)])
    ent_filmarks_.append(i)

print(len(id_filmarks_))
print(len(ent_filmarks_))
print(len(set(id_filmarks_)))
print(len(set(ent_filmarks_)))


a = set(ent_saku_)
b = set(ent_filmarks_)

c = list(set(a) & set(b))
print(len(c))


saku_id = []
filmark_id = []

for i in c:
    saku_id.append(kg_ent_saku[i])
    filmark_id.append(kg_ent_filmarks[i])
print(len(saku_id))
print(len(filmark_id))

new = pd.DataFrame()
new['ent1'] = saku_id
new['ent2'] = filmark_id

new.to_csv(path+"/ill_ent_ids_", header=None, index=None, sep='\t')





#################


tvvod = {}
id_tvvod = []
ent_tvvod = []
with open(items_tsv_path+"/items.tsv") as f:
    for line in f:
        (key, val_) = line.split("\t")
        val = val_.split("\n")[-2]
        tvvod[str(val)] = key
        id_tvvod.append(key)
        ent_tvvod.append(val)

filmark = {}
id_filmark = []
ent_filmark = []
with open("filmarks_kg_RAC.tsv") as f:
    for line in f:
        (key, val, val_) = line.split("\t")
        val = val_.split("\n")[-2]
        filmark[str(key)] = key
        id_filmark.append(key)
        ent_filmark.append(key)

a = set(ent_tvvod)
b = set(ent_filmark)
c_ = list(set(a) & set(b))
print(len(c_))

kg_ent_saku = {}
id_saku = []
ent_saku = []

with open(path+"/ent_ids_1") as f:
    for line in f:
        (key, val_) = line.split("\t")
        val = val_.split("\n")[-2]
        kg_ent_saku[str(val)] = key
        id_saku.append(key)
        ent_saku.append(val)

print(len(id_saku))
print(len(ent_saku))
print(len(set(id_saku)))
print(len(set(ent_saku)))


a = set(ent_saku)
b = set(c_)

print(len(a))
print(len(b))
c = list(set(a) & set(b))
print(len(c))




kg_ent_saku_ = {}
id_saku_ = []
ent_saku_ = []

for i in c:
    kg_ent_saku_[str(i)] = kg_ent_saku[i]
    id_saku_.append(kg_ent_saku_[str(i)])
    ent_saku_.append(i)

print(len(id_saku_))
print(len(ent_saku_))
print(len(set(id_saku_)))
print(len(set(ent_saku_)))




kg_ent_filmarks = {}
id_filmarks = []
ent_filmarks_ = []

with open(path+"/ent_ids_2") as f:
    for line in f:
        (key, val_) = line.split("\t")
        val = val_.split("\n")[-2]
        kg_ent_filmarks[str(val)] = key
        id_filmarks.append(key)
        ent_filmarks_.append(val)

print(len(id_filmarks))
print(len(ent_filmarks_))
print(len(set(id_filmarks)))
print(len(set(ent_filmarks_)))


a = set(ent_filmarks_)
b = set(c_)

print(len(a))
print(len(b))
c = list(set(a) & set(b))
print(len(c))


kg_ent_filmarks_ = {}
id_filmarks_ = []
ent_filmarks_ = []

for i in c:
    kg_ent_filmarks_[str(i)] = kg_ent_filmarks[i]
    id_filmarks_.append(kg_ent_filmarks_[str(i)])
    ent_filmarks_.append(i)

print(len(id_filmarks_))
print(len(ent_filmarks_))
print(len(set(id_filmarks_)))
print(len(set(ent_filmarks_)))


a = set(ent_saku_)
b = set(ent_filmarks_)

c = list(set(a) & set(b))
print(len(c))


saku_id = []
filmark_id = []

for i in c:
    saku_id.append(kg_ent_saku[i])
    filmark_id.append(kg_ent_filmarks[i])
print(len(saku_id))
print(len(filmark_id))


new = pd.DataFrame()
new['ent1'] = saku_id
new['ent2'] = filmark_id

new.to_csv(path+"/ill_ent_ids", header=None, index=None, sep='\t')