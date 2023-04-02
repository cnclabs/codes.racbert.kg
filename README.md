# RAC
Source code for CIKM 2021 paper Reinforced Active Entity Alignment

Codes based on [EAkit](https://github.com/THU-KEG/EAkit). Thanks for their contribution! 

## Dependencies

Please refer to [EAkit](https://github.com/THU-KEG/EAkit).

## Datasets
There are eight datasets in this folder:
- zh_en/ja_en/fr_en from the [DBP15K dataset](https://github.com/nju-websoft/BootEA)
- en_fr/en_de/dbp_wd/dbp_yg from the [SRPRS dataset](https://github.com/nju-websoft/RSN)
- [DBP-FB dataset](https://github.com/DexterZeng/EAE)

Please unzip them in the `data/` directory.

Take the dataset DBP15K (ZH-EN) as an example, it contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_2: ids for entities in target KG (EN);
* ill_ent_ids: entity links encoded by ids;
* ref_ent_ids: entity links for testing/validation;
* sup_ent_ids: entity links for training;
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);

### Query strategy scores
We have already provided the scores of the degree and Pagerank metrics in the datasets. The codes for generating such scores are also provided in the `data/` directory.


## Running
* Run
```
bash run_rl.sh
```

> Change the parameters in `run_rl.sh` to obtain the results under other settings.

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit  when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to zengweixin13@nudt.edu.cn.


