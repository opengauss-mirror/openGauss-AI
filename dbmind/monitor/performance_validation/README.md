# Automatic Performance Validation

The performance validation model is to predict query performance (e.g., latencies of all the operators) before actual workload execution using deep graph embedding model.  

## Before you run

Install dependencies.

```bash
pip install -r requirements.txt
```

You need to rename file `config_template.py` to `config.py` and set user, host, password and database in the file.

## How to run

1. (optional) Add user's data: 

- Place queries within the directory `data/query_plan/` (json format). And then execute the following commend to convert into graph data: 

```bash
python3 generate_dataset.py
```

2. Train the validation model:

```bash
python3 train.py
```


## TODO

- Add query execution relations

- Add inference module

- Finer-granularity batch training


### Contact
If you have any issue, feel free to post on [Project](https://gitee.com/opengauss-tsinghua/openGauss-server/tree/master/src/gausskernel/dbmind/).
