# Automatic Query Rewriter

Rewrite slow SQL (over 1 second) in logic level, which explores proper rewrite rule sequences based on the query characters. It includes five main components, i.e., query formatting, SQL2RelationalAlegebra, rewrite rule searching,
query rewriting, and RelationalAlegebra2SQL.

## Before you run

You need to configure `java` and `maven` environment properly.

Install dependencies.

```bash
pip install -r requirements.txt
```

You need to rename file `config_template.py` to `config.py` and set user, host, password and database in the file.

## How to run

1. Support two types of input queries:

- Save your `k` SQLs as k documents: ./queries/input_sql/1.sql - ./queries/input_sql/k.sql 


- Directly run the benchmark queries and nothing to do here 

2. Run SQL rewrite

    ```bash
    python3 test_rewriter.py [--workload tpch,user]
    ```

- --workload: `tpch` represents running tpch queries; `user` represents running SQL queries assigned by you.    

    For example:

    ```bash
    python3 test_rewriter.py --workload tpch
    ```


3. Check the rewrite running results within the directory ./rewrite_results/


## TODO

- Fix problems in monte carlo tree searching.

- Support more rewrite strategies

- Support complex SQL syntax.


### Contact
If you have any issue, feel free to post on [Project](https://gitee.com/opengauss-tsinghua/openGauss-server/tree/master/src/gausskernel/dbmind/kernel/sql_rewriter).
