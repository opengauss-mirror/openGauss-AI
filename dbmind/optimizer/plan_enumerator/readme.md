# Reinforcement Learning with Tree-LSTM for Join Order Selection

Reinforcement Learning with Tree-LSTM for Join Order Selection(RTOS) is an optimizer which focuses on Join order Selection(JOS) problem.  RTOS learns from previous queries to build plan for further queries with the help of DQN and TreeLSTM.

## Important parameters

Here we have listed the most important parameters you need to configure to run RTOS on a new database. 

- schemafile
  - <a href ="https://github.com/gregrahn/join-order-benchmark/blob/master/schema.sql"> a sample</a>
- sytheticDir
  - Directory contains the sytheic workload(queries), more sythetic queries will help RTOS make better plans. 
  - It is nesscery when you want apply RTOS for a new database.  
- JOBDir
  - Directory contains all JOB queries. 
- Configuration of openGauss
  - dbName : the name of database 
  - userName : the user name of database
  - password : your password of userName
  - ip : ip address of openGauss
  - port : port of openGauss

## Getting Started

Install python dependencies.

```bash
pip install -r requirements.txt
```

Configure database parameters in `ImportantConfig.py` file.

Cost Training. It will store the model which optimize on cost in openGauss.

```bash
python3 CostTraining.py
```

Latency Tuning. It will load the cost training model and generate the latency training model.
**You need to change `self.isCostTraining` to False in `ImportantConfig.py` file before you run the following commands!**

```bash
python3 LatencyTuning.py 
```
