from scipy import stats
import math
import pickle
from matplotlib import pyplot as plt
import numpy as np
import os


def get_field_num(read_f = './pack/0_30_50_res_final.pickle'):
    with open(read_f, 'rb') as f:
        a,_ = pickle.load(f)
        return len(a)

def density_series(input_time_series, window_size, stride):
    if stride > window_size:
        print('warning! stride is larger than window size!')
    
    res_list = []
    for beg in range(0, len(input_time_series) - 2*window_size, stride):
        s1 = beg
        s2 = beg + window_size
        # print('s1 =', s1, ' s2 =', s2)
        res = stats.ks_2samp(input_time_series[s1:s1 + window_size], input_time_series[s2:s2 + window_size])
        res_list.append(res.statistic)

    return res_list

def candidate_region():
    pass

rate_range = (70, 250, 40)
start_range = (40, 65, 5)


'''
利用K-S Test提炼异常指标

Args:
    mode: 异常指标数据类型，可选布尔型和浮点型
    resfile_path: 结果文件的路径
    anomaly_threshold: 异常判断的阈值
    fields_num: 时序数据的维度
    
Returns:
    None

Raises:
    暂无
'''
def anomaly_extract(mode='bool', resfile_path = './res_data.pickle', \
    anomaly_threshold = 0.5, fields_num = 120, window_size = 15, stride = 3):
    with open(resfile_path, 'rb') as f_res_data:

        aliyun_res = pickle.load(f_res_data)
        cause_list = ["cpu_saturation","io_saturation","database_backup",   # 问题根因列表
        "table_restore","poorly_physical_design","poorly_written_query",
        "workload_spike","vacuum_analyze","lock_contention",
        "network_congestion"]

        dataset = []                                                        # 结果数据集初始化
        anomaly_vec = [0 for _ in range(fields_num)]

        for causes in aliyun_res.keys():
            print()
            print('i = ',causes,' The root cause is', cause_list[causes])

            for rate in aliyun_res[causes].keys():
                for start in aliyun_res[causes][rate].keys():
                    res_lis = []
                    for k in range(fields_num):
                        example_ts, anomaly_region = (aliyun_res[causes][rate][start][k])
                        # split_line = (aliyun_res[causes][rate][start][k][1][0])
                        split_line = anomaly_region[0]
                        s = max(0,split_line - 30)
                        e = min(split_line+30, anomaly_region[-1], len(example_ts))
                        y_data = np.array(density_series(example_ts[s:e], window_size, stride))
                        res = np.max(y_data) - np.min(y_data)

                        if res>anomaly_threshold:
                            if mode=='bool':
                                res_lis.append(1)
                            elif mode=='numerical':
                                res_lis.append(res)
                            anomaly_vec[k]+=1
                        else:
                            res_lis.append(0)
                    res_lis.append(causes)
                    dataset.append(res_lis)
        print(anomaly_vec)
        return dataset

if __name__ == "__main__":
    dataset_bool = np.array(anomaly_extract(mode='bool'))    
    dataset_numerical = np.array(anomaly_extract(mode='numerical'))

    # print(dataset.shape)
    with open('./new_dataset_bool.pickle', 'wb+') as f:
        pickle.dump(dataset_bool, f)

    with open('./new_dataset_numerical.pickle', 'wb+') as f:
        pickle.dump(dataset_numerical, f)

    # get_field_num()

# def test():
#     with open('exp_data.pickle', 'rb') as f_exp_data:
#         test = pickle.load(f_exp_data)
#         # print(test.causes)
#         i = 6
#         for i in range(10):
#             print()
#             print('i = ',i,' The root cause is', test.causes[i])

#             for j in range(11):
#                 print('j = ',j,end=' ')
#                 cnt = 0
#                 res_lis = []
#                 for k in range(94):
#                     example_ts = (test.test_datasets[i][j][:,k])
#                     split_line = (test.abnormal_regions[i][j][0])
#                 # a = [math.sin(i) for i in range(100)]
#                 # x_data = range(15,len(example_ts) - 15, 5)
#                     start = max(0,split_line - 30)
#                     y_data = np.array(density_series(example_ts[start:split_line+30],15,3))
#                     res = np.max(y_data) - np.min(y_data)
#                     res_lis.append(res)
#                     if res>0.5:
#                         print(test.field_names[k],end=' ')
#                         cnt += 1
#             # print(,end=' ')
#                 print('cnt =', cnt)
#     # plt.plot(y_data)
#     # plt.show()


# def previous_anomaly_extract(mode='bool', resfile_path = './res_data.pickle', anomaly_threshold = 0.5, fields_num = 120):
#     with open(resfile_path, 'rb') as f_res_data:

#         aliyun_res = pickle.load(f_res_data)
#         cause_list = ["cpu_saturation","io_saturation","database_backup",
#         "table_restore","poorly_physical_design","poorly_written_query",
#         "workload_spike","vacuum_analyze","lock_contention",
#         "network_congestion"]
        
#         # cause_list = ["cpu_saturation","io_saturation","database_backup",
#         # "table_restore","poorly_physical_design","poorly_written_query",
#         # "workload_spike","flush_log","vacuum_analyze","lock_contention",
#         # "network_congestion"]

#         field_vec0 = [[] for i in range(fields_num)]
#         field_vec1 = [[] for i in range(fields_num)]
#         dataset = []
#         anomaly_vec = [0 for _ in range(fields_num)]

#         for causes in range(10):
#             print()
#             print('i = ',causes,' The root cause is', cause_list[causes])

#             for rate in range(*rate_range):
#                 for start in range(*start_range):
#                     cnt = 0
#                     res_lis = []
#                     for k in range(fields_num):
#                         example_ts, anomaly_region = (aliyun_res[causes][rate][start][k])
#                         # split_line = (aliyun_res[causes][rate][start][k][1][0])
#                         split_line = anomaly_region[0]
#                         s = max(0,split_line - 30)
#                         y_data = np.array(density_series(example_ts[s:split_line+30],15,3))
#                         res = np.max(y_data) - np.min(y_data)
#                         if causes==0:
#                             field_vec0[k].append(res)
#                         else:
#                             field_vec1[k].append(res)

#                         if res>0.5:
#                             if mode=='bool':
#                                 res_lis.append(1)
#                             elif mode=='numerical':
#                                 res_lis.append(res)
#                             anomaly_vec[k]+=1
#                             cnt += 1
#                         else:
#                             res_lis.append(0)
#                     res_lis.append(causes)
#                     dataset.append(res_lis)
#         print(anomaly_vec)
#         return dataset