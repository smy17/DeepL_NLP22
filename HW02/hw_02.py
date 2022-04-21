import random
import numpy as np
import matplotlib.pyplot as plt


def judge_coin_kind(rand):
    if 0 <= rand < num_prob[0] * 100:
        return 0
    elif num_prob[0] * 100 <= rand < (num_prob[1] + num_prob[0]) * 100:
        return 1
    else:
        return 2


def judge_coin_pos_or_neg(kind, rand):
    if 0 <= rand < pos_pron[kind] * 100:
        return 1
    else:
        return 0


num_prob = [0.4, 0.4, 0.2]
pos_pron = [0.3, 0.3, 0.8]
N = 1000  # 硬币的个数
M = 100  # 每个硬币投掷的次数
results = []  # 记录抛掷的结果，M列N行
epochs = 50  # 迭代轮数
init_parameters = [0.1, 0.7, 0.1, 0.1, 0.1]  # 种类一二比例、种类一二三为正概率
process_parameters = [init_parameters]
for coin in range(N):
    #  第n个硬币
    result = []
    rand_coin = random.randint(0, 99)
    coin_kind = judge_coin_kind(rand_coin)  # 返回当前硬币的种类
    for throw in range(M):
        #  第n个硬币投掷M次
        rand_result = random.randint(0, 99)
        pos_or_neg = judge_coin_pos_or_neg(coin_kind, rand_result)  # 返回硬币的正反面
        result.append(pos_or_neg)
    results.append(result)
# 开始迭代
for epoch in range(epochs):
    u1_num=np.power(init_parameters[2],np.sum(results,1))* np.power(1 - init_parameters[2],M - np.sum(results, 1)) * init_parameters[0]
    den = np.power(init_parameters[2], np.sum(results, 1)) * np.power(1 - init_parameters[2],M - np.sum(results, 1)) * init_parameters[0] + np.power(init_parameters[3], np.sum(results, 1)) * np.power(1-init_parameters[3],M-np.sum(results,1))*init_parameters[1]+np.power(init_parameters[4], np.sum(results, 1)) * np.power(1 - init_parameters[4], M - np.sum(results, 1)) * (1 - init_parameters[0] - init_parameters[1])
    u2_num=np.power(init_parameters[3], np.sum(results, 1)) * np.power(1 - init_parameters[3],
M - np.sum(results, 1)) * init_parameters[1]
    u1 = u1_num / den
    u2 = u2_num / den
    u3 = 1 - u1 - u2

    init_parameters[0] = sum(u1) / N
    init_parameters[1] = sum(u2) / N
    init_parameters[2] = sum(u1 * np.sum(results, 1)) / sum(u1 * M)
    init_parameters[3] = sum(u2 * np.sum(results, 1)) / sum(u2 * M)
    init_parameters[4] = sum(u3 * np.sum(results, 1)) / sum(u3 * M)
    process_parameters.append(init_parameters)
x = np.linspace(0, epochs, epochs + 1)
process_parameters = np.array(process_parameters).T
print(process_parameters)
