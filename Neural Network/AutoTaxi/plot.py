import matplotlib.pyplot as plt
import numpy as np

with open("AutoTaxi_64Relus_200Epochs_OneOutput.nnet_log.txt", "r") as f:
    line = f.readline()
    eps = [0.5, 0.1, 0.15]
    dic_1 = []
    while line.startswith("{"):
        line = f.readline()
        if line.startswith("{"):
            eval_line = eval(line)
            dic_1.append(eval_line)
    res_pri = []
    res_dual = []
    res_plus = []
    time_pri = []
    time_dual= []
    time_plus = []
    for i in dic_1:
        res_pri.append(i['Primal'])
        time_pri.append(i["Primal_time"])
        res_dual.append(i['Dual'])
        time_dual.append(i["Dual_time"])
        res_plus.append(i['res_plus'])
        time_plus.append(i['res_plus_time'])

    plt.plot([i for i in range(len(res_pri))], res_pri)
    plt.plot([i for i in range(len(res_pri))], res_dual)
    plt.plot([i for i in range(len(res_pri))], res_plus)
    plt.title("Comparision of three methods with epsilon = 0.04")
    plt.legend(["Primal Methods", "Dual Methods", "Deeplus Methods"])
    print(sum(time_pri)/ len(time_pri))
    print(sum(time_dual) / len(time_dual))
    print(sum(time_plus) / len(time_plus))
    plt.show()

    dic_2 = []

    for i in range(3):
        f.readline()
    line = f.readline()
    while line.startswith("{"):
        line = f.readline()
        if line.startswith("{"):
            eval_line = eval(line)
            dic_2.append(eval_line)
    res_pri = []
    res_dual = []
    res_plus = []
    time_pri = []
    time_dual = []
    time_plus = []
    for i in dic_2:
        res_pri.append(i['Primal'])
        time_pri.append(i["Primal_time"])
        res_dual.append(i['Dual'])
        time_dual.append(i["Dual_time"])
        res_plus.append(i['res_plus'])
        time_plus.append(i['res_plus_time'])
    plt.plot([i for i in range(len(res_pri))], res_pri)
    plt.plot([i for i in range(len(res_pri))], res_dual)
    plt.plot([i for i in range(len(res_pri))], res_plus)
    plt.title("Comparision of three methods with epsilon = 0.08")
    print(sum(time_pri)/ len(time_pri))
    print(sum(time_dual) / len(time_dual))
    print(sum(time_plus) / len(time_plus))
    plt.legend(["Primal Methods", "Dual Methods", "Deeplus Methods"])
    plt.show()

    dic_3 = []

    for i in range(3):
        f.readline()
    line = f.readline()
    while line.startswith("{"):
        line = f.readline()
        if line.startswith("{"):
            eval_line = eval(line)
            dic_3.append(eval_line)
    res_pri = []
    res_dual = []
    res_plus = []
    for i in dic_3:
        res_pri.append(i['Primal'])
        res_dual.append(i['Dual'])
        res_plus.append(i['res_plus'])
    plt.plot([i for i in range(len(res_pri))], res_pri)
    plt.plot([i for i in range(len(res_pri))], res_dual)
    plt.plot([i for i in range(len(res_pri))], res_plus)
    plt.legend(["Primal Methods", "Dual Methods", "Deeplus Methods"])

    plt.show()

