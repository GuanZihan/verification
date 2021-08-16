import matplotlib.pyplot as plt


def avg(times):
    return sum(times) / len(times)
advs = []
ibps = []
fo_count = 0
pgd_count = 0
with open("../log/test.txt", "r") as f:
    all = {}
    num = 100
    for i in range(num):
        line = eval(f.readline())
        all[i] = line
    for key, val in all.items():
        advs.append(val["adv_lb"])
        ibps.append((val["ibp_bound"]))
        if val["verified_ub"] < 0.0:
            fo_count += 1
        if val["adv_lb"] < 0:
            pgd_count += 1



with open("../log/5.txt", "r") as f:
    all = {}
    num = 100
    targets = []
    for i in range(num):
        line = eval(f.readline())
        all[i] = line
    primal_times = []
    primal_bounds = []
    plus_bounds = []

    for key, val in all.items():
        primal_times.append(val["Primal_time"])
        primal_bounds.append(val["Primal"])
    # dual_times = []
    # for key, val in all.items():
    #     dual_times.append(val["Dual_time"])
    plus_times = []
    for key, val in all.items():
        plus_times.append(val["res_plus_time"])
        targets.append(val["target"])
        plus_bounds.append(val["res_plus"])
    print(targets)

    primal_count = 0
    dual_count = 0
    plus_count = 0

    for key, val in all.items():
        if val["status_primal"] == 1.0:
            primal_count += 1

    for key, val in all.items():
        if val["status_dual"] == 1.0:
            dual_count += 1

    for key, val in all.items():
        if val["status_res_plus"] == 1.0:
            plus_count += 1


    print(primal_count / num)
    print(dual_count / num)
    print(plus_count / num)
    print("sdp-fo: ", fo_count / num)
    print("pgd: ", pgd_count / num)
    fo_count = 0
    fo_times = []
    # with open("sdpfo_log.txt", "r") as f2:
    #     sdp_all = {}
    #     num_fo = 72
    #     for i in range(num_fo):
    #         line = eval(f2.readline())
    #         sdp_all[i] = line
    #
    #     for key, val in sdp_all.items():
    #         if val["verified_ub"] < 0:
    #             fo_count += 1
    #         fo_times.append(val["time"])
    #     print(1 - fo_count / num_fo)


    X = [i for i in range(num)]
    plt.plot(X, primal_times)
    # plt.plot(X, dual_times)
    plt.plot(X, plus_times)


    # plt.plot(X, fo_times)
    plt.legend(["Primal_time", "Dual_time", "Plus_time", "SDP-FO"])
    print(avg(primal_times))
    # print(avg(dual_times))
    print(avg(plus_times))
    # print(avg(fo_times))

    plt.show()

    # plt.xlim(-8, 2)
    # plt.ylim(-8, 20)
    plt.scatter(advs, primal_bounds, marker="v", c="green")
    # plt.scatter(advs, ibps)
    plt.scatter(advs, plus_bounds, marker="o", c="orange")
    plt.hlines(0, -8, 2, colors="grey", linestyles="dotted")
    plt.vlines(0, -8, 4, colors="grey", linestyles="dotted")
    print(advs)
    print(primal_bounds)
    plt.plot(advs, advs, c="black")
    plt.plot()
    plt.legend(["y=x adv_lb","primal_sdp", "deeplus"])
    plt.title("PGD_NN eps=0.2 Primal_sdp and Deeplus")
    plt.show()