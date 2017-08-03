import numpy as np
import matplotlib as plt
import matplotlib.pyplot as pp

data_file = "Assignment_1_Data_and_Template.xlsx"


def read_training_data():
    from pandas import read_excel
    values = (read_excel(data_file, "Data", header=None)).values
    return values[1:16701, 0:3]


def process_data(raw_data):
    d = raw_data[:, 2]
    d[d == 'Male'] = 1
    d[d == 'Female'] = -1
    # print(d)
    processed_data = np.array(np.column_stack((raw_data[:, 0] * 12 + raw_data[:, 1], d)), dtype=int)
    # print(processed_data)
    return processed_data


def apply_stump_algo(processed_data):
    n = processed_data[:, 0].size
    unique, counts = np.unique(processed_data[:, 1], return_counts=True)
    # print(unique, counts)
    tp = counts[1]
    tn = counts[0]
    print(n, tp, tn)
    an = 0
    ap = 0
    i0 = iopt = (tn * tp) / (n * n)
    print("IO", iopt, i0)
    #sorted_data = np.sort(processed_data.view('i8,i8'), order=['f1'], axis=0).view(np.int)
    sorted_data = processed_data
    # print(sorted_data)
    tao = sorted_data[0, 0]
    for i in range(2, n):
        print(sorted_data[i])
        if sorted_data[i-1, 1] == -1:
            an = an + 1
        else:
            ap = ap + 1
        # print(an, ap, tn, tp)
        I = 1 / n * ((an * ap / (an + ap)) + ((tn - an) * (tp - ap) / (tn + tp - an - ap)))
        if I < iopt:
            iopt = I
            tao = sorted_data[i, 0]
        # print(i, I, tao)
    delta = i0 - iopt
    return delta, tao


def cal_linear_classifier(processed_data):
    xa = np.insert(processed_data[:, [0]], 0, 1, axis=1)
    t = processed_data[:, [1]]
    # print(xa, t)
    xat = np.linalg.pinv(xa)
    # print(xat)
    w = np.dot(xat, t)
    return w


def main():
    raw_data = np.array(read_training_data())
    # print(raw_data.shape)
    # print(raw_data)
    processed_data = process_data(raw_data)
    # pp.plot(processed_data[raw_data[:, 2] == -1][:,1], processed_data[raw_data[:, 2] == -1][:,0], marker='.',
    #         color="g")
    # pp.plot(processed_data[raw_data[:, 2] == 1][:,1], processed_data[raw_data[:, 2] == 1][:,0], marker='x',
    #         color="r")
    # plt.pyplot.savefig('scatter-plot.png')
    delta, tao = apply_stump_algo(processed_data)
    print("delta & tao", delta, tao)
    w = cal_linear_classifier(processed_data)
    print("w", w)
    output = w[0] + w[1]*tao
    print("output", output)


main()