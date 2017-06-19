import numpy as np  # using numpy for various calculations
import matplotlib.pyplot as plt
import pandas

data_file = "Assignment_2_Data_and_Template.xlsx"  # base file to load data from
query_data = np.array([[69, 17.5], [66, 22], [70, 21.5], [69, 23.5]], dtype=np.float32)

female_hist_meta = []
male_hist_meta = []


def read_excel_file():
    from pandas import read_excel
    excel_workbook = read_excel(data_file)
    sample_data = np.array(excel_workbook.values)
    # print(sample_data)
    female_rows = np.where(sample_data[:, 0] == 'Female')
    female_data = sample_data[female_rows]
    # print(female_data)
    # print(female_data.size)
    male_rows = np.where(sample_data[:, 0] == 'Male')
    male_data = sample_data[male_rows]
    # print(male_data)
    # print(male_data.size)
    bin_size = find_bin_size(sample_data.size)
    # print(bin_size)
    hist_meta = []
    max_values = np.amax(sample_data, axis=0)
    min_values = np.amin(sample_data, axis=0)
    print(max_values)
    print(min_values)

    h_slot = (max_values[1] - min_values[1]) / bin_size
    s_slot = (max_values[2] - min_values[2]) / bin_size
    hist_meta.append(min_values[1])
    hist_meta.append(min_values[2])
    hist_meta.append(h_slot)
    hist_meta.append(s_slot)
    hist_meta.append(max_values[1])
    hist_meta.append(max_values[2])

    #print(hist_meta)

    female_hist = create_hist(female_data, bin_size, hist_meta)
    male_hist = create_hist(male_data, bin_size, hist_meta)

    # female_hist_rec = create_hist(female_data, bin_size, female_hist_meta)
    # male_hist_rec = create_hist(male_data, bin_size, male_hist_meta)
    bayesian_meta = np.empty(5, dtype=object)
    create_bayesian_data(male_data, female_data, bayesian_meta)

    query_result_hist(male_hist, female_hist, [69, 17.5], hist_meta)
    query_result_hist(male_hist, female_hist, [66, 22], hist_meta)
    query_result_hist(male_hist, female_hist, [70, 21.5], hist_meta)
    query_result_hist(male_hist, female_hist, [69, 23.5], hist_meta)

    query_result_bayesian([69, 17.5], bayesian_meta)
    query_result_bayesian([66, 22], bayesian_meta)
    query_result_bayesian([70, 21.5], bayesian_meta)
    query_result_bayesian([69, 23.5], bayesian_meta)


def query_result_hist(male_hist, female_hist, query, hist_meta):
    # print("hist_meta", hist_meta)
    print("query: ", query)
    i = round((query[0] - hist_meta[0]) / hist_meta[2])
    j = round((query[1] - hist_meta[1]) / hist_meta[3])
    print("i & j", i, j)
    female_prob = female_hist[i][j] / (female_hist[i][j] + male_hist[i][j])
    print ("female probability : ", female_prob)


def query_result_bayesian(x, bayesian_meta):
    #print(bayesian_meta)
    print("query: ", x)
    nf = bayesian_meta[0][0]
    nm = bayesian_meta[0][1]
    female_mean = np.array(bayesian_meta[1], dtype=np.float32)
    male_mean = np.array(bayesian_meta[2], dtype=np.float32)
    female_cov = np.matrix(bayesian_meta[3])
    male_cov = bayesian_meta[4]

    female_prob = np.divide(calc_gaussian_pd(x, female_mean, female_cov, nf),
                    np.add(calc_gaussian_pd(x, female_mean, female_cov, nf),
                           calc_gaussian_pd(x, male_mean, male_cov, nm)))

    print("female probability : ", female_prob[0][0])



def calc_gaussian_pd(x, mean, cov, count):
    return np.multiply(count, np.multiply(1 / (2 * np.pi * (np.sqrt(np.linalg.det(cov)))), np.exp(np.multiply(-1 / 2,
                                                                                                              np.dot(
                                                                                                                  np.dot(
                                                                                                                      np.subtract(
                                                                                                                          x,
                                                                                                                          mean),
                                                                                                                      np.linalg.inv(
                                                                                                                          cov)),
                                                                                                                  np.transpose(
                                                                                                                      np.subtract(
                                                                                                                          x,
                                                                                                                          mean)))))))


def create_bayesian_data(male_data, female_data, bayesian_meta):
    # male_sample_size = male_data[:,0].size
    # female_sample_size = female_data[:,0].size
    bayesian_meta[0] = np.array([female_data[:, 0].size, male_data[:, 0].size])
    bayesian_meta[1] = np.mean(female_data[:, 1:3], axis=0)
    bayesian_meta[2] = np.mean(male_data[:, 1:3], axis=0)
    bayesian_meta[3] = np.cov(np.array(female_data[:, 1:3], dtype=np.float32), rowvar=False).round(decimals=2)
    bayesian_meta[4] = np.cov(np.array(male_data[:, 1:3], dtype=np.float32), rowvar=False).round(decimals=2)
    print(bayesian_meta)


def create_hist(input_data, bin_size, hist_meta):
    if (len(hist_meta) == 0):
        # print(bin_size)
        max_values = np.amax(input_data, axis=0);
        min_values = np.amin(input_data, axis=0);
        h_max = max_values[1]
        h_min = min_values[1]
        s_max = max_values[2]
        s_min = min_values[2]
        # print(max_values)
        # print(min_values)
        h_slot = (max_values[1] - min_values[1]) / bin_size
        s_slot = (max_values[2] - min_values[2]) / bin_size
        # print(h_slot)
        # print(s_slot)
    else:
        h_min = hist_meta[0]
        s_min = hist_meta[1]
        h_slot = hist_meta[2]
        s_slot = hist_meta[3]
        h_max = hist_meta[4]
        s_max = hist_meta[5]

    i = 0
    hist = np.zeros((bin_size, bin_size))
    # print(hist)
    for h in np.arange(h_min, h_max, h_slot):
        j = 0
        for s in np.arange(s_min, s_max, s_slot):
            # print("h and s ", h, s)
            temp_data = input_data[(np.where(np.logical_and(h <= input_data[:, 1], input_data[:, 1] < h + h_slot)))]
            # print(temp_data)
            # print(temp_data[(np.where(np.logical_and(s <= temp_data[:, 2] , temp_data[:, 2] < s+s_slot)))])
            # print("i & j", i, j)
            hist[i][j] = temp_data[(np.where(np.logical_and(s <= temp_data[:, 2], temp_data[:, 2] < s + s_slot)))][:,
                         0].size
            j += 1
        i += 1
    print(hist)
    return hist


def find_bin_size(n):
    return int(round(np.log2(n) + 1))


read_excel_file()
