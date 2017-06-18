import numpy as np # using numpy for various calculations
import xlrd as xlrd  # using xlrd module to read xls data
#from xlrd import open_workbook, cellname

data_file = "Assignment_1_Data_and_Template.xlsx" # base file to load data from
num_bins = 32  # defining number of bins
query_heights_data = [55, 60, 65, 70, 75, 80]
#query_heights_data = [62, 63, 64, 65, 66, 67, 68, 69, 70]
partial_data_size = 50

male_heights_data = []  # array to hold data for male heights
female_heights_data = []  # array to hold data for female heights

min_height = 0
max_height = 0

male_mean_height = 0
male_sd = 0
male_counts = 0

female_mean_height = 0
female_sd = 0
female_counts = 0

male_hist = []  # histogram for male
female_hist = []  # histogram for female


# reading base data xls
def read_xls_workbook(row_count):
    xl_workbook = xlrd.open_workbook(data_file)
    xl_sheet = xl_workbook.sheet_by_index(0)

    for row_idx in range(1, row_count+1 if row_count > 0 else xl_sheet.nrows):  # Iterate through rows
        # Extract three column values from the data
        row_height_ft = xl_sheet.cell(row_idx, 0).value
        row_height_in = xl_sheet.cell(row_idx, 1).value
        row_gender = xl_sheet.cell(row_idx, 2).value

        # seperate data into male and femake bucket
        if row_gender == "Male" :
            male_heights_data.append(row_height_ft * 12 + row_height_in)
        else:
            female_heights_data.append(row_height_ft * 12 + row_height_in)


# method to find mean of a list
def calculate_mean_height(input_list):
    mean_value = sum(input_list) / len(input_list)
    return round(mean_value)


# method to read the data from xls and generate male and female data
def load_and_parse_data(use_all_data) :
    global min_height
    global max_height
    global male_mean_height
    global female_mean_height
    global male_sd
    global female_sd
    global male_counts
    global female_counts

    if use_all_data :
        read_xls_workbook(0)
        male_max = max(male_heights_data)
        male_min = min(male_heights_data)
        female_max = max(female_heights_data)
        female_min = min(female_heights_data)
        min_height = int(min(male_min, female_min))
        max_height = int(max(male_max, female_max))
    else :
        read_xls_workbook(partial_data_size)
        min_height = 52
        max_height = 83

    male_mean_height = calculate_mean_height(male_heights_data)
    female_mean_height = calculate_mean_height(female_heights_data)

    male_sd = np.std(male_heights_data, dtype=np.int64)
    female_sd = np.std(female_heights_data, dtype=np.int64)

    male_counts = len(male_heights_data)
    female_counts = len(female_heights_data)


# method to generate histograms for males and females data based on the bin size provided as "num_bins"
def generate_mf_histogram():
    bin_size = int((max_height - min_height + 1) / num_bins)
    for hist_index in range(min_height, (max_height + 1), bin_size):
        male_hist.append(male_heights_data.count(hist_index))
        female_hist.append(female_heights_data.count(hist_index))


# method to find the probability of being female using histogram data for given height
def female_probability_query(query_height):
    bin_index = query_height - min_height
    if female_hist[bin_index] + male_hist[bin_index] > 0 :
        prob = round(female_hist[bin_index] / (female_hist[bin_index] + male_hist[bin_index]), 2)
    else :
        prob = -1
    return prob


# Gaussian probability dansity function using input x, mean value, sd value and total count
def gauss_pdf(x, mean_value, sd_value, total_count):
    gauss_pd = total_count * np.sqrt(2 * np.pi * sd_value) * np.exp(-np.square((x - mean_value) / sd_value) / 2)
    return gauss_pd


# Method to calculate bayesian probability
def female_bayesian_probability(x):
    prob = gauss_pdf(x, female_mean_height, female_sd, female_counts) / (
    gauss_pdf(x, female_mean_height, female_sd, female_counts) + gauss_pdf(x, male_mean_height, male_sd, male_counts))
    return round(prob, 2)


# Method to print processed data for min, max, mean, sd and histogram ...etc
def show_processed_data():
    print("mean for male height = ", male_mean_height)
    print("mean for female height = ", female_mean_height)
    print("SD for male height = ", male_sd)
    print("SD for female height = ", female_sd)
    print("Count for male samples = ", male_counts)
    print("Count for female samples = ", female_counts)
    print(female_hist)
    print(male_hist)


def prob_query_using_hist():
    for height in query_heights_data:
        print("Probability of being female for height %d" %height, female_probability_query(height))


def bayesian_prob_query():
    for height in query_heights_data:
        print("Bayesian probability of being female for height %d" %height, female_bayesian_probability(height))


def assignment_with_full_data():
    load_and_parse_data(True)
    generate_mf_histogram()
    show_processed_data()
    prob_query_using_hist()
    bayesian_prob_query()


def assignment_with_partial_data():
    load_and_parse_data(False)
    generate_mf_histogram()
    show_processed_data()
    prob_query_using_hist()
    bayesian_prob_query()

# use following method to run assignment for all the data
assignment_with_full_data()

# use following method to run it for partial data
#assignment_with_partial_data()
