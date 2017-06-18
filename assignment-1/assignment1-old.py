import numpy as np

num_bins = 32  # Defining number of bins
male_heights = []  # array to hold data for males
female_heights = []  # array to hold data for females
male_max = 0
male_min = 0
female_max = 0
female_min = 0
male_hist = []  # histogram for male
female_hist = []  # histogram for female
hight_value = []
male_mean_height = 0
female_mean_height = 0
male_sd = 0
female_sd = 0
male_counts = 0
female_counts = 0
min_height = 0
max_height = 0


# method to read the data from xls and generate male and female data
def read_xls_wrokbook():
    import xlrd  # using xlrd module to read xls data
    from xlrd import open_workbook, cellname
    xl_workbook = open_workbook('Assignment_1_Data_and_Template.xlsx')  # reading the workbook data sheet
    xl_sheet = xl_workbook.sheet_by_index(0)

    for row_idx in range(1, xl_sheet.nrows):  # Iterate through rows
        # Extract three column values from the data
        row_height_ft = xl_sheet.cell(row_idx, 0).value
        row_height_in = xl_sheet.cell(row_idx, 1).value
        row_gender = xl_sheet.cell(row_idx, 2).value

        # seperate data into male and femake bucket
        if row_gender == "Male":
            male_heights.append(row_height_ft * 12 + row_height_in)
        else:
            female_heights.append(row_height_ft * 12 + row_height_in)


def generate_mf_histogram():
    bin_size = int((max_height - min_height + 1) / num_bins)
    generate_histogram(male_heights, bin_size)
    generate_histogram(female_heights, bin_size)


def generate_histogram(input_list, bin_size):
    # print(min_height)
    # print(max_height)
    # print(bin_size)
    for hist_index in range(min_height, max_height + 1):
        hight_value.append(hist_index)
        male_hist.append(male_heights.count(hist_index))
        female_hist.append(female_heights.count(hist_index))


def calculate_mean_height(input_list):
    mean_value = sum(input_list) / len(input_list)
    return round(mean_value)


def probability_query(query_height):
    bin_index = query_height - min_height;
    print("bin_index", bin_index)
    prob = round(female_hist[bin_index] / (female_hist[bin_index] + male_hist[bin_index]), 2)
    return prob


def gauss_pdf(x, mean_value, sd_value, total_count):
    gauss_pd = total_count * np.sqrt(2 * np.pi * sd_value) * np.exp(-np.square((x - mean_value) / sd_value) / 2)
    return gauss_pd


def female_bayesian_probability(x):
    prob = gauss_pdf(x, female_mean_height, female_sd, female_counts) / (
    gauss_pdf(x, female_mean_height, female_sd, female_counts) + gauss_pdf(x, male_mean_height, male_sd, male_counts))
    return round(prob, 2)


read_xls_wrokbook()
male_max = max(male_heights)
male_min = min(male_heights)
female_max = max(female_heights)
female_min = min(female_heights)
min_height = int(min(male_min, female_min))
max_height = int(max(male_max, female_max))

generate_mf_histogram()
male_mean_height = calculate_mean_height(male_heights)
female_mean_height = calculate_mean_height(female_heights)

male_sd = np.std(male_heights, dtype=np.int64)
female_sd = np.std(female_heights, dtype=np.int64)

male_counts = len(male_heights)
female_counts = len(female_heights)

# print(male_hist)
print("mean for male height = ", male_mean_height)
print("mean for female height = ", female_mean_height)
print("STD for male height = ", male_sd)
print("STD for female height = ", female_sd)
print("Count for male samples = ", male_counts)
print("Count for female samples = ", female_counts)
print(hight_value, sep='\n')
print(female_hist, sep='\n')
print(male_hist, sep='\n')

# print(male_hist[0])
# print("Probability of 64", probability_query(64))
print("Probability of 55", probability_query(55))
print("Probability of 60", probability_query(60))
print("Probability of 65", probability_query(65))
print("Probability of 70", probability_query(70))
print("Probability of 75", probability_query(75))
print("Probability of 80", probability_query(80))

print("Bayesian probability of 55", female_bayesian_probability(55))
print("Bayesian probability of 60", female_bayesian_probability(60))
print("Bayesian probability of 65", female_bayesian_probability(65))
print("Bayesian probability of 70", female_bayesian_probability(70))
print("Bayesian probability of 75", female_bayesian_probability(75))
print("Bayesian probability of 80", female_bayesian_probability(80))

# print(len(male_heights))
# print(len(female_heights))
# print(male_heights[1])
# print(max(male_heights))
