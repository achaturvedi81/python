num_bins = 32  # Defining number of bins
male_heights = []  # array to hold data for males
female_heights = []  # array to hold data for females
male_max = 0
male_min = 0
female_max = 0
female_min = 0
male_hist = [] #histogram for male
female_hist = [] #histogram for female

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
    male_max = max(male_heights)
    male_min = min(male_heights)
    female_max = max(female_heights)
    female_min = min(female_heights)
    min_value = int(min(male_min, female_min))
    max_value = int(max(male_max, female_max))
    bin_size = int((max_value - min_value + 1) / num_bins)
    generate_histogram(male_heights, min_value, max_value, bin_size)
    generate_histogram(female_heights, min_value, max_value, bin_size)

def generate_histogram(input_list, min_value, max_value, bin_size):
    print(min_value)
    print(max_value)
    print(bin_size)
    for hist_index in range(min_value, max_value):
        male_hist.append(male_heights.count(hist_index))
        female_hist.append(female_heights.count(hist_index))

read_xls_wrokbook()
generate_mf_histogram()
#print(male_hist)
print(male_hist, sep='\n')
print(female_hist, sep='\n')

# print(len(male_heights))
# print(len(female_heights))
# print(male_heights[1])
# print(max(male_heights))
