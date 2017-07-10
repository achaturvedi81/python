import numpy as np

data_file = "Assignment_4_Data_and_Template.xlsx"


def read_training_data():
    from pandas import read_excel
    values = (read_excel(data_file, "Training Data", header=None)).values
    return values[1:6601, 0:17]


def write_excel_data(x, sheet_name, start_row, start_col):
    from pandas import DataFrame, ExcelWriter
    from openpyxl import load_workbook
    df = DataFrame(x)
    book = load_workbook(data_file)
    writer = ExcelWriter(data_file, engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(writer, sheet_name=sheet_name, startrow=start_row - 1, startcol=start_col - 1, header=False,
                index=False)
    writer.save()
    writer.close()


def class_to_number(x):
    temp_array = np.negative(np.ones(shape=(6), dtype=int))
    temp_array[x] = 1
    return temp_array


training_data = np.array(read_training_data(), dtype=int)
print("Check shape of matrix", training_data.shape)
# print("Data", training_data[6599,:])
# print("Data", training_data[0,:])
# print(training_data)

Xa = np.append(np.ones(shape=(6600, 1), dtype=int), training_data[:, 0:15], axis=1)
print("Xa.shape", Xa.shape)
T = training_data[:, [15]]
# print(T)
Xat = np.linalg.pinv(Xa)
# print(Xat)
W = np.dot(Xat, T)
# print(W)
# write_excel_data(W,"Classifiers",5,1)
T6 = np.apply_along_axis(class_to_number, 1, training_data[:, [16]])
# print(T6)
W6 = np.dot(Xat, T6)


# print(W6)
# write_excel_data(W6,"Classifiers",5,5)


def read_testing_data():
    from pandas import read_excel
    values = (read_excel(data_file, "To be classified", header=None)).values
    return values[4:54, 0:15]


testing_data = np.array(read_testing_data(), dtype=int)
# print(testing_data)
# print(np.append(np.ones(shape=(50, 1), dtype=int), testing_data, axis=1))
result_failure = np.dot(np.append(np.ones(shape=(50, 1), dtype=int), testing_data, axis=1), W)


def rounding_data(x):
    return [1] if x > 0 else [-1]


# print(result1[:,0])
output_failure = np.apply_along_axis(rounding_data, 1, result_failure[:, [0]])
# print("output", output_failure)
# write_excel_data(output_failure, "To be classified", 5, 16)


result_type = np.dot(np.append(np.ones(shape=(50, 1), dtype=int), testing_data, axis=1), W6)
# print(result_type[:])


def number_to_class(x):
    return [np.argmax(x, axis=0)]

output_type = np.apply_along_axis(number_to_class, 1, result_type)
# write_excel_data(output_type, "To be classified", 5, 17)


result_training_failure = np.dot(Xa, W)
#print(result_training_failure)
output_training_failure = np.apply_along_axis(rounding_data, 1, result_training_failure[:, [0]])
# print(output_training_failure)

def confusion_matrix(tc, cc):
    # print(tc)
    # print(cc)
    cm = np.zeros(shape=(2, 2), dtype=int)
    # temp = [sum(x) for x in zip(tc, cc)]
    temp = np.append(tc, cc, axis=1)
    # print(temp)
    unique, counts = np.unique(temp, return_counts=True, axis=0)
    # print(unique)
    # print(counts)
    cm[1, 0] = counts[0]
    cm[1, 1] = counts[1]
    cm[0, 1] = counts[2]
    cm[0, 0] = counts[3]
    # print(cm)
    return cm


confusion_matrix = confusion_matrix(T, output_training_failure)
#write_excel_data(confusion_matrix, "Performance", 10, 3)


def accuracy(cm):
    # print(cm[0,0] + cm[1,1])
    # print(np.sum(cm))
    return (cm[0,0] + cm[1,1])/np.sum(cm)


def sensitivity(cm):
    # print(cm[1,1])
    # print(cm[1,1] + cm[1,0])
    return cm[1,1]/(cm[1,1] + cm[1,0])


def specifcity(cm):
    # print(cm[0,0])
    # print(cm[0,0] + cm[0,1])
    return cm[0,0]/(cm[0,0] + cm[0,1])


def ppv(cm):
    # print(cm[1,1])
    # print(cm[1,1] + cm[0,1])
    return cm[1,1]/(cm[1,1] + cm[0,1])

print(accuracy(confusion_matrix))
#write_excel_data([accuracy(confusion_matrix)], "Performance", 8, 7)

print(sensitivity(confusion_matrix))
#write_excel_data([sensitivity(confusion_matrix)], "Performance", 9, 7)

print(specifcity(confusion_matrix))
#write_excel_data([specifcity(confusion_matrix)], "Performance", 10, 7)

print(ppv(confusion_matrix))
#write_excel_data([ppv(confusion_matrix)], "Performance", 11, 7)


result_training_type = np.dot(Xa, W6)
#print(result_training_type)
output_training_type = np.apply_along_axis(number_to_class, 1, result_training_type)
#print(output_training_type)


def confusion_matrix_6(tc, cc):
    cm = np.zeros(shape=(6, 6), dtype=int)
    temp = np.append(tc, cc, axis=1)
    unique, counts = np.unique(temp, return_counts=True, axis=0)
    for h in np.arange(unique[:,0].size):
        cm[unique[h][0], unique[h][1]] = counts[h]

    print(cm)
    return cm


confusion_matrix_6 = confusion_matrix_6(training_data[:, [16]], output_training_type)
#write_excel_data(confusion_matrix_6, "Performance", 19, 3)

def ppv_6(cm, class_value):
    # print(sum(cm[:, class_value]))
    # print(cm[class_value, class_value])
    return cm[class_value, class_value]/sum(cm[:, class_value])

max_ppv = 0.0
min_ppv = 0.0
max_ppv_class = 0
min_ppv_class = 0

for x in np.arange(confusion_matrix_6[:,0].size):
    temp = ppv_6(confusion_matrix_6, x)
    #print(x , temp)
    if temp > max_ppv:
        max_ppv = temp
        max_ppv_class = x
    if temp < min_ppv or min_ppv == 0.0:
        min_ppv = temp
        min_ppv_class = x

    print(max_ppv, min_ppv)


write_excel_data([max_ppv], "Performance", 20, 12)
write_excel_data([max_ppv_class], "Performance", 20, 13)
write_excel_data([min_ppv], "Performance", 21, 12)
write_excel_data([min_ppv_class], "Performance", 21, 13)