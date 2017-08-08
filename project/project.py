import numpy as np
import math as m

data_file = "vertebral_2C.xlsx"

counter=0


def read_training_data():
    from pandas import read_excel
    values = (read_excel(data_file, "Sheet2", header=None)).values
    return values[1:311, 0:7]


def process_data(raw_data):
    raw_data[raw_data[:, 6] == 'NO', 6] = 1
    raw_data[raw_data[:, 6] == 'AB', 6] = -1
    # print(d)
    processed_data = np.array(raw_data, dtype=np.float64)
    #print(processed_data)
    np.random.shuffle(processed_data)
    training, test = processed_data[:260, :], processed_data[260:, :]

    return training, test


def add_squared_data(data):
    elements = 2*(data[0].size-1)
    for i in range(0, elements, 2):
        #print(i)
        data = np.insert(data, i+1, np.square(data[:,i]), axis=1)
    return data


def add_multiplied_data(data):
    #print(data)
    elements = data.size
    #print(m.factorial(elements))
    # print(np.array(data)*np.vstack(np.array(data)))
    for i in range(elements):
        print(np.multiply(data[i],data[i+1:elements]).flatten())


def cal_linear_classifier(input_data):
    elements = input_data[0].size
    xa = np.insert(input_data[:, 0:elements-1], 0, 1, axis=1)
    t = input_data[:, [elements-1]]
    #print(xa, t)
    xat = np.linalg.pinv(xa)
    # print(xat)
    w = np.dot(xat, t)
    return w


def test_model(model_w, test_data):
    elements = test_data[0].size
    xa = np.insert(test_data[:, 0:elements - 1], 0, 1, axis=1)
    observed = test_data[:, [elements - 1]]
    predicted = np.dot(xa, model_w)
    # print(predicted)
    predicted[predicted[:, 0] >0, 0] = 1
    predicted[predicted[:, 0] <=0, 0] = -1
    # print(predicted)
    unique, counts = np.unique(np.append(observed, predicted, axis=1), return_counts=True, axis=0)
    print(unique, counts)


def test_tree_model(tree_model, test_data):
    # print("test_data", test_data)
    elements = test_data[:, 0].size
    observed = test_data[:, [test_data[0, :].size - 1]]
    predicted = np.zeros(shape=(observed.size,1))

    for i in range(elements):
        feature_vector = test_data[i]
        # print("\n\n")
        class_label = match_tree(tree_model, feature_vector, 0)
        # print(feature_vector, class_label)
        predicted[i] = [class_label]

    # print("Predicted", predicted)
    unique, counts = np.unique(np.append(observed, predicted, axis=1), return_counts=True, axis=0)
    print(unique, counts)


def match_tree(tree_model, feature_vector, index):
    index = int(index)
    #print(index, "data", tree_model[index], class_label)
    class_label = 0
    if tree_model[index][3].astype(int) != 0:
        if feature_vector[tree_model[index][0].astype(int)] < tree_model[index][2]:
            class_label = match_tree(tree_model,feature_vector, tree_model[index][3])
        elif feature_vector[tree_model[index][0].astype(int)] >= tree_model[index][2]:
            class_label = match_tree(tree_model, feature_vector, tree_model[index][4])
    else :
        class_label = tree_model[index][1]

    return class_label


def apply_stump_algo(processed_data):
    # print(processed_data)
    n = processed_data[:, 0].size
    unique, counts = np.unique(processed_data[:, 1], return_counts=True)
    # print(unique, counts)
    tp = counts[1]
    tn = counts[0]
    # print(n, tp, tn)
    an = 0
    ap = 0
    i0 = iopt = (tn * tp) / (n * n)
    # print("IO", iopt, i0)
    sorted_data = sort_data(processed_data, 0)
    #sorted_data = processed_data
    # print(sorted_data)
    tao = sorted_data[0, 0]
    for i in range(2, n):
        # print(sorted_data[i])
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


def sort_data(data, field) :
    return data[data[:, field].argsort()]


def generate_tree_model(data, tree_model):
    #print("data",data)
    global counter
    index = counter
    counter = counter + 1
    #print("Making index", index, counter)
    tree_node = np.array([0, 0.0, 0.0, 0, 0])
    unique, counts = np.unique(data[:, data[0,:].size - 1], return_counts=True)
    # print("Counts", counts)
    if counts.size > 1 and data[:, 0].size > 5:
        num_features = data[0,:].size - 1
        #last feature is the class label
        # print(num_features)
        # print("tree_node", tree_node)
        for i in range(num_features):
            delta, tao = apply_stump_algo(data[:,[i,num_features]])
            #print(i, delta, tao)
            if delta > tree_node[1] :
                tree_node[0] = i
                tree_node[1] = delta
                tree_node[2] = tao
        #print("\n")
        #print(tree_node)
        #print("\n\n")
        data_1 = data[np.where(data[:,tree_node.astype(int)[0]] < tree_node[2])]
        data_2 = data[np.where(data[:, tree_node.astype(int)[0]] >= tree_node[2])]
        #print(np.amax(data_1[:,3]), np.amin(data_1[:,3]))
        #print("\n\n")
        #print(np.amax(data_2[:,3]), np.amin(data_2[:,3]))
        left_index = generate_tree_model(data_1, tree_model)
        tree_node[3] = left_index
        right_index = generate_tree_model(data_2, tree_model)
        tree_node[4] = right_index
        #print("Index details", index, left_index, right_index)
    else:
        class_label = -1
        if counts.size == 1:
            class_label = unique[0]
        elif counts[0] > counts[1]:
            class_label = -1
        else:
            class_label = 1
        #print("ELSE Counts", unique, counts, class_label)
        tree_node[1] = class_label

    #print("tree node", index, tree_node)
    tree_model[index] = tree_node
    return index



def main() :
    raw_data = read_training_data()
    # print(raw_data.shape)
    training_data, test_data = process_data(raw_data)
    w = cal_linear_classifier(training_data)
    print(training_data.shape, test_data.shape)
    test_model(w, test_data)

    squared_data = add_squared_data(training_data)
    # print(training_data[0])
    # print(squared_data[0])
    w_squared = cal_linear_classifier(squared_data)
    test_model(w_squared, add_squared_data(test_data))
    # print(w_squared)
    # a = range(1,6)[::-1]                       # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # add_multiplied_data(np.array(a))
    tree_model = np.empty(shape=100, dtype=np.ndarray)
    generate_tree_model(training_data, tree_model)
    #print("\n\n\n")
    #tree_model = np.around(tree_model, decimals=2)
    print("Tree", tree_model)
    # print("tree", tree_model[:,0])
    #print(test_data)
    test_tree_model(tree_model, training_data)


main()
