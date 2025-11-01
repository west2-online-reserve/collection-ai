import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data = np.empty([5, 10000, 3072])
datameta = []
datalabel = np.empty([5, 10000], dtype=int)
test_data = np.empty([10000, 3072])
testlabel = np.empty([10000], dtype=int)


def fetch_data():
    global datameta
    global data, datalabel
    global test_data, testlabel
    # E:\projects\datasets
    datameta = unpickle(
        'E:/projects/datasets/cifar-10-batches-py/batches.meta')
    for i in range(1, 6):
        file = f'E:/projects/datasets/cifar-10-batches-py/data_batch_{i}'
        dict = unpickle(file)
        data[i-1] = dict[b'data']
        datalabel[i-1] = np.asarray(dict[b'labels'])
    data = np.asarray(data)
    data = data.reshape(5, 10000, 3, 32, 32).astype('float32')/255.0
    test_data = unpickle('E:/projects/datasets/cifar-10-batches-py/test_batch')
    testlabel = np.asarray(test_data[b'labels'])
    test_data = np.asarray(test_data[b'data']).reshape(
        10000, 3, 32, 32).astype('float32')/255.0


'''
def find_the_k():
    total_num = 10000.0
    correct_num = 0.0
    maxn = 0.0
    res = None
    for i in range(1, 11):  # i is the k
        for check_id in range(1, 6):  # 进行测试的组
            for i in range(1, 10001):
                test_data_temp = data[check_id-1][i-1]
                check_arr = np.zeros(10)
                distance = np.empty([40001])
                for test_id in range(1, 6):  # 对其他组进行遍历
                    if check_id == test_id:
                        continue
                    else:
                        temp = np.empty([10000])
                        train_data = data[test_id-1]
                        # 计算距离
                        temp = np.sqrt(
                            np.sum(np.square(test_data_temp - train_data), axis=(1, 2, 3)))
                        distance = np.concatenate((distance, temp))
                distance = np.argpartition(distance, i)[:i]
                for t in distance:
                    check_arr[datalabel[int(np.floor(
                        t/10000))][t % 10000-1]] += 1
                check_arr = np.argmax(check_arr)+1
                if check_arr == datalabel[check_id-1][i-1]:
                    correct_num += 1
        if maxn < correct_num/total_num:
            maxn = correct_num/total_num
            res = i
        print(i)
    return res
md重构

def find_the_k():
    total_num = 10000.0
    max_accuracy = 0.0
    best_k = None
    for k in [1, 3, 5, 10, 20, 50, 100]:
        print(f'Now k is {k}')
        correct_num = 0.0
        for check_id in range(1, 6):
            print(f'Now batch_id is {check_id}')
            batch_correct_num = 0.0
            for i in range(1000):
                print(f'Now sample_id is {i}')
                test_sample = data[check_id-1][i]
                check_arr = np.zeros(10)
                distance_list = []
                for train_id in range(1, 6):
                    if check_id == train_id:
                        train_data = data[train_id-1]
                        distances = np.sqrt(
                            np.sum(np.square(test_sample-train_data), axis=(1, 2, 3)))
                        distance_list.extend(distances)
                    else:
                        train_data = data[train_id-1]
                        distances = np.sqrt(
                            np.sum(np.square(test_sample-train_data), axis=(1, 2, 3)))
                        distance_list.extend(distances)
                distance_list = np.array(distance_list)
                distance_list = np.argpartition(distance_list, k)[:k]
                for t in distance_list:
                    check_arr[datalabel[t//10000][((t+1) % 10000)-1]] += 1
                if np.argmax(check_arr) == datalabel[check_id-1][i]:
                    batch_correct_num += 1
            correct_num += batch_correct_num
        accuracy = correct_num/total_num
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_k = k
    return best_k
md再重构
'''


def find_the_k():
    total_num = 1000.0
    max_accuracy = 0.0
    best_k = None
    for k in [1, 3, 5, 10, 20, 50, 100]:
        # print(f'Now k is {k}')
        correct_num = 0.0
        for i in range(1000):
            # print(f'Now sample_id is {i}')
            test_sample = data[0][i]
            check_arr = np.zeros(10)
            distance_list = []
            train_data = data[0]
            distances = np.sqrt(
                np.sum(np.square(test_sample-train_data[1000:]), axis=(1, 2, 3)))
            distance_list.extend(distances)
            for train_id in range(2, 6):
                train_data = data[train_id-1]
                distances = np.sqrt(
                    np.sum(np.square(test_sample-train_data), axis=(1, 2, 3)))
                distance_list.extend(distances)
            distance_list = np.array(distance_list)
            distance_list = np.argpartition(distance_list, k)[:k]
            for t in distance_list:
                check_arr[datalabel[(t+1000)//10000]
                          [(t+1000) % 10000]] += 1
            if np.argmax(check_arr) == datalabel[0][i]:
                correct_num += 1
        accuracy = correct_num/total_num
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_k = k
    return best_k


'''
def test(k):
    print('Start to final test')
    total_num = 10000.0
    correct_num = 0.0
    for test_id in range(10000):
        print(test_id)
        test_data_temp = test_data[test_id]
        check_arr = np.zeros(10)
        distance_list = []
        for train_id in range(1, 6):
            train_data = data[train_id-1]
            distances = np.sqrt(
                np.sum(np.square(test_data_temp-train_data), axis=(1, 2, 3)))
            distance_list.extend(distances)
        distance_list = np.array(distance_list)
        distance_list = np.argpartition(distance_list, k)[:k]
        for t in distance_list:
            check_arr[datalabel[t//10000][t % 10000]] += 1
        if np.argmax(check_arr) == testlabel[test_id]:
            correct_num += 1
    return correct_num/total_num

def test(k):
    print('Start to final test')
    total_num = 10000
    correct_num = 0
    batch_size = 100  # 添加批量处理

    # 转换为PyTorch张量并移动到GPU
    test_data_tensor = torch.from_numpy(test_data).to(device)
    train_data_tensor = torch.from_numpy(
        data.reshape(-1, 3, 32, 32)).to(device)
    testlabel_tensor = torch.from_numpy(testlabel).to(device)
    train_labels_tensor = torch.from_numpy(datalabel.reshape(-1)).to(device)

    for i in range(0, total_num, batch_size):
        batch_end = min(i + batch_size, total_num)
        print(
            f'Processing batch {i//batch_size + 1}/{(total_num-1)//batch_size + 1}')

        # 获取当前批次
        test_batch = test_data_tensor[i:batch_end]

        # 计算距离矩阵
        diff = test_batch.unsqueeze(1) - train_data_tensor.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=(2, 3, 4)))

        # 获取最近的k个邻居
        _, nearest_indices = torch.topk(distances, k, largest=False)

        # 统计标签并预测
        predictions = torch.mode(
            train_labels_tensor[nearest_indices], dim=1)[0]

        # 计算准确率
        correct_num += torch.sum(predictions ==
                                 testlabel_tensor[i:batch_end]).item()

    accuracy = correct_num / total_num
    print(f'Test accuracy: {accuracy:.4f}')
    return accuracy
'''


def test(k):
    print('Start to final test')
    total_num = 10000.0
    correct_num = 0.0
    for test_id in range(10000):
        # print(test_id)
        test_data_temp = test_data[test_id]
        check_arr = np.zeros(10)
        distance_list = []
        for train_id in range(1, 6):
            train_data = data[train_id-1]
            distances = np.sqrt(
                np.sum(np.square(test_data_temp-train_data), axis=(1, 2, 3)))
            distance_list.extend(distances)
        distance_list = np.array(distance_list)
        distance_list = np.argpartition(distance_list, k)[:k]
        for t in distance_list:
            check_arr[datalabel[t//10000][t % 10000]] += 1
        if np.argmax(check_arr) == testlabel[test_id]:
            correct_num += 1
    return correct_num/total_num


if __name__ == '__main__':
    # print(torch.cuda.is_available())
    # exit()
    fetch_data()
    # k = find_the_k()
    k = 5
    print(f'{k} is the best k')
    print(test(k))
