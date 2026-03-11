
# def find_the_k():
#     total_num = 10000.0
#     max_accuracy = 0.0
#     best_k = None

#     for k in range(1, 11):
#         total_correct = 0.0
#         print(f"Testing k={k}")

#         for check_id in range(1, 6):
#             print(f"Testing batch {check_id}/5")
#             batch_correct = 0.0

#             for i in range(10000):
#                 test_sample = data[check_id-1][i]
#                 check_arr = np.zeros(10)
#                 distance_list = []

#                 # 计算与其他所有batch的距离
#                 for test_id in range(1, 6):
#                     if check_id == test_id:
#                         continue

#                     train_data = data[test_id-1]
#                     # 向量化计算距离
#                     distances = np.sqrt(
#                         np.sum(np.square(test_sample - train_data), axis=(1, 2, 3)))
#                     distance_list.extend(distances)

#                 # 获取k个最近邻
#                 distance = np.array(distance_list)
#                 nearest_indices = np.argpartition(distance, k)[:k]

#                 # 统计标签
#                 for idx in nearest_indices:
#                     batch_idx = idx // 10000
#                     sample_idx = idx % 10000
#                     check_arr[datalabel[batch_idx][sample_idx]] += 1

#                 # 预测标签
#                 predicted_label = np.argmax(check_arr)
#                 if predicted_label == datalabel[check_id-1][i]:
#                     batch_correct += 1

#             total_correct += batch_correct

#         # 计算准确率
#         accuracy = total_correct / (total_num * 5)
#         print(f"Accuracy for k={k}: {accuracy}")

#         if accuracy > max_accuracy:
#             max_accuracy = accuracy
#             best_k = k

#     return best_k


# def test(k):
#     print('Start to final test')
#     print(f'Using device: {device}')
#     total_num = 10000
#     correct_num = 0
#     batch_size = 10  # 减小batch_size
#     chunk_size = 1000  # 添加分块大小

#     # 转换为PyTorch张量并移动到GPU
#     test_data_tensor = torch.from_numpy(test_data).to(device)
#     train_data_tensor = torch.from_numpy(
#         data.reshape(-1, 3, 32, 32)).to(device)
#     train_labels_tensor = torch.from_numpy(datalabel.reshape(-1)).to(device)
#     testlabel_tensor = torch.from_numpy(testlabel).to(device)

#     for i in range(0, total_num, batch_size):
#         batch_end = min(i + batch_size, total_num)
#         print(
#             f'Processing batch {i//batch_size + 1}/{(total_num-1)//batch_size + 1}')

#         test_batch = test_data_tensor[i:batch_end]
#         batch_predictions = []

#         # 对训练数据进行分块处理
#         for j in range(0, len(train_data_tensor), chunk_size):
#             chunk_end = min(j + chunk_size, len(train_data_tensor))
#             train_chunk = train_data_tensor[j:chunk_end]

#             # 计算当前块的距离
#             diff = test_batch.unsqueeze(1) - train_chunk.unsqueeze(0)
#             distances = torch.sqrt(torch.sum(diff ** 2, dim=(2, 3, 4)))

#             # 获取当前块的k个最近邻
#             _, nearest_indices = torch.topk(distances, k, largest=False)
#             chunk_predictions = train_labels_tensor[j:chunk_end][nearest_indices]
#             batch_predictions.append(chunk_predictions)

#         # 合并所有块的结果
#         batch_predictions = torch.cat(batch_predictions, dim=1)
#         predictions = torch.mode(batch_predictions, dim=1)[0]

#         # 计算准确率
#         correct_num += torch.sum(predictions ==
#                                  testlabel_tensor[i:batch_end]).item()

#     accuracy = correct_num / total_num
#     print(f'Test accuracy: {accuracy:.4f}')
#     return accuracy
