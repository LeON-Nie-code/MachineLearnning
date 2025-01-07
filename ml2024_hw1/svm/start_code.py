import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import trange


def load_text_dataset(filename, positive='joy', negative='sadness'):
    """
    从文件filename读入文本数据集
    """
    data = pd.read_csv(filename)
    is_positive = data.Emotion == positive
    is_negative = data.Emotion == negative
    data = data[is_positive | is_negative]
    X = data.Text  # 输入文本
    y = np.array(data.Emotion == positive) * 2 - 1  # 1: positive, -1: negative
    return X, y


def vectorize(train, val):
    """
    将训练集和验证集中的文本转成向量表示

    Args：
        train - 训练集，大小为 num_instances 的文本数组
        val - 测试集，大小为 num_instances 的文本数组
    Return：
        train_normalized - 向量化的训练集 (num_instances, num_features)
        val_normalized - 向量化的测试集 (num_instances, num_features)
    """
    tfidf = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    val_normalized = tfidf.transform(val).toarray()
    return train_normalized, val_normalized


def linear_svm_subgrad_descent(X, y, alpha=0.05, lambda_reg=0.0001, num_iter=60000, batch_size=1):
    """
    线性SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。梯度下降步长，可自行调整为默认值以外的值或扩展为步长策略
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量损失函数的历史，数组大小(num_iter)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 3.5.1
    for i in range(num_iter):
        indices = np.random.choice(num_instances, batch_size, replace=False)  # 随机选择batch_size个样本
        X_batch = X[indices]
        y_batch = y[indices]

        # 计算损失和梯度
        margins = 1 - y_batch * (X_batch @ theta)
        loss = np.maximum(0, margins)
        loss_mean = np.mean(loss) + (lambda_reg / 2) * np.dot(theta, theta)
        loss_hist[i] = loss_mean

        # 计算梯度
        gradient = -np.mean((y_batch * (margins > 0)).reshape(-1, 1) * X_batch, axis=0) + lambda_reg * theta
        theta -= alpha * gradient  # 更新参数

    return theta, loss_hist


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    results = []

    for alpha in [0.01, 0.05, 0.1]:
        for lambda_reg in [0.0001, 0.001]:
            for batch_size in [1, 10, 50]:
                print("batch size: ", batch_size)
                theta, loss_hist = linear_svm_subgrad_descent(X_train, y_train, alpha=alpha, lambda_reg=lambda_reg, num_iter=60000, batch_size=batch_size)

                # 计算训练集和验证集准确率
                train_accuracy = np.mean(np.sign(X_train @ theta) == y_train)
                val_accuracy = np.mean(np.sign(X_val @ theta) == y_val)
                results.append((alpha, lambda_reg, batch_size, train_accuracy, val_accuracy))

    return results

def gaussian_kernel(X, sigma=1.0):
    K = np.exp(-np.linalg.norm(X[:, None] - X, axis=2) ** 2 / (2 * sigma ** 2))
    return K
    

def kernel_svm_subgrad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000, batch_size=1):
    """
    Kernel SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。初始梯度下降步长
        lambda_reg - 正则化系数
        num_iter - 遍历整个训练集的次数（即次数）
        batch_size - 批大小

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter, num_features)
        loss hist - 正则化损失函数向量的历史，数组大小(num_iter,)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_instances)  # Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros((num_iter+1,))  # Initialize loss_hist

    # TODO 3.5.3
    for i in range(num_iter):
        indices = np.random.choice(num_instances, batch_size, replace=False)  # 随机选择batch_size个样本
        y_batch = y[indices]
        K_batch = K[indices][:, indices]

        margins = 1 - y_batch * (theta @ K_batch)  # 使用核函数计算边际
        loss = np.maximum(0, margins)
        loss_mean = np.mean(loss) + (lambda_reg / 2) * np.dot(theta, theta)
        loss_hist[i] = loss_mean

        # 计算梯度
        gradient = -np.mean((y_batch * (margins > 0))[:, None] * K_batch, axis=0) + lambda_reg * theta
        theta -= alpha * gradient  # 更新参数

    return theta, loss_hist

def evaluate_model(X_val, y_val, theta):
    predictions = np.sign(X_val @ theta)
    accuracy = np.mean(predictions == y_val)
    f1 = f1_score(y_val, predictions)
    cm = confusion_matrix(y_val, predictions)
    return accuracy, f1, cm


def main():
    # 加载所有数据
    X_train, y_train = load_text_dataset("data_train.csv", "joy", "sadness")
    X_val, y_val = load_text_dataset("data_val.csv")
    print("Training Set Size: {} Validation Set Size: {}".format(len(X_train), len(X_val)))
    print("Training Set Text:", X_train, sep='\n')

    # 将训练集和验证集中的文本转成向量表示
    X_train_vect, X_val_vect = vectorize(X_train, X_val)
    X_train_vect = np.hstack((X_train_vect, np.ones((X_train_vect.shape[0], 1))))  # 增加偏置项
    X_val_vect = np.hstack((X_val_vect, np.ones((X_val_vect.shape[0], 1))))  # 增加偏置项

    # 调整超参数并记录结果
    results = tune_hyperparameters(X_train_vect, y_train, X_val_vect, y_val)

    # 打印结果
    print("超参数调整结果:")
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("步长", "正则化", "批大小", "训练准确率", "验证准确率"))
    for alpha, lambda_reg, batch_size, train_acc, val_acc in results:
        print("{:<10} {:<10} {:<10} {:<10.4f} {:<10.4f}".format(alpha, lambda_reg, batch_size, train_acc, val_acc))


    
    
    # 在最佳超参数下训练 SVM
    # 这里可以选择结果中验证集准确率最高的组合进行训练
    best_result = max(results, key=lambda x: x[4])  # 根据验证集准确率找到最佳组合
    best_alpha, best_lambda, best_batch_size, _, _ = best_result
    theta, loss_hist = linear_svm_subgrad_descent(X_train_vect, y_train, alpha=best_alpha, lambda_reg=best_lambda, num_iter=60000, batch_size=best_batch_size)

    # 计算最终模型性能
    accuracy, f1, cm = evaluate_model(X_val_vect, y_val, theta)
    print("最终验证集准确率: {:.4f}".format(accuracy))
    print("F1-Score: {:.4f}".format(f1))
    print("混淆矩阵:\n", cm)

    # SVM的随机次梯度下降训练
    # TODO
    

    # 计算SVM模型在验证集上的准确率，F1-Score以及混淆矩阵
    # TODO


if __name__ == '__main__':
    main()
