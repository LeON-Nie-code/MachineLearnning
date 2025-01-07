import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split_data(X, y, split_size=[0.8, 0.2], shuffle=False, random_seed=None):
    """
    对数据集进行划分

    Args：
        X - 特征向量，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        y - 标签向量，一个大小为 (num_instances) 的一维 numpy 数组
        split_size - 划分比例，期望为一个浮点数列表，如[0.8, 0.2]表示将数据集划分为两部分，比例为80%和20%
        shuffle - 是否打乱数据集
        random_seed - 随机种子
        
    Return：
        X_list - 划分后的特征向量列表
        y_list - 划分后的标签向量列表
    """
    assert sum(split_size) == 1
    num_instances = X.shape[0]
    if shuffle:
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(num_instances)
        X = X[indices]
        y = y[indices]

    # TODO 2.1.1
        
    # 计算训练集的大小
    train_size = int(num_instances * split_size[0])
    
    # 分割数据集
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 返回分割后的数据
    X_list = [X_train, X_test]
    y_list = [y_train, y_test]
    
    return X_list, y_list
    
    

    

def feature_normalization(train, test):
    """将训练集中的所有特征值映射至[0,1]，对测试集上的每个特征也需要使用相同的仿射变换

    Args：
        train - 训练集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        test - 测试集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
    Return：
        train_normalized - 特征归一化后的训练集
        test_normalized - 特征归一化后的测试集

    """
    # TODO 2.1.2

    # 计算训练集中的最小值和最大值
    train_min = np.min(train, axis=0)  # 对每个特征的最小值
    train_max = np.max(train, axis=0)  # 对每个特征的最大值
    
    # 防止除以零，避免最大值等于最小值的情况
    diff = train_max - train_min
    diff[diff == 0] = 1  # 避免出现除以 0 的情况
    
    # 归一化训练集
    train_normalized = (train - train_min) / diff
    
    # 使用训练集的最小值和最大值来归一化测试集
    test_normalized = (test - train_min) / diff
    
    return train_normalized, test_normalized



def compute_regularized_square_loss(X, y, theta, lambda_reg):
    """
    给定一组 X, y, theta，计算用 X*theta 预测 y 的岭回归损失函数

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小 (num_features)
        lambda_reg - 正则化系数

    Return：
        loss - 损失函数，标量
    """
    # TODO 2.2.2
    m = X.shape[0]  # 样本数量
    h_theta = X @ theta  # 预测值
    loss = np.sum((h_theta - y) ** 2) / (2 * m)  # 均方误差
    regularization = lambda_reg * np.sum(theta ** 2) / 2  # 正则化项
    J = (loss + regularization)
    return J



def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    计算岭回归损失函数的梯度

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数

    返回：
        grad - 梯度向量，数组大小（num_features）
    """
    # TODO 2.2.4
    m = X.shape[0]
    h_theta = X @ theta  # 预测值
    error = h_theta - y
    grad = (X.T @ error) / m + lambda_reg * theta  # 计算梯度
    # grad = grad * 2
    return grad



def grad_checker(X, y, theta, lambda_reg, epsilon=0.01, tolerance=1e-4):
    """梯度检查
    如果实际梯度和近似梯度的欧几里得距离超过容差，则梯度计算不正确。

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数
        epsilon - 步长
        tolerance - 容差

    Return：
        梯度是否正确

    """
    grad_computed = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
    num_features = theta.shape[0]
    grad_approx = np.zeros(num_features)

    for h in np.identity(num_features):
        J0 = compute_regularized_square_loss(X, y, theta - epsilon * h, lambda_reg)
        J1 = compute_regularized_square_loss(X, y, theta + epsilon * h, lambda_reg)
        grad_approx += (J1 - J0) / (2 * epsilon) * h
    dist = np.linalg.norm(grad_approx - grad_computed)
    return dist <= tolerance


def grad_descent(X, y, lambda_reg, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    全批量梯度下降算法

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        alpha - 梯度下降的步长，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        check_gradient - 更新时是否检查梯度

    Return：
        theta_hist - 存储迭代中参数向量的历史，大小为 (num_iter+1, num_features) 的二维 numpy 数组
        loss_hist - 全批量损失函数的历史，大小为 (num_iter) 的一维 numpy 数组
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 2.3.3

    for i in range(num_iter):
        # 计算当前梯度
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        
        # 如果需要检查梯度
        if check_gradient:
            grad_diff = grad_checker(X, y, theta, lambda_reg)
            print(f"Iteration {i}, Gradient Check Diff: {grad_diff}")
        
        # 更新 theta
        theta = theta - alpha * grad
        # print(f"theta: {theta}, alpha: {alpha}, grad: {grad}")
        # 判断theta是否是NaN
        if np.isnan(theta).any():
            print("theta is NaN")
            exit(1)
      
        
        # 记录 theta 和损失
        theta_hist[i + 1] = theta
        loss_hist[i] = compute_regularized_square_loss(X, y, theta, lambda_reg)
        
    return theta_hist, loss_hist

# 运行梯度下降实验
def run_experiment(X, y, lambda_reg=0):
    # learning_rates = [0.5, 0.1, 0.05, 0.01]
    learning_rates = [ 0.1, 0.05, 0.01]

    num_iter = 1000
    
    for alpha in learning_rates:
        theta_hist, loss_hist = grad_descent(X, y, lambda_reg, alpha=alpha, num_iter=num_iter)
        plt.plot(loss_hist, label=f"alpha={alpha}")

    plt.title("Loss vs Iterations for Different Learning Rates")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def check_overflow(theta):
    if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
        print("Overflow or invalid value detected in theta")
        return True
    return False





def stochastic_grad_descent(X_train, y_train, X_val, y_val, lambda_reg, alpha=0.1, num_iter=1000, batch_size=1):
    """
    随机梯度下降，并随着训练过程在验证集上验证

    参数：
        X_train - 训练集特征向量，数组大小 (num_instances, num_features)
        y_train - 训练集标签向量，数组大小 (num_instances)
        X_val - 验证集特征向量，数组大小 (num_instances, num_features)
        y_val - 验证集标签向量，数组大小 (num_instances)
        alpha - 梯度下降的步长，可自行调整为默认值以外的值
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量正则化损失函数的历史，数组大小(num_iter)
        validation hist - 验证集上全批量均方误差（不带正则化项）的历史，数组大小(num_iter)
    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    validation_hist = np.zeros(num_iter)  # Initialize validation_hist\

    for i in range(num_iter):
        # 随机选择一个小批量数据
        indices = np.random.choice(num_instances, batch_size, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        
        # 计算梯度
        gradient = compute_regularized_square_loss_gradient(X_batch, y_batch, theta, lambda_reg)
        
        # 更新参数
        theta = theta - alpha * gradient

        # print(theta)
        if(check_overflow(theta)):
            theta = np.delete(theta, -1)
            break
        
        
        # 保存当前参数向量
        theta_hist[i + 1] = theta
        
        # 计算当前训练集上的损失
        loss_hist[i] = compute_regularized_square_loss(X_batch, y_batch, theta, lambda_reg)

        # 计算验证集上的损失（不带正则化项）
        validation_errors = X_val.dot(theta) - y_val
        validation_hist[i] = (1 / (2 * len(y_val))) * np.sum(validation_errors**2)
        # validation_hist[i] = np.sum((X_val @ theta - y_val) ** 2) / (2 * len(y_val))
        

        
    
    return theta_hist, loss_hist, validation_hist


    

    # TODO 2.4.3


def test_batch_sizes(X, y, batch_sizes, lambda_reg=0, alpha=0.01, num_iter=1000):
    """
    测试不同批大小对模型收敛情况的影响，并记录验证集上的全批量损失
    
    参数：
        X - 输入特征矩阵
        y - 输出标签向量
        batch_sizes - 批大小列表
        lambda_reg - 正则化系数
        alpha - 学习率
        num_iter - 迭代次数
    
    返回：
        results - 批大小和对应的验证集损失
    """
    # 划分数据集
    (X_train, X_val), (y_train, y_val) = split_data(X, y,shuffle=True, random_seed=0) # 划分数据集,shuffle=True,否则会导致验证集上的损失越来越大

    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)
    
    # 存储结果
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Training with batch size: {batch_size}")
        
        # 进行随机梯度下降
        theta_hist, loss_hist, validation_hist = stochastic_grad_descent(
            X_train, y_train, X_val, y_val, lambda_reg, alpha, num_iter, batch_size)
        
        # 记录验证集上的最后一次全批量损失
        final_validation_loss = validation_hist[-1]
        results[batch_size] = validation_hist
    
    # print(results)
    
    return results


def run_test_batch_sizes(X,y):
    # 测试不同批大小的效果
    # batch_sizes = [1, 5, 10,15, 20,25, 30,35, 40, 45, 50,55, 60, 65, 70, 75,  80, 85,90,95,  100,150]
    batch_sizes = [1, 5, 10, 50, 100]
    # batch_sizes = [150]
    results = test_batch_sizes(X, y, batch_sizes)

    # # 绘制验证集损失与批大小的关系
    # plt.plot(list(results.keys()), list(results.values()))
    # plt.xlabel('Batch Size')
    # plt.ylabel('Final Validation Loss')
    # plt.title('Effect of Batch Size on Model Convergence')
    # plt.show()
    for batch_size in batch_sizes:
        plt.plot(results[batch_size], label=f"batch={batch_size}")

    plt.title("Loss vs Iterations for Different batch sizes")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


cross_validation_K = 5
def K_fold_split_data(X, y, K=cross_validation_K, shuffle=False, random_seed=None):
    """
    K 折划分数据集

    Args：
        X - 特征向量，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        y - 标签向量，一个大小为 (num_instances) 的一维 numpy 数组
        K - 折数
        shuffle - 是否打乱数据集
        random_seed - 随机种子

    Return：
        X_train_list - 划分后的训练集特征向量列表
        y_train_list - 划分后的训练集标签向量列表
        X_valid_list - 划分后的验证集特征向量列表
        y_valid_list - 划分后的验证集标签向量列表
    """
    num_instances = X.shape[0]
    if shuffle:
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(num_instances)
        X = X[indices]
        y = y[indices]
    X_train_list, y_train_list = [], []
    X_valid_list, y_valid_list = [], []
    
    # TODO 2.5.1
    # 计算每折的大小
    fold_size = num_instances // K
    X_train_list, y_train_list = [], []
    X_valid_list, y_valid_list = [], []

    # 划分数据集
    for k in range(K):
        # 划分验证集
        start_valid = k * fold_size
        end_valid = (k + 1) * fold_size if k < K - 1 else num_instances
        
        X_valid = X[start_valid:end_valid]
        y_valid = y[start_valid:end_valid]

        # 划分训练集
        X_train = np.concatenate((X[:start_valid], X[end_valid:]), axis=0)
        y_train = np.concatenate((y[:start_valid], y[end_valid:]), axis=0)

        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_valid_list.append(X_valid)
        y_valid_list.append(y_valid)

    return X_train_list, y_train_list, X_valid_list, y_valid_list



def K_fold_cross_validation(X, y, alphas, lambdas, num_iter=1000, K=cross_validation_K, shuffle=False, random_seed=None):
    """
    K 折交叉验证

    Args：
        X - 特征向量，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        y - 标签向量，一个大小为 (num_instances) 的一维 numpy 数组
        alphas - 搜索的步长列表
        lambdas - 搜索的正则化系数列表
        num_iter - 要运行的迭代次数
        K - 折数
        shuffle - 是否打乱数据集
        random_seed - 随机种子

    Return：
        alpha_best - 最佳步长
        lambda_best - 最佳正则化系数
    """
    # alpha_best, lambda_best = None, None
    # X_train_list, y_train_list, X_valid_list, y_valid_list = K_fold_split_data(X, y, K, shuffle, random_seed)
    
    # TODO 2.5.2

    # 分割数据集
    X_train_list, y_train_list, X_valid_list, y_valid_list = K_fold_split_data(X, y, K, shuffle, random_seed)
    
    best_alpha, best_lambda = None, None
    min_mse = float('inf')  # 用于记录最小的均方误差

    # 用于存储不同超参数组合下的 MSE 值
    results_table = []

    # 迭代所有可能的 alpha 和 lambda 组合
    for alpha in alphas:
        for lambda_reg in lambdas:
            mse_total = 0

            # 对每一折进行交叉验证
            for k in range(K):
                X_train, y_train = X_train_list[k], y_train_list[k]
                X_valid, y_valid = X_valid_list[k], y_valid_list[k]

                # 训练模型
                theta_hist, _ = grad_descent(X_train, y_train, lambda_reg, alpha=alpha, num_iter=num_iter)

                # 获取最终的参数 theta
                theta_final = theta_hist[-1]

                # 在验证集上计算均方误差
                h_theta = X_valid @ theta_final
                mse = np.mean((h_theta - y_valid) ** 2)
                mse_total += mse
            
            # 计算当前超参数组合下的平均 MSE
            avg_mse = mse_total / K
            results_table.append((alpha, lambda_reg, avg_mse))
            
            # 如果当前组合的 MSE 最小，更新最佳超参数
            if avg_mse < min_mse:
                min_mse = avg_mse
                best_alpha = alpha
                best_lambda = lambda_reg

    # 打印超参数搜索结果
    for alpha, lambda_reg, avg_mse in results_table:
        print(f"alpha={alpha}, lambda={lambda_reg}, avg_mse={avg_mse}")

    return best_alpha, best_lambda



def analytical_solution(X, y, lambda_reg):
    """
    岭回归解析解

    Args：
        X - 特征向量，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        y - 标签向量，一个大小为 (num_instances) 的一维 numpy 数组
        lambda_reg - 正则化系数

    Return：
        theta - 参数向量
    """
    assert lambda_reg > 0
    # TODO 2.6.1

     
    # 计算岭回归解析解
    return np.linalg.inv(X.T @ X + lambda_reg * np.eye(X.shape[1])) @ X.T @ y

def run_test_analytical_solution(X, y):
    # 划分数据集
    (X_train, X_val), (y_train, y_val) = split_data(X, y, shuffle=True, random_seed=0)
    # 假设 X_train, y_train 是训练集，X_test, y_test 是测试集
    lambdas=[1e-7,1e-5,1e-3,1e-1,1,10, 100]
    mse_results = []

    # 特征归一化
    X_train, X_val = feature_normalization(X_train, X_val)

    # 增加偏置项
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))

    for lambda_ in lambdas:
        theta = analytical_solution(X_train, y_train, lambda_)
        h_theta = X_val @ theta
        mse = np.mean((h_theta - y_val) ** 2)
        mse_results.append(mse)
        print(f"Lambda: {lambda_}, MSE: {mse}")


    # 打印结果

    results_df = pd.DataFrame({'Lambda': lambdas, 'MSE': mse_results})
    print(results_df)



def main():
    # 加载数据集
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    (X_train, X_test), (y_train, y_test) = split_data(X, y, split_size=[0.8, 0.2], shuffle=True, random_seed=0)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # 增加偏置项
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # 增加偏置项
    


    # TODO

    # 运行梯度下降实验
    # 2.3.3 步长选择,绘制不同步长下的目标函数值随训练时间的变化曲线
    run_experiment(X_train, y_train)
    
    # 2.4.2 K_fold_cross_validation, 交叉验证,寻找最优的模型训练超参数
    best_alpha, best_lambda = K_fold_cross_validation(X_train, y_train, alphas=[0.05, 0.04,0.03,0.02,0.01], lambdas=[1e-7,1e-5,1e-3,1e-1,1,10], num_iter=1000, K=5, shuffle=True, random_seed=0)
    print(f"Best alpha: {best_alpha}, Best lambda: {best_lambda}")

    # 2.5.4 随机梯度下降,测试不同批大小的效果
    # 记录随着批大小逐渐增大时，训练曲线发生的变化
    run_test_batch_sizes(X, y)

    # 2.6.1 岭回归解析解,测试解析解的效果
    run_test_analytical_solution(X, y)


if __name__ == "__main__":
    main()
