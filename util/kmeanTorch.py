import torch
import time
class KMEANS:
    def __init__(self, n_clusters=10, max_iter=None, verbose=True, device=torch.device("cpu")):

        # self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        # print(init_row.shape)    # shape 10
        init_points = x[init_row]
        # print(init_points.shape) # shape (10, 2048)
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        return self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        # print(labels.shape)  # shape (250000)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        # print(dists.shape)   # shape (0, 10)
        # for i, sample in tqdm(enumerate(x)):
        for i, sample in enumerate(x):
            # print(self.centers.shape) # shape(10, 2048)
            # print(sample.shape)       # shape 2048
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            # print(dist.shape)         # shape 10
            labels[i] = torch.argmin(dist)
            # print(labels.shape)       # shape 250000
            # print(labels[:10])
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
            # print(dists.shape)        # shape (1,10)
            # print('*')
        self.labels = labels           # shape 250000
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists              # 250000, 10
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device) # shape (0, 250000)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))  # 10, 2048
        self.centers = centers  # shape (10, 2048)

    def representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        # print(self.dists.shape)
        self.representative_samples = torch.argmin(self.dists, 1)
        # print(self.representative_samples.shape)  # shape 250000
        # print('*')
        return self.representative_samples


def time_clock(matrix, device):
    a = time.time()
    k = KMEANS(max_iter=10,verbose=False,device=device)
    classifier_n_labels_result = k.fit(matrix)
    b = time.time()
    return (b-a)/k.count


def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import pickle
    from tqdm import tqdm

    device = choose_device(True)

    # read data
    train_ava_data_list = list(pd.read_csv('train_val_dataset.txt', header=None, sep=' ', index_col=False)[0])
    test_ava_data_list = list(pd.read_csv('test_dataset_19929.txt', header=None, sep=' ', index_col=False)[0])
    all_data_list = train_ava_data_list + test_ava_data_list
    print(len(all_data_list))
    all_data_tensor = torch.empty(0, 2048).cuda()
    for i in tqdm(all_data_list):
        elem_path = '/home/flyingbird/Data/feature_extract/feature_2048/train/' + str(i)
        with open(elem_path, 'rb') as f:
            elem_tensor = pickle.load(f, encoding='latin1')
        all_data_tensor = torch.cat((all_data_tensor, torch.Tensor(elem_tensor).unsqueeze(0).cuda()), dim=0)
        # print(all_data_tensor.shape)
        # print('*')
    print(all_data_tensor.shape)

    # knn
    k = KMEANS(max_iter=10, verbose=False, device=device)
    classifier_result = k.fit(all_data_tensor.cuda())
    # print(classifier_result[:10])
    print(classifier_result.shape)
    classifier_result = classifier_result.cpu().numpy()

    # save result (img_id : label)
    dict = {0:all_data_list, 1:classifier_result}
    pd.DataFrame(dict).to_csv('k_means_all_ava_data_label.txt', sep=' ', header=None, index=False)


    # speed = time_clock(matrix, device)
    # print(speed)
    # cpu_speeds.append(speed)
    # l1, = plt.plot(2048, cpu_speeds,color = 'r',label = 'CPU')

    # device = choose_device(True)
    #
    # gpu_speeds = []
    # for i in tqdm([20, 100, 500, 2000, 8000, 20000]):
    #     matrix = torch.rand((250000, i)).to(device)
    #     speed = time_clock(matrix,device)
    #     gpu_speeds.append(speed)
    # l2, = plt.plot([20, 100, 500, 2000, 8000, 20000], gpu_speeds, color='g',label = "GPU")

    # plt.xlabel("num_features")
    # plt.ylabel("speed(s/iter)")
    # plt.title("Speed with cuda")
    # plt.legend(handles = [l1],labels = ['GPU'],loc='best')
    # plt.savefig("speed.jpg")
