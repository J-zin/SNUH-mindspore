import datetime
import scipy.io
import numpy as np
from copy import deepcopy
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.dataset as ds
from scipy.special import softmax
transpose = ops.Transpose()
matmul = ops.MatMul()
topk = ops.TopK(sorted=True)

class Data:
    def __init__(self, file_path, num_neighbors):
        self.file_path = file_path
        self.load_datasets()
        self.GetTopK_UsingCosineSim(TopK=num_neighbors, queryBatchSize=500, docBatchSize=100, useTest = False)

    def load_datasets(self):
        raise NotImplementedError

    def load_datasets_ng20text(self):
        raise NotImplementedError

    def GetTopK_UsingCosineSim(self, TopK, queryBatchSize, docBatchSize, useTest = False):
        raise NotImplementedError

    def get_loaders(self, num_trees, alpha, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        raise NotImplementedError

class LabeledDocuments(Data):
    def __init__(self, file_path, num_neighbors):
        super().__init__(file_path=file_path, num_neighbors=num_neighbors)

    def load_datasets(self):
        dataset = scipy.io.loadmat(self.file_path)

        # (num documents) x (vocab size) tensors containing tf-idf values
        self.X_train = dataset['train'].toarray()
        self.X_val = dataset['cv'].toarray()
        self.X_test = dataset['test'].toarray()

        # (num documents) x (num labels) tensors containing {0,1}
        self.Y_train = dataset['gnd_train']
        self.Y_val = dataset['gnd_cv']
        self.Y_test = dataset['gnd_test']

        self.vocab_size = self.X_train.shape[1]
        self.num_labels = self.Y_train.shape[1]
        print("train num:", self.X_train.shape[0], "val num:", self.X_val.shape[0], "test num:", self.X_test.shape[0], "vocab size:", self.vocab_size)

    def get_loaders(self, num_trees, alpha, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        self.edges = self.get_spanning_trees(num_trees, alpha)
        self.num_nodes = self.X_train.shape[0]
        self.num_edges = self.edges.shape[0]

        train_dataset = ds.GeneratorDataset(source=TrainGenerator(self.X_train, self.Y_train, self.edges),
                                             column_names=["data", "label", "edges1", "edges2", "weight"], shuffle=True)
        database_dataset = ds.GeneratorDataset(source=TestGenerator(self.X_train, self.Y_train), column_names=["data", "label"], shuffle=False)
        val_dataset = ds.GeneratorDataset(source=TestGenerator(self.X_val, self.Y_val), column_names=["data", "label"], shuffle=False)
        test_dataset = ds.GeneratorDataset(source=TestGenerator(self.X_test, self.Y_test), column_names=["data", "label"], shuffle=False)

        train_loader = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
        database_loader = database_dataset.batch(batch_size=batch_size, drop_remainder=True)
        val_loader = val_dataset.batch(batch_size=batch_size, drop_remainder=True)
        test_loader = test_dataset.batch(batch_size=batch_size, drop_remainder=True)
        return train_loader, database_loader, val_loader, test_loader

    def GetTopK_UsingCosineSim(self, TopK, queryBatchSize, docBatchSize, useTest = False):
        documents = deepcopy(self.X_train)
        queries = deepcopy(self.X_test) if useTest else deepcopy(self.X_train)
        Y_documents = deepcopy(self.Y_train)
        Y_queries = deepcopy(self.Y_test) if useTest else deepcopy(self.Y_train)

        # normalize 
        documents = documents / np.linalg.norm(documents, axis=-1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=-1, keepdims=True)

        # compute cosine similarity
        cos_sim_scores = matmul(Tensor(queries.astype(np.float32)), transpose(Tensor(documents.astype(np.float32)), (1,0)))
        
        if useTest: TopK = 100
        scores, indices = topk(cos_sim_scores, TopK+1)
        self.topK_scores = scores[:, 1: ].asnumpy()
        self.topK_indices = indices[:, 1: ].asnumpy()

        # test 
        if useTest:
            print("test Top100 accuracy: {:.4f}".format(np.mean(np.sum(np.repeat(np.expand_dims(Y_queries, axis=1), TopK, axis=1) * Y_documents[self.topK_indices], axis=-1) > 0)))
            exit()
        else:
            print("graph (K={:d}) accuracy: {:.4f}".format(TopK, np.mean(np.sum(np.repeat(np.expand_dims(Y_queries, axis=1), TopK, axis=1) * Y_documents[self.topK_indices], axis=-1) > 0)))
        
        del documents, queries, Y_documents, Y_queries

    def get_spanning_trees(self, num_trees, alpha):
        edges = self.topK_indices
        edges_scores = softmax(self.topK_scores / alpha, axis=-1)

        N = edges.shape[0]
        w_m = {}
        for _ in range(num_trees):
            visited = np.array([False for i in range(N)])
            while False in visited:                                      
                init_node = np.random.choice(np.where(visited == False)[0], 1)[0]
                visited[init_node] = True
                queue = [init_node]
                while len(queue) > 0:
                    now = queue[0]
                    visited[now] = True
                    edge_idx = np.where(visited[edges[now]] == False)[0]
                    if len(edge_idx) == 0:
                        queue.pop(-1)
                        break
                    next = np.random.choice(edges[now][edge_idx], 1, p=edges_scores[now][edge_idx] / np.sum(edges_scores[now][edge_idx]))[0]
                    visited[next] = True
                    queue.append(next)
                    if (now * N + next) not in w_m:
                        w_m[now * N + next] = 1
                    else:
                        w_m[now * N + next] += 1
        
        edges = [[key // N, key % N, val / num_trees] for key, val in w_m.items()]
        np.random.shuffle(edges)
        return np.array(edges)
    

class TrainGenerator():
    def __init__(self, data, labels, edges):
        self.data = data
        self.labels = labels
        self.edges = edges
        
        self.edge_idx = 0

    def __getitem__(self, index):
        if self.edge_idx >= len(self.edges):
            self.edge_idx = 0
        text = self.data[index]
        labels = self.labels[index]
        edge1 = self.data[int(self.edges[self.edge_idx][0])]
        edge2 = self.data[int(self.edges[self.edge_idx][1])]
        weight = self.edges[self.edge_idx][2]
        self.edge_idx += 1
        return text.astype(np.float32), labels.astype(np.float32), edge1.astype(np.float32),\
                edge2.astype(np.float32), weight.astype(np.float32)

    def __len__(self):
        return len(self.data)

class TestGenerator():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.float32)
    
    def __len__(self):
        return len(self.data)
