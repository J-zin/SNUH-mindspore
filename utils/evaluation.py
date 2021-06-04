import datetime
import numpy as np
import mindspore.ops as ops
import mindspore.numpy as mindnp
transpose = ops.Transpose()
matmul = ops.MatMul()
topk = ops.TopK(sorted=True)

def compute_retrieval_precision(train_loader, eval_loader,
                                encode_discrete=None, distance_metric='hamming',
                                num_retrieve=100, num_features=128):
    def extract_data(loader):
        encoding_chunks = []
        label_chunks = []
        for (docs, labels) in loader:
            encoding_chunks.append(docs if encode_discrete is None else
                                   encode_discrete(docs))
            label_chunks.append(labels)

        encoding_mat = mindnp.concatenate(encoding_chunks, axis=0)
        label_mat = mindnp.concatenate(label_chunks, axis=0).asnumpy()
        label_lists = [[j for j in np.nonzero(label_mat[i])[0]] for i in
                       range(label_mat.shape[0])]
        return encoding_mat, label_lists

    src_encodings, src_label_lists = extract_data(train_loader)
    tgt_encodings, tgt_label_lists = extract_data(eval_loader)

    prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                          src_encodings, src_label_lists,
                                          num_retrieve, distance_metric, num_features)
    return prec

def compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                   src_encodings, src_label_lists,
                                   num_retrieve, distance_metric='hamming',
                                   chunk_size=100, binary=True, num_features=128):
    K = min(num_retrieve, len(src_encodings))
    D = compute_distance(tgt_encodings, src_encodings, distance_metric,
                         chunk_size, binary)
    _, list_topK_nearest_indices = topk(num_features-D, K)

    average_precision = 0.
    for i, topK_nearest_indices in enumerate(list_topK_nearest_indices.asnumpy()):
        gold_set = set(tgt_label_lists[i])
        candidate_lists = [src_label_lists[j] for j in topK_nearest_indices]
        precision = len([_ for candidates in candidate_lists
                         if not gold_set.isdisjoint(candidates)]) / K * 100
        average_precision += precision / tgt_encodings.shape[0]
    return average_precision

def compute_distance(X1, X2, distance_metric='hamming', chunk_size=1000,
                     binary=True):
    if distance_metric == 'hamming':
        D = compute_hamming_distance(X1, X2, chunk_size=chunk_size,
                                     binary=binary)
    else:
        raise Exception('Unsupported distance: {0}'.format(distance_metric))
    return D

def compute_hamming_distance(X1, X2, chunk_size=100, binary=True):
    assert X1.shape[1] == X2.shape[1]

    D = []
    for i in range(0, X1.shape[0], chunk_size):
        X1_chunk = X1[i:i + chunk_size]
        
        A = matmul((1 - X1_chunk), transpose(X2, (1,0)))  # X2 one, X1_chunk zero
        B = matmul(X1_chunk, transpose((1 - X2), (1,0)))  # X1_chunk one, X2 zero
        D.append(A + B)
    
    return mindnp.concatenate(D, axis=0)  # N x M