import numpy as np

      
def get_batch(X, Y, i, subsample_size, batch_size):
    """ Retrieves minibatch
    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        Input sparse matrix of abundances
    Y : np.array
        Output dense matrix of abundances
    i : int
        Sample index
    subsample_size : int
        Number of sequences to randomly draw per iteration
    batch_size : int
        Number of samples to load per minibatch
    TODO
    ----
    - It will be worth offloading this work to the torch.data.DataLoader class.
      One advantage is that the GPU work can be done in parallel
      to preparing the data for transfer.  This could yield at least
      a 2x speedup.
    """
    Xs = []
    Ys = []
    for n in range(batch_size):
        k = (i + n) % Y.shape[0]
        row = X.getrow(k)
        cnts = np.hstack([row.data[row.indptr[i]:row.indptr[i+1]]
                          for i in range(len(row.indptr)-1)])
        feats = np.hstack([row.indices[row.indptr[i]:row.indptr[i+1]]
                           for i in range(len(row.indptr)-1)])
        inp = np.random.choice(feats, p=cnts/np.sum(cnts), size=subsample_size)
        Xs.append(inp)
        Ys.append(Y[k])

    Xs = np.hstack(Xs)
    Ys = np.repeat(np.vstack(Ys), subsample_size, axis=0)
    return torch.from_numpy(Xs).long(), torch.from_numpy(Ys).float()
