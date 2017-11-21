import sys
import numpy as np
from scipy import sparse


def dense_to_sparse(dense_model, sparse_model):
    dense = np.load(dense_model)[()]
    sp_data = {}
    for key, value in dense.items():
        if 'weights' in key:
            sp_data[key] = sparse.csr_matrix(value.ravel())
        else:
            sp_data[key] = value

    np.save(sparse_model, sparse)


def main(args):
    dense_to_sparse(
        dense_model=args[0], sparse_model=args[1])

if __name__ == '__main__':
    main(sys.argv[1:])
