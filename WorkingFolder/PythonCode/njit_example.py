
from time import time
from numba import njit, float64, int64, boolean
import numba as nb


def fill_matrx(x):
    matrix = np.empty_like(x)
    N_1,N_2 = x.shape
    for i in range(N_1):
        for j in range(N_2):
            matrix[i,j] =np.random.randint(1)

@njit
def fill_matrx_fast(x):
    matrix = np.empty_like(x)
    N_1,N_2 = x.shape
    for i in range(N_1):
        for j in range(N_2):
            matrix[i,j] =np.random.randint(1)


## compare
N_1 = 4840
N_2 = 4840

x = np.zeros((N_1, N_2), dtype=np.int64)


t_start = time()
mat1 = fill_matrx(x)
t_finish = time()
t1 = t_finish-t_start


t_start = time()
mat2 = fill_matrx_fast(x)
t_finish = time()
t2 = t_finish-t_start

print(str(t1/t2))
print(t1)

print(t2)
