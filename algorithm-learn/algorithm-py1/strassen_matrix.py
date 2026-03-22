import math

# ---------- 基础运算 ----------

def add_matrix(A, B):
    """矩阵加法"""
    n = len(A)
    C = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(A[i][j] + B[i][j])
        C.append(row)
    return C

def sub_matrix(A, B):
    """矩阵减法"""
    n = len(A)
    C = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(A[i][j] - B[i][j])
        C.append(row)
    return C

def naive_mult(A, B):
    """普通 O(n^3) 矩阵乘法"""
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]#生成一个n*n的0矩阵
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# ---------- 分割与合并 ----------

def split_matrix(A):
    """把矩阵分割成四个子块"""
    n = len(A)
    m = n // 2
    A11, A12, A21, A22 = [], [], [], []
    for i in range(m):
        A11.append(A[i][:m])
        A12.append(A[i][m:])
    for i in range(m, n):
        A21.append(A[i][:m])
        A22.append(A[i][m:])
    return A11, A12, A21, A22

def join_quadrants(C11, C12, C21, C22):
    """合并四个子块"""
    m = len(C11)
    C = []
    for i in range(m):
        C.append(C11[i] + C12[i])#这里是按照行操作的
    for i in range(m):
        C.append(C21[i] + C22[i])
    return C

# ---------- 填充与裁剪 ----------

def next_power_of_two(n):
    p = 1
    while p < n:
        p *= 2
    return p

def pad_matrix(A, size):
    """把矩阵填充到 size x size"""
    n, m = len(A), len(A[0])
    padded = []
    for i in range(size):
        row = []
        for j in range(size):
            if i < n and j < m:
                row.append(A[i][j])
            else:
                row.append(0)
        padded.append(row)
    return padded

def unpad_matrix(A, rows, cols):
    """裁剪回原始大小"""
    return [row[:cols] for row in A[:rows]]

# ---------- Strassen 算法 ----------

THRESHOLD = 2  # 小于等于这个规模时用普通乘法

def strassen_recursive(A, B):
    n = len(A)
    if n <= THRESHOLD:
        return naive_mult(A, B)

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    # 七个中间矩阵
    M1 = strassen_recursive(add_matrix(A11, A22), add_matrix(B11, B22))
    M2 = strassen_recursive(add_matrix(A21, A22), B11)
    M3 = strassen_recursive(A11, sub_matrix(B12, B22))
    M4 = strassen_recursive(A22, sub_matrix(B21, B11))
    M5 = strassen_recursive(add_matrix(A11, A12), B22)
    M6 = strassen_recursive(sub_matrix(A21, A11), add_matrix(B11, B12))
    M7 = strassen_recursive(sub_matrix(A12, A22), add_matrix(B21, B22))

    # 组合结果
    C11 = add_matrix(sub_matrix(add_matrix(M1, M4), M5), M7)
    C12 = add_matrix(M3, M5)
    C21 = add_matrix(M2, M4)
    C22 = add_matrix(add_matrix(sub_matrix(M1, M2), M3), M6)

    return join_quadrants(C11, C12, C21, C22)

def strassen(A, B):
    """外层接口，支持非 2^k 阶方阵"""
    n, m = len(A), len(B[0])
    p = len(A[0])
    if p != len(B):
        raise ValueError("矩阵维度不匹配")

    size = next_power_of_two(max(n, m, p))
    A_pad = pad_matrix(A, size)
    B_pad = pad_matrix(B, size)

    C_pad = strassen_recursive(A_pad, B_pad)
    C = unpad_matrix(C_pad, n, m)
    return C

# ---------- 示例 ----------
A = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

B = [
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
]

C = strassen(A, B)

print("A * B =")
for row in C:
    print(row)
