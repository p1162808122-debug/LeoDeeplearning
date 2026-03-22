def matrix_chain_order(p):
    """
    矩阵链乘法的动态规划解法
    :param p: 维度数组 [p0, p1, p2, ..., pn]，表示n个矩阵的维度
    :return: m表（最小代价），s表（分割点）
    """
    n = len(p) - 1  # 矩阵个数
    # 初始化DP表：m[i][j]表示计算A_i到A_j的最小代价
    m = [[0] * (n + 1) for _ in range(n + 1)]
    # s[i][j]记录A_i到A_j的最优分割位置
    s = [[0] * (n + 1) for _ in range(n + 1)]

    # l是链的长度
    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = float('inf')
            # 尝试所有可能的分割点k
            for k in range(i, j):
                # 计算代价
                cost = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k

    return m, s


def print_optimal_parens(s, i, j):
    """
    打印最优括号化方案
    """
    if i == j:
        print(f"A{i}", end="")
    else:
        print("(", end="")
        print_optimal_parens(s, i, s[i][j])
        print_optimal_parens(s, s[i][j] + 1, j)
        print(")", end="")


# 测试
if __name__ == "__main__":
    # 矩阵维度数组：A1=5x4, A2=4x6, A3=6x2, A4=2x3
    p = [5, 4, 6, 2, 3]

    m, s = matrix_chain_order(p)

    print("最小标量乘法次数:", m[1][len(p) - 1])
    print("最优括号化方案: ", end="")
    print_optimal_parens(s, 1, len(p) - 1)
    print()

    # 打印完整的m表和s表
    print("\nm表（最小代价）:")
    for i in range(1, len(m)):
        for j in range(1, len(m[0])):
            if i <= j:
                print(f"{m[i][j]:6}", end=" ")
            else:
                print("      ", end=" ")
        print()

    print("\ns表（分割点）:")
    for i in range(1, len(s)):
        for j in range(1, len(s[0])):
            if i < j:
                print(f"{s[i][j]:3}", end=" ")
            else:
                print("   ", end=" ")
        print()