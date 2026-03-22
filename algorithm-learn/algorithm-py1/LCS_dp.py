def LCS(X, Y):
    m, n = len(X), len(Y)
    # 初始化二维表格
    dp = [[0] * (n+1) for _ in range(m+1)]# 创建一个 (m+1) x (n+1) 的二维 dp 数组

    # 填表
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:        # 字符相等
                dp[i][j] = dp[i-1][j-1] + 1             #有相等的就加一
            else:                       # 字符不等
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])  #上一排和左边的最大值

    # dp[m][n] 是 LCS 的长度
    return dp[m][n]

# 测试
X = "ABCBDAB"
Y = "BDCABA"
print("最长公共子序列长度:", LCS(X, Y))
