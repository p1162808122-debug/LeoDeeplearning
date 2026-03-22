def knapsack_01(weights, values, capacity):
    """
    0-1背包问题动态规划解法
    :param weights: 物品重量列表
    :param values: 物品价值列表
    :param capacity: 背包容量
    :return: 最大价值，选择的物品索引列表
    """
    n = len(weights)
    # 初始化动态规划表：dp[i][j] 表示前i个物品在容量j下的最大价值
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # 填表
    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i - 1] <= j:  # 当前物品可以装入背包
                dp[i][j] = max(
                    dp[i - 1][j],  # 不拿当前物品
                    values[i - 1] + dp[i - 1][j - weights[i - 1]]  # 拿当前物品
                )
            else:
                dp[i][j] = dp[i - 1][j]  # 当前物品太重，无法装入

    # 回溯找出选择的物品
    selected_items = []
    j = capacity
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:  # 说明拿了第i-1个物品（因为i从1开始，物品索引是i-1）
            selected_items.append(i - 1)
            j -= weights[i - 1]

    return dp[n][capacity], selected_items[::-1]  # 返回最大价值和物品索引（按原始顺序）


# 示例测试
if __name__ == "__main__":
    weights = [1, 2, 3]  # 物品重量
    values = [6, 10, 12]  # 物品价值
    capacity = 5  # 背包容量

    max_value, selected_items = knapsack_01(weights, values, capacity)
    print(f"最大价值: {max_value}")
    print(f"选择的物品索引: {selected_items}")
    print(f"选择的物品重量: {[weights[i] for i in selected_items]}")
    print(f"选择的物品价值: {[values[i] for i in selected_items]}")