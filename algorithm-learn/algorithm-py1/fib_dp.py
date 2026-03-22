def fib(n, memo={}):
    if n in memo:         # 已经计算过
        return memo[n]
    if n <= 1:            # 边界
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)  # 递归计算并存储
    return memo[n]

print(fib(10))  # 输出第10个斐波那契数

def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)  # 表格
    dp[0], dp[1] = 0, 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

print(fib(10))  # 输出第10个斐波那契数

