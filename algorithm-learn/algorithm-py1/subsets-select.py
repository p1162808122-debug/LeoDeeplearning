def subsets(nums):
    result = []
    path = []

    def backtracking(start_index):
        # 每次进入递归，都意味着是一个新的子集状态，直接加入结果
        result.append(path[:]) # 必须使用副本

        # 从start_index开始遍历，避免产生重复的子集（如[1,2]和[2,1]）
        for i in range(start_index, len(nums)):
            path.append(nums[i])        # 做出选择：添加nums[i]
            backtracking(i + 1)         # 递归：从i+1开始，处理下一个元素
            path.pop()                  # 撤销选择（回溯）：移除nums[i]

    backtracking(0) # 从索引0开始
    return result

# 测试
print(subsets([1, 2, 3]))
# 输出: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]