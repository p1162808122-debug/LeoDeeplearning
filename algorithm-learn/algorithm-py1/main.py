# 输入一组数
nums = [3, 7, 2, 9, 5]

# 假设第一个数是最大值
max_val = nums[0]

# 遍历比较
for num in nums[1:]:
    if num > max_val:
        max_val = num

# 输出最大值
print("最大值是:", max_val)
