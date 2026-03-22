def max_crossing_sum(nums, left, mid, right):
    # 计算跨中点的最大子数组和
    left_sum = float('-inf')#表示设置为负无穷大
    s = 0
    for i in range(mid, left-1, -1):   # 从中点往左扫描
        s += nums[i]
        left_sum = max(left_sum, s)

    right_sum = float('-inf')
    s = 0
    for i in range(mid+1, right+1):   # 从中点往右扫描
        s += nums[i]
        right_sum = max(right_sum, s)

    return left_sum + right_sum


def max_subarray(nums, left, right):
    # 递归边界
    if left == right:
        return nums[left]

    mid = (left + right) // 2

    # 三种情况：左边最大、右边最大、跨越中点最大
    left_max = max_subarray(nums, left, mid)
    right_max = max_subarray(nums, mid+1, right)
    cross_max = max_crossing_sum(nums, left, mid, right)

    return max(left_max, right_max, cross_max)


# 调用示例
nums = [-2,1,-3,4,-1,2,1,-5,4]
print(max_subarray(nums, 0, len(nums)-1))  # 输出 6
