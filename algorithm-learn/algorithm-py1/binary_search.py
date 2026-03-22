# def binary_search(nums, target):#迭代版
#     left, right = 0, len(nums) - 1
#
#     while left <= right:
#         mid = (left + right) // 2
#         if nums[mid] == target:
#             return mid  # 找到目标
#         elif nums[mid] < target:
#             left = mid + 1  # 去右边找
#         else:
#             right = mid - 1  # 去左边找
#     return -1 # 未找到
#
#
# # 测试调用
# nums = [1, 3, 5, 7, 9, 11, 13]
# print(binary_search(nums, 7))  # 输出 3，因为 7 在 nums[3]
# print(binary_search(nums, 2))  # 输出 -1，因为 2 不在数组里

def binary_search_recursive(nums, left, right, target):
    if left > right:
        return -1  # 区间为空

    mid = (left + right) // 2
    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        return binary_search_recursive(nums, mid + 1, right, target)
    else:
        return binary_search_recursive(nums, left, mid - 1, target)


# 测试调用
nums = [1, 3, 5, 7, 9, 11, 13]
print(binary_search_recursive(nums, 0, len(nums) - 1, 7))  # 输出 3
print(binary_search_recursive(nums, 0, len(nums) - 1, 2))  # 输出 -1

