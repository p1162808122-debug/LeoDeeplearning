def merge_sort(arr):
    # 基本情况：只有一个元素或空时，直接返回
    if len(arr) <= 1:
        return arr

    # 1. 分解
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])   # 左边递归排序
    right = merge_sort(arr[mid:])  # 右边递归排序

    # 2. 合并
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0

    # 两个有序数组合并
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # 拼接剩余的元素
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# 测试
arr = [38, 27, 43, 3, 9, 82, 10]
print("排序结果:", merge_sort(arr))
