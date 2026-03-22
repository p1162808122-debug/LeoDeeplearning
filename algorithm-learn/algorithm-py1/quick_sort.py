def quick_sort(arr):
    # 基本情况：空数组或只有一个元素时，直接返回
    if len(arr) <= 1:
        return arr

    # 1. 选择基准（这里选择最后一个元素）
    pivot = arr[-1]

    # 2. 分区
    left = [x for x in arr[:-1] if x <= pivot]  # 比 pivot 小的放左边
    right = [x for x in arr[:-1] if x > pivot]  # 比 pivot 大的放右边

    # 3. 递归排序，并合并结果
    return quick_sort(left) + [pivot] + quick_sort(right)


# 测试
arr = [38, 27, 43, 3, 9, 82, 10]
print("排序结果:", quick_sort(arr))
