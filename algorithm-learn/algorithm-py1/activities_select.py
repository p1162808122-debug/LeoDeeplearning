def activity_selection(start, finish):
    """
    使用贪心算法解决活动选择问题
    :param start: 活动开始时间列表
    :param finish: 活动结束时间列表
    :return: 最大兼容活动集的索引列表
    """
    n = len(start)

    # 按结束时间升序排序（同时保持开始时间的对应关系）
    # 这里使用 zip 将 start 和 finish 打包，并按 finish 排序
    activities = sorted(zip(start, finish, range(n)), key=lambda x: x[1])

    selected = []
    last_finish = 0  # 初始化上一个选中活动的结束时间

    for s, f, idx in activities:
        if s >= last_finish:  # 如果当前活动的开始时间 >= 上一个活动的结束时间
            selected.append(idx)  # 选择该活动
            last_finish = f  # 更新结束时间

    return selected


# 测试用例
if __name__ == "__main__":
    # 示例 1: 经典案例（已按结束时间排序）
    start_times = [1, 3, 0, 5, 3, 5, 6, 8, 8, 2, 12]
    finish_times = [4, 5, 6, 7, 9, 9, 10, 11, 12, 14, 16]

    print("活动列表（编号 | 开始时间 | 结束时间）:")
    for i, (s, f) in enumerate(zip(start_times, finish_times)):
        print(f"活动 {i}: {s} -> {f}")

    result = activity_selection(start_times, finish_times)
    print("\n最大兼容活动集的编号:", result)

    # 示例 2: 未排序的案例
    start_times = [5, 1, 3, 0, 8, 5]
    finish_times = [9, 2, 4, 6, 11, 7]

    print("\n未排序的活动列表:")
    for i, (s, f) in enumerate(zip(start_times, finish_times)):
        print(f"活动 {i}: {s} -> {f}")

    result = activity_selection(start_times, finish_times)
    print("\n最大兼容活动集的编号:", result)