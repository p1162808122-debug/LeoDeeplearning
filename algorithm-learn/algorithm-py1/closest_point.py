import math


# 计算两点间的欧几里得距离
def dist(p1, p2):
    return math.dist(p1, p2)


# 跨区检查：只在条带中找最近点对
def strip_closest(strip, d):
    min_d = d
    strip.sort(key=lambda x: x[1])  # 按 y 排序
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j][1] - strip[i][1]) < min_d:
            min_d = min(min_d, dist(strip[i], strip[j]))
            j += 1
    return min_d


# 分治递归
def closest_util(points):
    n = len(points)
    if n <= 3:  # 小规模直接暴力求解
        return min(dist(points[i], points[j]) for i in range(n) for j in range(i + 1, n))

    mid = n // 2
    mid_point = points[mid]

    # 左右递归
    dl = closest_util(points[:mid])
    dr = closest_util(points[mid:])
    d = min(dl, dr)

    # 构建跨区带
    strip = [p for p in points if abs(p[0] - mid_point[0]) < d]

    # 返回综合结果
    return min(d, strip_closest(strip, d))


def closest_pair(points):
    points.sort(key=lambda x: x[0])  # 按 x 排序
    return closest_util(points)


# 测试
points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
print("最近点对距离:", closest_pair(points))
