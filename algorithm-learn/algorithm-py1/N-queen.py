def totalNQueens(n):
    """
    解决N皇后问题，返回解的数量
    :param n: 棋盘大小
    :return: 解的数量
    """
    def is_valid(board, row, col):
        """
        检查在 (row, col) 放置皇后是否与之前放置的皇后冲突
        """
        # 检查当前列是否有冲突
        for i in range(row):
            if board[i] == col:
                return False
            # 检查对角线：如果两点的行差 == 列差，说明在同一对角线上
            if abs(i - row) == abs(board[i] - col):
                return False
        return True

    def backtrack(row, board):
        """
        回溯函数
        :param row: 当前处理的行
        :param board: 记录每行皇后所在的列（索引从0开始）
        """
        if row == n:  # 所有行都放置完毕，找到一个解
            nonlocal count
            count += 1
            return

        # 尝试在当前行的每一列放置皇后
        for col in range(n):
            if is_valid(board, row, col):
                board[row] = col  # 放置皇后
                backtrack(row + 1, board)  # 进入下一行
                # 无需显式回溯（board[row]会被覆盖）

    count = 0
    board = [-1] * n  # 初始化棋盘，-1表示未放置
    backtrack(0, board)
    return count

# 测试不同大小的N皇后问题
for n in range(1, 10):
    print(f"{n} 皇后问题有 {totalNQueens(n)} 种解")