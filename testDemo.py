import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def huizhidenggaoxian(s1, s2):
    # 定义两个变量的范围
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    # 创建网格
    X, Y = np.meshgrid(x, y)

    # 定义简单函数 z = x^2 + y^2
    Z = s1 * X**2 + s2 * Y**2

    # 创建一个包含两个子图的画布
    fig = plt.figure(figsize=(12, 6))

    # 左边的三维图像
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap="plasma", edgecolor="none")
    ax1.set_title("3D plot of $z = {}x^2 + {}y^2$".format(s1, s2))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    # 右边的等高线图
    ax2 = fig.add_subplot(1, 2, 2)
    cp = ax2.contour(X, Y, Z, levels=10, cmap="plasma")  # 使用 10 条等高线
    ax2.clabel(cp, inline=True, fontsize=8)  # 显示等高线上的标签
    fig.colorbar(cp, ax=ax2)  # 添加颜色条
    ax2.set_title("Contour plot of $z = {}x^2 + {}y^2$".format(s1, s2))
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    # 显示图像
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # huizhidenggaoxian(s1=1, s2=1)
    huizhidenggaoxian(s1=1, s2=3)
