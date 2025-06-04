from functools import reduce
from math import factorial
import numpy as np
from matplotlib import pyplot as plt


# Построение многочлена Лагранжа
def lagrange_polynomial(xs, ys, n):
    return lambda x: sum([
        ys[i] * reduce(
            lambda a, b: a * b,
                        [(x - xs[j]) / (xs[i] - xs[j])
            for j in range(n) if i != j])
        for i in range(n)])


# Вычисление разделённых разностей (коэффициенты Ньютона)
def divided_differences(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])

    return coef


# Таблица разделенных разностей
def divided_differences_table(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            numerator = table[i + 1][j - 1] - table[i][j - 1]
            denominator = x[i + j] - x[i]
            table[i][j] = numerator / denominator

    return table


# Вывод таблицы разделенных разностей
def print_divided_differences_table(table):
    n = table.shape[0]
    print("Таблица разделённых разностей:")
    for i in range(n):
        row = []
        for j in range(n):
            if j + i < n:
                row.append(f"{table[i, j]:.6f}")
            else:
                row.append("")
        print("\t".join(row))


# Построение многочлена Ньютона по разделённым разностям
def newton_divided_difference_polynomial(xs, ys, n):
    coef = divided_differences(xs, ys)

    # print_divided_differences_table(divided_differences_table(xs, ys))

    return lambda x: ys[0] + sum([
        coef[k] * reduce(lambda a, b: a * b, [x - xs[j] for j in range(k)]) for k in range(1, n)
    ])


# Таблица конечных разностей
def finite_differences(y):
    n = len(y)
    delta_y = np.zeros((n, n))
    delta_y[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            delta_y[i,j] = delta_y[i+1,j-1] - delta_y[i,j-1]
    return delta_y


# Вывод таблицы конечных разностей
def print_finite_differences_table(delta_y):
    n = delta_y.shape[0]
    print("Таблица конечных разностей:")
    for i in range(n):
        row = [f"{delta_y[i, j]:.4f}" if i + j < n else "" for j in range(n)]
        print("\t".join(row))


# Многочлен Ньютона по конечным разностям (равномерная сетка)
def newton_interpolation(xs, ys, n):
    h = xs[1] - xs[0]

    def forward(x):
        diffs = finite_differences(ys)
        t = (x - xs[0]) / h
        result = ys[0]
        mult = 1

        for i in range(1, n):
            mult *= (t - (i - 1))
            result += (mult / factorial(i)) * diffs[0][i]
        return result

    def backward(x):
        diffs = finite_differences(ys)
        t = (x - xs[-1]) / h
        result = ys[-1]
        mult = 1

        for i in range(1, n):
            mult *= (t + (i - 1))
            result += (mult / factorial(i)) * diffs[n - i - 1][i]
        return result

    # Возвращаем лямбду, которая выбирает forward/backward динамически
    return lambda x: forward(x) if x < (xs[0] + xs[-1]) / 2 else backward(x)


# Вспомогательная функция для отрисовки графиков
def draw_plot(a, b, func, name, dx=0.001):
    xs, ys = [], []
    a -= dx
    b += dx
    x = a
    while x <= b:
        xs.append(x)
        ys.append(func(x))
        x += dx
    plt.plot(xs, ys, 'g', label=name)


# Основная функция решения: построение таблицы, вычисления, графики
def solve(xs, ys, x, n):
    delta_y = finite_differences(ys)
    print_finite_differences_table(delta_y)
    print()
    print_divided_differences_table(divided_differences_table(xs, ys))

    print('\n' + '-' * 60)

    methods = [("Многочлен Лагранжа", lagrange_polynomial),
               ("Многочлен Ньютона с разделенными разностями", newton_divided_difference_polynomial),
               ("Многочлен Ньютона с конечными разностями", newton_interpolation)]

    for name, method in methods:
        finite_difference = True
        last = xs[1] - xs[0]
        for i in range(1, n):
            new = abs(xs[i] - xs[i - 1])
            if abs(new - last) > 0.0001:
                finite_difference = False
            last = new

        if method is newton_interpolation and not finite_difference:
            continue

        # h = xs[1] - xs[0]
        # alpha_ind = n // 2
        # t = (x - xs[alpha_ind]) / h
        # print("t: ", t)

        print(name)
        P = method(xs, ys, n)
        print(f'P({x}) = {P(x)}')
        print('-' * 60)

        plt.title(name)
        draw_plot(xs[0], xs[-1], P, name)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.scatter(x, P(x), c='r')
        for i in range(len(xs)):
            plt.scatter(xs[i], ys[i], c='b')

        plt.show()