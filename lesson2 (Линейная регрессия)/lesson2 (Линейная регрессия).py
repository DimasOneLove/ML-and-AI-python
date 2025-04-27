# Линейная регрессия

# Задача: на основе наблюдаемых точек (имеющихся данных) построить прямую,
# которая отображает связь между двумя и более переменными
# Регрессия пытается подогнать функцию к наблюдаемым данным, чтобы спрогнозировать новые данные
# Линейная регрессия подгоняет данные к прямой линии, пытаемся установить линейную связь
# между переменными и предсказать новые данные

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr
import random

# features, target = make_regression(
#     n_samples = 100, n_features = 1, n_informative = 1, n_targets = 1, noise = 15, random_state =2
# )
#
# # print(features)
# print(features.shape)
# print(target.shape)
#
# model = LinearRegression().fit(features, target)
#
# plt.scatter(features, target)
#
# # предсказание:
# x = np.linspace(features.min(), features.max(), 100)
# # y = kx + b
# plt.plot(x, model.coef_[0] * x + model.intercept_, color = 'red')
# plt.show()
#

### Простая линейная регрессия

# Линейная -> линейная зависимость

# + плюсы:
#     1)Прогнозирование на новых данных
#     2)Анализ взаимного влияния переменных друг на друга

# - минусы:
#     1)точки обучаемых данных НЕ будут точно лежать на прямой (из-за шума) => область погрешности
#     2)НЕ позволяет делать прогнозы ВНЕ диапазона имеющихся данных

# Данные, на основе которых строится модель, — это выборка из совокупности, хотелось бы чтобы это была РЕПРЕЗЕНТАТИВНАЯ выборка

data = np.array(
    [
        [1, 4],
        [2, 9],
        [3, 12],
        [4, 5],
        [5, 8],
        [6, 9],
        [7, 16],
        [8, 19],
        [9, 22],
        [10, 27],
    ]
)

## Аналитическое решение
# 1. Формульное для простой регрессии
x = data[:,0]
y = data[:,1]

n = len(x)

w_1 = (n*sum(x[i]*y[i] for i in range(n)) - sum(x[i] for i in range(n)) * sum(y[i] for i in range(n))
)/ (n*sum(x[i]**2 for i in range(n)) - sum(x[i] for i in range(n))**2)

w_0 = (sum(y[i] for i in range(n)) / n) - w_1 * (sum(x[i] for i in range(n))) / n

print(w_1, w_0)
# 2.224242424242424 0.8666666666666671

# 2. Метод обратных матриц
x_1 = np.vstack([x, np.ones(len(x))]).T
print(x_1)
w = inv(x_1.transpose() @ x_1) @ (x_1.transpose() @ y)
print(w)
#[2.22424242 0.86666667]

# 3. Разложение матриц - QR
# В случае приближенных вычислений позволяет минимизировать вычислительную ошибку
Q, R = qr(x_1)
w = inv(R).dot(Q.transpose()).dot(y)
print(w)
# [2.22424242 0.86666667]

# 4. Градиентный спуск
# Метод оптимизации, используются производные и итерации
# Два параметра -> рассматриваем частные производные по одному из параметров
# Позволяет определить угловой коэф и изменение параметра
# Выполняется в ту сторону, где он макс/мин
# Для бОльших угловых коэф-в делается более широкий шаг, для маленьких - более узкий
# Ширина шага обычно вычисляется как доля от углового коэф-та
# Напрямую связан со скоростью обучения. Чем выше скорость, тем быстрее будет работать система, но делается это засчет снижения точности
# Чем ниже скорость, тем больше времени займет обучение, но точность будет выше

# def f(x):
#     return (x-3)**2 + 4
#
# def df(x):
#     return 2*(x-3)
#
# x = np.linspace(-10, 10, 100)
# ax = plt.gca()
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
#
# plt.plot(x, df(x))
# plt.plot(x, f(x))
#
# plt.grid()
# plt.show()

# # Зададим начальные значения коэф-тов
w1 = 0.0
w0 = 0.0

L = 0.001 # Шаг, ~Скорость обучения
iterations = 100_000

# Берем частные производные функции потерь и обновляем значение коэф-в w0 w1
for i in range(iterations):
    D_w0 = 2 * sum(-y[i] + w0 + w1 * x[i] for i in range(n))
    D_w1 = 2 * sum((x[i] * (-y[i] + w0 + w1 * x[i])) for i in range(n))
    w1 -= L * D_w1
    w0 -= L * D_w0

print(w1, w0)
# 2.224242424242427 0.8666666666666504

## Рассмотрим функцию потерь:

w1 = np.linspace(-10, 10, 100)
w0 = np.linspace(-10,10,100)

def E(w1, w0, x, y):
    return sum((y[i] - (w0 + w1 * x[i]))**2 for i in range(len(x)))

W1, W0 = np.meshgrid(w1,w0)
EW = E(W1, W0, x, y)

fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.plot_surface(W1,W0, EW)

w1_fit = 2.224242424242424
w0_fit = 0.86666667

E_fit = E(w1_fit, w0_fit, x, y)

ax.scatter3D(w1_fit,w0_fit, E_fit, color = 'red')
# По градиентному методу минимизировали функцию потерь и нашли точку - w1 w0

plt.show()