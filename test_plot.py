import matplotlib.pyplot as plt
import numpy as np

# Задаем параметры прямой y = kx + b
k = 2  # угловой коэффициент
b = 1  # свободный член

# Создаем массив значений x
x = np.linspace(-10, 10, 100)  # от -10 до 10, 100 точек

# Вычисляем y
y = k * x + b

# Строим график
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2, label=f'$y = {k}x + {b}$')

# Настройка графика
plt.title('График прямой линии', fontsize=14)
plt.xlabel('Ось X', fontsize=12)
plt.ylabel('Ось Y', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)  # ось X
plt.axvline(x=0, color='k', linewidth=0.5)  # ось Y
plt.legend()

# Показываем график
plt.show()
