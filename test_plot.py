import numpy as np
import matplotlib.pyplot as plt

# Создаем массив значений x от -2π до 2π с шагом 0.01
x = np.arange(-2 * np.pi, 2 * np.pi, 0.01)

# Вычисляем значения y = sin(x)
y = np.sin(x)

# Создаем фигуру и оси
plt.figure(figsize=(10, 6))

# Строим график
plt.plot(x, y, linewidth=2, color='blue', label='y = sin(x)')

# Настраиваем оси и сетку
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Горизонтальная линия y=0
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)  # Вертикальная линия x=0
plt.grid(True, alpha=0.3)  # Сетка с прозрачностью

# Настраиваем подписи осей
plt.xlabel('x (радианы)', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('График функции y = sin(x)', fontsize=14, fontweight='bold')

# Настраиваем метки на осях x
pi_ticks = [-2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2, 0, 
            np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
pi_labels = ['-2π', '-3π/2', '-π', '-π/2', '0', 
             'π/2', 'π', '3π/2', '2π']
plt.xticks(pi_ticks, pi_labels)

# Ограничиваем диапазон осей
plt.xlim(-2*np.pi, 2*np.pi)
plt.ylim(-1.5, 1.5)

# Добавляем легенду
plt.legend(loc='upper right', fontsize=10)

# Добавляем точки в характерных местах
special_x = [-2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2, 0, 
             np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
special_y = np.sin(special_x)
plt.scatter(special_x, special_y, color='red', s=30, zorder=5)

# Показываем график
plt.tight_layout()
plt.show()

# Дополнительно: выводим значения в характерных точках
print("Значения функции в характерных точках:")
for x_val, y_val in zip(special_x, special_y):
    print(f"sin({x_val:6.2f}) = {y_val:6.2f}")