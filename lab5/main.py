import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.patches import Ellipse

def quadrant_correlation(x, y):
    x_med = np.median(x)
    y_med = np.median(y)
    n = len(x)
    q = np.sum(np.sign(x - x_med) * np.sign(y - y_med))
    return q / n

def generate_data(n, rho, is_mixture=False):
    if not is_mixture:
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        return np.random.multivariate_normal(mean, cov, n)
    else:
        # Смесь распределений
        n1 = int(0.9 * n)
        n2 = n - n1
        
        # Первая компонента
        mean1, cov1 = [0, 0], [[1, 0.9], [0.9, 1]]
        data1 = np.random.multivariate_normal(mean1, cov1, n1)
        
        # Вторая компонента (выбросы)
        mean2, cov2 = [0, 0], [[100, -90], [-90, 100]] # sigma=10 -> sigma^2=100
        data2 = np.random.multivariate_normal(mean2, cov2, n2)
        
        return np.vstack([data1, data2])

def draw_ellipse(x, y, ax):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Парамы эллипса
    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((np.mean(x), np.mean(y)), 
                      width=rad_x * 2, height=rad_y * 2,
                      angle=45, facecolor='none', edgecolor='red', linewidth=2)
    ax.add_patch(ellipse)


sizes = [20, 60, 100]
rhos = [0, 0.5, 0.9]
repeats = 1000

for n in sizes:
    print(f"\n=================Результаты для n = {n}=================")
    
    # Проходим по обычным распределениям и смеси
    tasks = [0, 0.5, 0.9, "mixture"]
    
    for task in tasks:
        p_list, s_list, q_list = [], [], []
        
        for _ in range(repeats):
            if task == "mixture":
                data = generate_data(n, 0, is_mixture=True)
            else:
                data = generate_data(n, task, is_mixture=False)
            
            x, y = data[:, 0], data[:, 1]
            
            #коэфы
            p_val, _ = pearsonr(x, y)
            s_val, _ = spearmanr(x, y)
            q_val = quadrant_correlation(x, y)
            
            p_list.append(p_val)
            s_list.append(s_val)
            q_list.append(q_val)
            
        print(f"\nТип данных: {task}")
        print(f"Пирсон:  Среднее = {np.mean(p_list):.3f}, Дисперсия = {np.var(p_list):.3f}")
        print(f"Спирмен: Среднее = {np.mean(s_list):.3f}, Дисперсия = {np.var(s_list):.3f}")
        print(f"Квадр.:  Среднее = {np.mean(q_list):.3f}, Дисперсия = {np.var(q_list):.3f}")
    print("====================================\n")


fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for i, n in enumerate(sizes):
    data = generate_data(n, 0, is_mixture=True)
    x, y = data[:, 0], data[:, 1]
    
    axs[i].scatter(x, y, s=15, alpha=0.9, edgecolors='black', linewidth=0.5)
    
    draw_ellipse(x, y, axs[i])
    
    axs[i].set_title(f"Смесь распределений (n = {n})")
    axs[i].set_xlabel("X")
    axs[i].set_ylabel("Y")
    axs[i].grid(True, linestyle='--', alpha=0.6)
    

    axs[i].set_xlim(-5, 5)
    axs[i].set_ylim(-5, 5)
    
    if i == 0:
        axs[i].legend()

plt.show()
#export QT_QPA_PLATFORM=wayland
#python lab5/main.py