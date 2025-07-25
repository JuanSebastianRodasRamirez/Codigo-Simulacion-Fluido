import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Tamaño malla
nx, ny = 200, 20
h = 1.0  # Tamaño de la malla

# Inicializar la malla
vy_vals = np.full((ny, nx), 0.001)
vx_vals = np.zeros((ny, nx))

print(vy_vals)
#y_vals = np.zeros((ny, nx))
# Primera columna
for j in range(1, ny - 1):  # Desde l fila 1 hasta ny - 1
    vx_vals[j, 0] = 1

# Centro de la malla
for j in range(1, ny - 1):  # Filas internas
    for i in range(1, nx - 1):  # Columnas internas
        vx_vals[j, i] = 1

# Mapeo por fila para nodos interiores
def idx(i, j, ny):
    return (i - 1) * (ny - 2) + (j - 1)

# Calcular el residuo F para toda la grilla
def Fx(vx, vy, h, nx, ny):
    F = np.zeros((ny - 2, nx - 2))
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            term = vx[j, i+1] + vx[j, i-1] + vx[j+1, i] + vx[j-1, i]
            conv1 = (h / 2.0) * vx[j, i] * (vx[j, i+1] - vx[j, i-1])
            conv2 = (h / 2.0) * vy[j, i] * (vx[j+1, i] - vx[j-1, i])
            F[j-1, i-1] = vx[j, i] - 0.25 * (term - conv1 - conv2)
    return F.flatten()

# Calcular el Jacobiano para toda la grilla
def jacobian_vx(vx, vy, h, nx, ny):
    alpha = 0.5 * h
    N = (ny - 2) * (nx - 2)
    J = sp.lil_matrix((N, N))

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = (j - 1) * (nx - 2) + (i - 1)

            diag = 1.0 # Diagonal
            if i + 1 < nx:
                diag += 0.25 * alpha * vx[j, i+1]
            if i - 1 >= 0:
                diag -= 0.25 * alpha * vx[j, i-1]
            J[k, k] = diag

            # Vecino al este
            if i + 1 < nx - 1:
                J[k, k + 1] = -0.25 + 0.25 * alpha * vx[j, i]

            # Vecino al oeste
            if i - 1 >= 1:
                J[k, k - 1] = -0.25 - 0.25 * alpha * vx[j, i]

            # Vecino al norte
            if j + 1 < ny - 1:
                J[k, k + (nx - 2)] = -0.25 + 0.25 * alpha * vy[j, i]

            # Vecino al sur
            if j - 1 >= 1:
                J[k, k - (nx - 2)] = -0.25 - 0.25 * alpha * vy[j, i]

    return J.tocsr()  # Convertimos a formato CSR para operaciones eficientes

# Metodo de Newton-Raphson para resolver vx
def newton_solver_vx(vx, vy, h, nx, ny, tol=1e-6, max_iter=50):
    for it in range(max_iter):
        F = Fx(vx, vy, h, nx, ny)
        normF = np.linalg.norm(F)
        print(f"Iteración {it}: ||F|| = {normF:.2e}")
        if normF < tol:
            print("Convergencia alcanzada.")
            break
        J = jacobian_vx(vx, vy, h, nx, ny)
        delta = spla.spsolve(J, -F)
        # Actualizar los nodos interiores
        vx_interior = vx[1:ny-1, 1:nx-1].flatten() + delta
        vx[1:ny-1, 1:nx-1] = vx_interior.reshape((ny-2, nx-2))
    return vx

# Ejecutar el metodo de Newton-Raphson
vx_final = newton_solver_vx(vx_vals, vy_vals, h, nx, ny)

# Crear un grafico de calor para las velocidades finales vx
plt.figure(figsize=(10, 5))
plt.imshow(vx_final, cmap='jet', interpolation='nearest', origin='upper', aspect='auto')
plt.colorbar(label='v_x (velocidad en x)')
plt.title('Grafica de calor velocidades (v_x)', fontsize=14)
plt.xlabel('Índice i (x)', fontsize=12)
plt.ylabel('Índice j (y)', fontsize=12)
plt.show()


J = jacobian_vx(vx_vals, vy_vals, h, nx, ny)

plt.figure(figsize=(8, 6))
plt.spy(J, markersize=0.5)
plt.title("Matriz Jacobiana")
plt.xlabel("Columnas")
plt.ylabel("Filas")
plt.grid(True)
plt.show()