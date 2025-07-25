import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline  # <--- añadido

# Índice para vectorizar nodos interiores
def idx(i, j, ny):
    return (i - 1) * (ny - 2) + (j - 1)

# Residuo F(vx) en nodos interiores
def compute_F(vx, vy, h, nx, ny):
    m, n = nx - 2, ny - 2
    F = np.empty(m * n)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            k = idx(i, j, ny)
            lap = vx[i+1,j] + vx[i-1,j] + vx[i,j+1] + vx[i,j-1]
            conv_x = (h/2) * vx[i,j] * (vx[i+1,j] - vx[i-1,j])
            conv_y = (h/2) * vy[i,j] * (vx[i,j+1] - vx[i,j-1])
            F[k] = vx[i,j] - 0.25 * (lap - conv_x - conv_y)
    return F

# Jacobiano disperso de F respecto a vx interior
def compute_J(vx, vy, h, nx, ny):
    m, n = nx-2, ny-2
    N = m * n
    alpha = h/2
    rows, cols, vals = [], [], []
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            k = idx(i,j,ny)
            # diagonal
            diag = 1.0 + 0.25*alpha*(vx[i+1,j] - vx[i-1,j])
            rows.append(k); cols.append(k); vals.append(diag)
            # este
            if i+1 < nx-1:
                rows.append(k); cols.append(idx(i+1,j,ny));
                vals.append(-0.25 + 0.25*alpha*vx[i,j])
            # oeste
            if i-1 > 0:
                rows.append(k); cols.append(idx(i-1,j,ny));
                vals.append(-0.25 - 0.25*alpha*vx[i,j])
            # norte
            if j+1 < ny-1:
                rows.append(k); cols.append(idx(i,j+1,ny));
                vals.append(-0.25 + 0.25*alpha*vy[i,j])
            # sur
            if j-1 > 0:
                rows.append(k); cols.append(idx(i,j-1,ny));
                vals.append(-0.25 - 0.25*alpha*vy[i,j])
    return sp.coo_matrix((vals,(rows,cols)),shape=(N,N)).tocsr()

# Solución de sistema A x = b por Gauss-Seidel
def gauss_seidel(A, b, x0=None, tol=1e-8, maxiter=1000):
    A = A.tocsr()
    n = b.size
    x = np.zeros(n) if x0 is None else x0.copy()
    diag = A.diagonal()
    indptr, indices, data = A.indptr, A.indices, A.data
    for it in range(maxiter):
        for i in range(n):
            start, end = indptr[i], indptr[i+1]
            row_idx = indices[start:end]
            row_data = data[start:end]
            sigma = 0.0
            for a_ij, j in zip(row_data, row_idx):
                if j != i:
                    sigma += a_ij * x[j]
            x[i] = (b[i] - sigma) / diag[i]
        r = b - A.dot(x)
        if np.linalg.norm(r, np.inf) < tol:
            print(f"GS convergió en {it} iteraciones")
            break
    return x

# Newton-Raphson para actualizar vx interior
def newton_vx(vx, vy, h, nx, ny, tol=1e-6, maxit=20):
    for k in range(maxit):
        F = compute_F(vx, vy, h, nx, ny)
        if np.linalg.norm(F, np.inf) < tol:
            print(f"Newton convergió en {k} iteraciones")
            return vx
        J = compute_J(vx, vy, h, nx, ny)
        delta = gauss_seidel(J, -F)
        vx[1:nx-1,1:ny-1] += delta.reshape((nx-2,ny-2))
    print("Newton no convergió")
    return vx

# --- PROGRAMA PRINCIPAL ---
if __name__ == '__main__':
    nx, ny, h = 200, 20, 1
    vx = np.zeros((nx,ny))
    vy = np.full((nx,ny), 0.0001)

    # Condición de frontera izquierda
    vx[0,1:ny-1] = 1

    # Inicialización del interior
    for j in range(1,ny-1):
        vx[1:nx-1,j] = 0.9 - (0.9/(nx-2))*(j-1)

    # Resolver con Newton-Raphson + Gauss-Seidel
    vx = newton_vx(vx, vy, h, nx, ny)

    # Suavizado con spline bicúbico
x = np.arange(nx)
y = np.arange(ny)
spline = RectBivariateSpline(y, x, vx.T)  # <- orden corregido

x_fino = np.linspace(0, nx-1, 4 * nx)
y_fino = np.linspace(0, ny-1, 4 * ny)

vx_suave = spline(y_fino, x_fino)  # <- orden también corregido

# Graficar
plt.figure(figsize=(10, 5))  # Ajusta el tamaño de la figura
plt.imshow(vx_suave, origin='lower', aspect='auto', cmap='jet',
           extent=[0, nx-1, 0, ny-1])

plt.colorbar(label='$v_x$')
plt.title('vx suavizado usando Splines')
plt.xlabel('i'); plt.ylabel('j')
plt.show()