import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import trimesh
import cupyx.scipy.sparse as sparse
import cupyx.scipy.sparse.linalg as linalg
from progress.bar import Bar
from scipy.sparse import lil_matrix

class SchrodingerEquation:

    H_BAR = 1.05457e-34

    def __init__(self, M, N, L, V, m, X, Y, Z, BC='periodic'):
        self.M = M
        self.N = N
        self.L = L

        self.m = m

        self.V = V

        self.dx = X / self.M
        self.dy = Y / self.N
        if self.L != 0:
            self.dz = Z / self.L

        self.BC = BC        

        if self.L == 0:
            self.n_nonzero = 5 * self.M * self.N
            if self.BC != 'periodic':
                self.n_nonzero -= 2 * self.M
                self.n_nonzero -= 2 * self.N

            self.n_unknowns = self.M * self.N

        else:
            self.n_nonzero = 7 * self.M * self.N * self.L
            if self.BC != 'periodic':
                self.n_nonzero -= 2 * self.M * self.N
                self.n_nonzero -= 2 * self.M * self.L
                self.n_nonzero -= 2 * self.N * self.L

            self.n_unknowns = self.M * self.N * self.L

        self.data = cp.zeros(self.n_nonzero)
        self.row  = cp.zeros(self.n_nonzero)
        self.col  = cp.zeros(self.n_nonzero)
        
    def ind(self, x, y, z):
        x = x % self.M
        y = y % self.N
        z = z % self.L
        return z * self.M * self.N + y * self.M + x
    
    def ind_2d(self, x, y):
        x = x % self.M
        y = y % self.N
        return y * self.M + x
    
    def get_V(self, x, y, z):
        x = x % self.M
        y = y % self.N
        z = z % self.L

        return self.V[z, y, x]
    
    def get_V_2d(self, x, y):
        x = x % self.M
        y = y % self.N

        return self.v[y, x]
    
    def set_diag(self, x, y, z, counter):
        row_col = self.ind(x, y, z)
        self.row[counter] = row_col
        self.col[counter] = row_col

        self.data[counter] += self.H_BAR**2 / (self.dx**2 * self.m)
        self.data[counter] += self.H_BAR**2 / (self.dy**2 * self.m)
        self.data[counter] += self.H_BAR**2 / (self.dz**2 * self.m)
        self.data[counter] += self.V[z, y, x]

        return counter + 1
    
    def set_diag_2d(self, x, y, counter):
        row_col = self.ind_2d(x, y)
        self.row[counter] = row_col
        self.col[counter] = row_col

        self.data[counter] += self.H_BAR**2 / (self.dx**2 * self.m)
        self.data[counter] += self.H_BAR**2 / (self.dy**2 * self.m)
        self.data[counter] += self.V[y, x]

        return counter + 1

    def set_off_diag(self, x, y, z, counter):
        row = self.ind(x, y, z)
        if x > 0:
            self.row[counter] = row
            self.col[counter] = self.ind(x - 1, y, z)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dx**2)
            counter += 1
        if x < self.M - 1:
            self.row[counter] = row
            self.col[counter] = self.ind(x + 1, y, z)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dx**2)
            counter += 1

        if y > 0:
            self.row[counter] = row
            self.col[counter] = self.ind(x, y - 1, z)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dy**2)
            counter += 1
        if y < self.N - 1:
            self.row[counter] = row
            self.col[counter] = self.ind(x, y + 1, z)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dy**2)
            counter += 1

        if z > 0:
            self.row[counter] = row
            self.col[counter] = self.ind(x, y, z - 1)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dz**2)
            counter += 1
        if z < self.L - 1:
            self.row[counter] = row
            self.col[counter] = self.ind(x, y, z + 1)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dz**2)
            counter += 1

        return counter
    
    def set_off_diag_2d(self, x, y, counter):
        row = self.ind_2d(x, y)
        if x > 0 or self.BC == 'periodic':
            self.row[counter] = row
            self.col[counter] = self.ind_2d(x - 1, y)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dx**2)
            counter += 1
        if x < self.M - 1 or self.BC == 'periodic':
            self.row[counter] = row
            self.col[counter] = self.ind_2d(x + 1, y)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dx**2)
            counter += 1

        if y > 0 or self.BC == 'periodic':
            self.row[counter] = row
            self.col[counter] = self.ind_2d(x, y - 1)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dy**2)
            counter += 1
        if y < self.N - 1 or self.BC == 'periodic':
            self.row[counter] = row
            self.col[counter] = self.ind_2d(x, y + 1)
            self.data[counter] = -self.H_BAR**2 / (2 * self.m * self.dy**2)
            counter += 1

        return counter

    def populate_matrix(self):
        counter = 0
        bar = Bar("Populating Matrix...", max=self.M * self.N * self.L)
        for z in range(self.L):
            for y in range(self.N):
                for x in range(self.M):
                    counter = self.set_diag(x, y, z, counter)
                    counter = self.set_off_diag(x, y, z, counter)
                    bar.next()
        bar.finish()

        self.A = sparse.csr_matrix((self.data, (self.row, self.col)))

    def populate_matrix_efficient(self):
        x = cp.arange(self.M)
        y = cp.arange(self.N)
        z = cp.arange(self.L)

        z, y, x = cp.meshgrid(z, y, x, indexing='ij')
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        self.A = lil_matrix((self.n_unknowns, self.n_unknowns))

        bar = Bar("Populating matrix", max=9)

        row_col = self.ind(x, y, z).get()

        self.A[row_col, row_col] += np.ones_like(row_col) * self.H_BAR**2 / (self.dx**2 * self.m)
        bar.next()
        self.A[row_col, row_col] += np.ones_like(row_col) * self.H_BAR**2 / (self.dy**2 * self.m)
        bar.next()
        self.A[row_col, row_col] += np.ones_like(row_col) * self.H_BAR**2 / (self.dz**2 * self.m)
        bar.next()
        self.A[row_col, row_col] += self.V[z, y, x].get()
        bar.next()

        self.A[row_col, self.ind(x - 1, y, z).get()] = -self.H_BAR**2 / (2 * self.m * self.dx**2)
        bar.next()
        self.A[row_col, self.ind(x + 1, y, z).get()] = -self.H_BAR**2 / (2 * self.m * self.dx**2)
        bar.next()
        self.A[row_col, self.ind(x, y - 1, z).get()] = -self.H_BAR**2 / (2 * self.m * self.dy**2)
        bar.next()
        self.A[row_col, self.ind(x, y + 1, z).get()] = -self.H_BAR**2 / (2 * self.m * self.dy**2)
        bar.next()
        self.A[row_col, self.ind(x, y, z - 1).get()] = -self.H_BAR**2 / (2 * self.m * self.dz**2)
        bar.next()
        self.A[row_col, self.ind(x, y, z + 1).get()] = -self.H_BAR**2 / (2 * self.m * self.dz**2)
        bar.finish()

        self.A = sparse.csr_matrix(self.A)

    def populate_matrix_2d(self):
        counter = 0
        bar = Bar("Populating Matrix...", max=self.M * self.N)
        for y in range(self.N):
            for x in range(self.M):
                counter = self.set_diag_2d(x, y, counter)
                counter = self.set_off_diag_2d(x, y, counter)
                bar.next()
        bar.finish()

        self.A = sparse.csr_matrix((self.data, (self.row, self.col)))

    def populate_matrix_2d_efficient(self):
        x = cp.arange(self.M)
        y = cp.arange(self.N)

        y, x = cp.meshgrid(y, x, indexing='ij')
        x = x.flatten()
        y = y.flatten()

        self.A = lil_matrix((self.n_unknowns, self.n_unknowns))

        bar = Bar("Populating matrix", max=6)

        row_col = self.ind_2d(x, y).get()

        self.A[row_col, row_col] += np.ones_like(row_col) * self.H_BAR**2 / (self.dx**2 * self.m)
        bar.next()
        self.A[row_col, row_col] += np.ones_like(row_col) * self.H_BAR**2 / (self.dy**2 * self.m)
        bar.next()
        self.A[row_col, row_col] += self.V[y, x].get()
        bar.next()

        self.A[row_col, self.ind_2d(x - 1, y).get()] = -self.H_BAR**2 / (2 * self.m * self.dx**2)
        bar.next()
        self.A[row_col, self.ind_2d(x + 1, y).get()] = -self.H_BAR**2 / (2 * self.m * self.dx**2)
        bar.next()
        self.A[row_col, self.ind_2d(x, y - 1).get()] = -self.H_BAR**2 / (2 * self.m * self.dy**2)
        bar.next()
        self.A[row_col, self.ind_2d(x, y + 1).get()] = -self.H_BAR**2 / (2 * self.m * self.dy**2)
        bar.finish()

        self.A = sparse.csr_matrix(self.A)

    def populate_phi(self, eigenvector):
        z = cp.arange(self.L)
        y = cp.arange(self.N)
        x = cp.arange(self.M)

        z, y, x = cp.meshgrid(z, y, x, indexing='ij')

        inds = self.ind(x, y, z)

        return eigenvector[inds]
    
    def populate_phi_2d(self, eigenvector):
        y = cp.arange(self.N)
        x = cp.arange(self.M)

        y, x = cp.meshgrid(y, x, indexing='ij')

        inds = self.ind_2d(x, y)

        return eigenvector[inds]



        
if __name__=='__main__':
    M, N = 700, 700
    L = 0

    J_to_eV = 6.242e18

    er = 2e-10
    me = 9.10938e-31
    q = 1.6021766e-19
    alpha = 1 / 137
    c = 2.998e8
    epsilon_0 = 8.8541878e-12

    Hr = SchrodingerEquation.H_BAR / (me * c * alpha)

    X = 150 * Hr
    Y = 150 * Hr
    Z = 150 * Hr

    x = cp.linspace(-X / 2, X / 2, M)
    y = cp.linspace(-Y / 2, Y / 2, N)
    z = cp.linspace(-Z / 2, Z / 2, L)

    z, y, x = cp.meshgrid(z, y, x, indexing='ij')

    # 4 well example
    r1 = cp.sqrt((x - 1.5*Hr)**2 + (y - 1.5*Hr)**2 + z**2)
    r2 = cp.sqrt((x + 1.5*Hr)**2 + (y - 1.5*Hr)**2 + z**2)
    r3 = cp.sqrt((x - 1.5*Hr)**2 + (y + 1.5*Hr)**2 + z**2)
    r4 = cp.sqrt((x + 1.5*Hr)**2 + (y + 1.5*Hr)**2 + z**2)

    V1 = -q**2 / (4 * np.pi * epsilon_0 * r1)
    V2 = -q**2 / (4 * np.pi * epsilon_0 * r2)
    V3 = -q**2 / (4 * np.pi * epsilon_0 * r3)
    V4 = -q**2 / (4 * np.pi * epsilon_0 * r4)

    V = V1
    V += V2
    V += V3
    V += V4

    V -= V.max()

    m = 0.511e6

    se = SchrodingerEquation(M, N, L, V, me, X, Y, Z, BC='periodic')

    se.populate_matrix_2d_efficient()
    
    print("Calculating Eigensolutions")
    eigenvalues, eigenvectors = linalg.eigsh(se.A, k=N, which='SA', )

    phis = [se.populate_phi_2d(ev).get()**2 for ev in eigenvectors.T]

    for i in range(20):
        plt.imshow(phis[i])
        plt.show()
