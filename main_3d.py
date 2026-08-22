from dolfin import *
from mshr import *
import numpy as np

def MyMesh3D(r=1.0, h=2.0, n=25):
    cilindro = Cylinder(Point(0, 0, h/2), Point(0, 0, -h/2), r, r)
    mesh = generate_mesh(cilindro, n)
    mesh.radius = r
    mesh.height = h
    return mesh

class ForwardProblem3D(object):
    def __init__(self, mesh):
        self.mesh = mesh

    def solve_forward(self, V, gamma, I_all):
        mesh = self.mesh
        n_g = len(I_all)
        
        # Espaço Misturado H1 x R para multiplicador de Lagrange
        R = FiniteElement('R', mesh.ufl_cell(), 0) 
        W = FunctionSpace(mesh, MixedElement([V, R]))

        (u, c) = TrialFunctions(W)
        (v, d) = TestFunctions(W)

        ds = Measure("ds", domain=mesh)

        # Condição de unicidade int(u)ds = 0
        lagrMult = (v * c + u * d) * ds
        a = inner(gamma * grad(u), grad(v)) * dx + lagrMult
        A = assemble(a)

        sol_u = []

        for j in range(n_g):
            L = I_all[j] * v * ds
            b = assemble(L)
            
            w = Function(W)
            U = w.vector()
            solve(A, U, b)
            
            u_sol, c_sol = w.split(deepcopy=True)
            sol_u.append(u_sol)

        self.sol_u = sol_u
        return sol_u

if __name__ == "__main__":
    # Gerar malha
    mesh_direct = MyMesh3D(r=1.0, h=2.0, n=20)
    
    # Espaço de elementos finitos
    VD = FiniteElement('CG', mesh_direct.ufl_cell(), 1)
    
    # Definição da condutividade (fundo 1.0, esfera 5.0)
    Q_DG = FunctionSpace(mesh_direct, "DG", 0)
    gamma_expr = Expression("x[0]*x[0] + x[1]*x[1] + x[2]*x[2] <= 0.16 ? 5.0 : 1.0", degree=1)
    gamma0 = interpolate(gamma_expr, Q_DG)

    # Corrente na borda topo/base
    g1 = Expression("x[2] > 0.9 ? 1.0 : (x[2] < -0.9 ? -1.0 : 0.0)", degree=1)
    list_gs = [g1]

    # Resolver problema direto
    F_Problem = ForwardProblem3D(mesh_direct)
    list_u0 = F_Problem.solve_forward(VD, gamma0, list_gs)

    # Salvar arquivos para visualização
    xdmf_u = XDMFFile("solucao_u0_3d.xdmf")
    xdmf_u.write(list_u0[0])

    xdmf_gamma = XDMFFile("gamma0_3d.xdmf")
    xdmf_gamma.write(gamma0)

    print("Cálculo 3D concluído! Arquivos .xdmf gerados.")