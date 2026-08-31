from dolfin import *
from mshr import *
import numpy as np
import pyvista as pv

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
    # Etapa 1 - Gerar malha
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

    # Etapa 2 - Resolver problema direto
    F_Problem = ForwardProblem3D(mesh_direct)
    list_u0 = F_Problem.solve_forward(VD, gamma0, list_gs)

    # Salvar arquivos para visualização
    xdmf_u = XDMFFile("solucao_u0_3d.xdmf")
    xdmf_u.write(list_u0[0])

    xdmf_gamma = XDMFFile("gamma0_3d.xdmf")
    xdmf_gamma.write(gamma0)

    print("Cálculo 3D concluído! Arquivos .xdmf gerados.")

    # Etapa 3 - Visualização da malha
    pontos = mesh_direct.coordinates()
    celulas = mesh_direct.cells()

    num_celulas = celulas.shape[0]
    celulas_pv = np.hstack((np.full((num_celulas, 1), 4), celulas)).flatten()
    tipos_celula = np.full(num_celulas, 10, dtype=np.uint8)

    malha_pv = pv.UnstructuredGrid(celulas_pv, tipos_celula, pontos)

    # Mapear os dados do FEniCS para o PyVista
    malha_pv.point_data["Potencial"] = list_u0[0].compute_vertex_values(mesh_direct)
    malha_pv.point_data["Condutividade"] = gamma0.compute_vertex_values(mesh_direct)

    # Configurar Plotter para exportação HTML interativa
    plotter = pv.Plotter(notebook=False)
    plotter.set_background("lightgray")

    # Filtro para extrair e desenhar todas as linhas internas da malha
    arestas_internas = malha_pv.extract_all_edges()
    plotter.add_mesh(arestas_internas, scalars="Potencial", cmap="coolwarm", line_width=1)

    # Salva o arquivo interativo
    arquivo_html = "visualizacao_malha_interna.html"
    plotter.export_html(arquivo_html)

    # Versão com cor uniforme
    plotter = pv.Plotter(notebook=False)
    plotter.set_background("lightgray")

    arestas_internas = malha_pv.extract_all_edges()

    # Adiciona o esqueleto tridimensional completo na tela
    plotter.add_mesh(
        arestas_internas, 
        color="navy",       # Cor uniforme de todas as linhas
        line_width=0.1
    )

    # Salva o arquivo interativo
    arquivo_html = "visualizacao_uniforme.html"
    plotter.export_html(arquivo_html)