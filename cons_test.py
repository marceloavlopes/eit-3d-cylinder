from dolfin import *
from mshr import *
from main_3d import MyMesh3D, ForwardProblem3D

mesh_direct = MyMesh3D(r=1.0, h=2.0, n=25)
VD = FiniteElement('CG', mesh_direct.ufl_cell(), 1)
ds = Measure("ds", domain=mesh_direct)

# 1. Gamma - Constante e igual a 1
gamma = Constant(1.0)

# 2. Função harmônica
x = SpatialCoordinate(mesh_direct)
u_exata_ufl = x[0]**2 - x[1]**2

# 3. Condição de fronteira
n = FacetNormal(mesh_direct)
g = dot(grad(u_exata_ufl), n) #g = (\nabla u).\eta

# 4. Integral de u e de u-c
integral_u = assemble(u_exata_ufl * ds)
print(f"Integral de u na fronteira: {integral_u:.5e}")

area_fronteira = assemble(Constant(1.0) * ds)
c_val = integral_u / area_fronteira
u_exata_ufl = u_exata_ufl - Constant(c_val)

nova_integral = assemble(u_exata_ufl * ds)
print(f"Nova integral (u - c): {nova_integral:.5e}")
print(f"Constante c subtraída: {c_val:.5e}")

# 5. Problema direto
F_Problem = ForwardProblem3D(mesh_direct)
list_sol = F_Problem.solve_forward(VD, gamma, [g])
u_sol = list_sol[0]

# 6. Comparação exata vs calculada
u_exata_expr = Expression("pow(x[0], 2) - pow(x[1], 2) - c", c=c_val, degree=3)
erro_L2 = errornorm(u_exata_expr, u_sol, 'L2')
print(f"Erro de u na norma L2: {erro_L2:.5e}")

# 7. Gradiente na fronteira
erro_fluxo = assemble((dot(grad(u_sol), n) - g)**2 * ds)
print(f"Erro na fronteira: {erro_fluxo:.5e}")