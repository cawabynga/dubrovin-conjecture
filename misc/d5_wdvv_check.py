"""D5: homogeneous polynomial Frobenius potentials, WDVV, semisimplicity.

Requires SymPy: python -m pip install sympy
Run: python d5_wdvv_check.py

Coordinates t,u,v,w,z have weights 1,3/4,1/2,1/4,5/8.
The Frobenius metric pairs t--w, u--v with coefficient 1,
and z--z with coefficient -4.
"""
import itertools
import sympy as S

t,u,v,w,z = S.symbols('t u v w z')
xs=(t,u,v,w,z)
p,q,b,c = S.symbols('p q b c')
metric=S.zeros(5)
for i,j in ((0,3),(3,0),(1,2),(2,1)):
    metric[i,j]=1
metric[4,4]=-4
inverse=metric.inv()
unit=S.Rational(1,2)*t**2*w+t*u*v-2*t*z**2

# All 16 monomials in the part independent of t, of total weight 9/4.
monoms=(
    u**3, u**2*v*w, u**2*w**3, u*v**3,
    u*v**2*w**2,u*v*w**4,u*w**6,u*w*z**2,
    v**4*w,v**3*w**3,v**2*w**5,v**2*z**2,
    v*w**7,v*w**2*z**2,w**9,w**4*z**2,
)
coeff=S.symbols('c0:16')
F_general=unit+sum(a*m for a,m in zip(coeff,monoms))

def multiplication_matrices(F):
    d3=[[[S.diff(F,xs[i],xs[j],xs[k]) for k in range(5)]
         for j in range(5)] for i in range(5)]
    return [S.Matrix(5,5,lambda r,s:sum(inverse[r,k]*d3[i][s][k]
                                            for k in range(5)))
            for i in range(5)]

def wdvv_constraints(L):
    equations=set()
    for i,j in itertools.combinations(range(1,5),2):
        for expression in L[i]*L[j]-L[j]*L[i]:
            if expression != 0:
                for _,coefficient in S.Poly(expression,u,v,w,z).terms():
                    equations.add(S.expand(coefficient))
    return equations

L=multiplication_matrices(F_general)
assert L[0]==S.eye(5)
equations=wdvv_constraints(L)
print('Coefficient equations from all six WDVV commutators:',len(equations))

# Branch e1^2|0 = 0, i.e. c0 = 0.
branch={a:S.Integer(0) for a in coeff}
branch.update({coeff[3]:b,coeff[11]:c,
               coeff[8]:(c**2+6*b*c)/12})
assert all(S.expand(E.subs(branch))==0 for E in equations)
F_degenerate=S.expand(F_general.subs(branch))
assert S.diff(F_degenerate,w,2)==0
print('a=0 branch:',F_degenerate)
print('On this branch d_w squared = 0 identically, so never semisimple.')

# The second WDVV branch, c0=p, c3=q; a=6p.
generic={a:S.Integer(0) for a in coeff}
generic.update({
    coeff[0]:p,
    coeff[1]:9*p*q,
    coeff[2]:54*p**2*q**2,
    coeff[3]:q,
    coeff[4]:27*p*q**2,
    coeff[7]:36*p*q,
    coeff[8]:S.Rational(9,4)*q**2,
    coeff[9]:27*p*q**3,
    coeff[10]:S.Rational(1458,5)*p**2*q**4,
    coeff[11]:3*q,
    coeff[13]:-108*p*q**2,
    coeff[14]:S.Rational(13122,7)*p**4*q**6,
    coeff[15]:486*p**2*q**3,
})
assert all(S.expand(E.subs(generic))==0 for E in equations)

# Trace pairing B_ij = Tr(L_i L_j) at (t,u,v,w,z)=(0,0,0,1,0).
point={t:0,u:0,v:0,w:1,z:0}
Lg=[S.Matrix(M.subs(generic).subs(point)) for M in L]
trace_pairing=S.Matrix(5,5,lambda i,j:S.trace(Lg[i]*Lg[j]))
delta=S.factor(trace_pairing.det())
print('Trace determinant at (0,0,0,1,0):',delta)
assert S.simplify(delta-2**25*3**25*p**10*q**15)==0
print('With a=6p, this is 6^15 * a^10 * q^15.')
print('CONCLUSION: generic semisimplicity is equivalent to p*q != 0.')
