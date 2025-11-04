from sympy import *
import torch

def convert_dimensionless(name, data, labels):

    if 'I.6.2' in name and 'I.6.2b' not in name:
        # Transform data
        # Code order: theta[0], sigma[1]
        # Y_char = 1 (empty formula)
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: sigma
        X_DL_1 = data[:,[1]]
        # var2: theta
        X_DL_2 = data[:,[0]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.6.2b' in name:
        # Transform data
        # Code order: theta[0], theta1[1], sigma[2]
        # Y_char = 1 (empty formula)
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: sigma
        X_DL_1 = data[:,[2]]
        # var2: theta
        X_DL_2 = data[:,[0]]
        # var3: theta1
        X_DL_3 = data[:,[1]]
        X = torch.cat([X_DL_1, X_DL_2, X_DL_3], dim=1)
        return X, y

    if 'I.9.18' in name:
        # Transform data
        # Code order: G[0], m1[1], m2[2], x1[3], x2[4], y1[5], y2[6], z1[7], z2[8]
        # Y_char = G*m1^2/x1^2
        Y_char = data[:,[0]] * data[:,[1]]**2 / data[:,[3]]**2
        y = labels / Y_char
        # var1: m2/m1
        X_DL_1 = data[:,[2]] / data[:,[1]]
        # var2: x2/x1
        X_DL_2 = data[:,[4]] / data[:,[3]]
        # var3: y1/x1
        X_DL_3 = data[:,[5]] / data[:,[3]]
        # var4: y2/x1
        X_DL_4 = data[:,[6]] / data[:,[3]]
        # var5: z1/x1
        X_DL_5 = data[:,[7]] / data[:,[3]]
        # var6: z2/x1
        X_DL_6 = data[:,[8]] / data[:,[3]]
        X = torch.cat([X_DL_1, X_DL_2, X_DL_3, X_DL_4, X_DL_5, X_DL_6], dim=1)
        return X, y

    if 'I.12.11' in name:
        # Transform data
        # Code order: q[0], Ef[1], B[2], v[3], theta[4]
        # Y_char = Ef*q
        Y_char = data[:,[1]] * data[:,[0]]
        y = labels / Y_char
        # var1: B*v/Ef
        X_DL_1 = (data[:,[2]] * data[:,[3]]) / data[:,[1]]
        # var2: theta
        X_DL_2 = data[:,[4]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.13.12' in name:
        # Transform data
        # Code order: G[0], m1[1], m2[2], r1[3], r2[4]
        # Y_char = G*m1^2/r1
        Y_char = data[:,[0]] * data[:,[1]]**2 / data[:,[3]]
        y = labels / Y_char
        # var1: m2/m1
        X_DL_1 = data[:,[2]] / data[:,[1]]
        # var2: r2/r1
        X_DL_2 = data[:,[4]] / data[:,[3]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.15.3x' in name:
        # Transform data
        # Code order: x[0], u[1], t[2], c[3]
        # Y_char = x
        Y_char = data[:,[0]]
        y = labels / Y_char
        # var1: c/u
        X_DL_1 = data[:,[3]] / data[:,[1]]
        # var2: t*u/x
        X_DL_2 = (data[:,[2]] * data[:,[1]]) / data[:,[0]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.16.6' in name:
        # Transform data
        # Code order: u[0], v[1], c[2]
        # Y_char = c
        Y_char = data[:,[2]]
        y = labels / Y_char
        # var1: v/c
        X_DL_1 = data[:,[1]] / data[:,[2]]
        # var2: u/c
        X_DL_2 = data[:,[0]] / data[:,[2]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.18.4' in name:
        # Transform data
        # Code order: m1[0], r1[1], m2[2], r2[3]
        # Y_char = r1
        Y_char = data[:,[1]]
        y = labels / Y_char
        # var1: m2/m1
        X_DL_1 = data[:,[2]] / data[:,[0]]
        # var2: r2/r1
        X_DL_2 = data[:,[3]] / data[:,[1]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.27.6' in name:
        # Transform data
        # Code order: d1[0], d2[1], n[2]
        # Y_char = d1
        Y_char = data[:,[0]]
        y = labels / Y_char
        # var1: d2/d1
        X_DL_1 = data[:,[1]] / data[:,[0]]
        # var2: n
        X_DL_2 = data[:,[2]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.26.2' in name:
        # Transform data
        # Code order: n[0], theta2[1]
        # Y_char = 1 (empty formula)
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: n
        X_DL_1 = data[:,[0]]
        # var2: theta2
        X_DL_2 = data[:,[1]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.29.16' in name:
        # Transform data
        # Code order: x1[0], x2[1], theta1[2], theta2[3]
        # Y_char = x1
        Y_char = data[:,[0]]
        y = labels / Y_char
        # var1: x2/x1
        X_DL_1 = data[:,[1]] / data[:,[0]]
        # var2: theta1
        X_DL_2 = data[:,[2]]
        # var3: theta2
        X_DL_3 = data[:,[3]]
        X = torch.cat([X_DL_1, X_DL_2, X_DL_3], dim=1)
        return X, y

    if 'I.30.3' in name:
        # Transform data
        # Code order: I0[0], n[1], theta[2]
        # Y_char = I0
        Y_char = data[:,[0]]
        y = labels / Y_char
        # var1: theta
        X_DL_1 = data[:,[2]]
        # var2: n
        X_DL_2 = data[:,[1]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.30.5' in name:
        # Transform data
        # Code order: lamb[0], n[1], d[2]
        # Y_char = 1 (n^0.0 = 1)
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: d/lamb
        X_DL_1 = data[:,[2]] / data[:,[0]]
        # var2: n
        X_DL_2 = data[:,[1]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.37.4' in name:
        # Transform data
        # Code order: I1[0], I2[1], delta[2]
        # Y_char = I1
        Y_char = data[:,[0]]
        y = labels / Y_char
        # var1: I2/I1
        X_DL_1 = data[:,[1]] / data[:,[0]]
        # var2: delta
        X_DL_2 = data[:,[2]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.40.1' in name:
        # Transform data
        # Code order: n0[0], m[1], g[2], x[3], kb[4], T[5]
        # Y_char = 1
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: n0
        X_DL_1 = data[:,[0]]
        # var2: T*kb/(g*m*x)
        X_DL_2 = (data[:,[5]] * data[:,[4]]) / (data[:,[2]] * data[:,[1]] * data[:,[3]])
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.44.4' in name:
        # Transform data
        # Code order: n[0], kb[1], T[2], V1[3], V2[4]
        # Y_char = T*kb
        Y_char = data[:,[2]] * data[:,[1]]
        y = labels / Y_char
        # var1: n
        X_DL_1 = data[:,[0]]
        # var2: V2/V1
        X_DL_2 = data[:,[4]] / data[:,[3]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'I.50.26' in name:
        # Transform data
        # Code order: x1[0], alpha[1], omega[2], t[3]
        # Y_char = x1
        Y_char = data[:,[0]]
        y = labels / Y_char
        # var1: omega*t
        X_DL_1 = data[:,[2]] * data[:,[3]]
        # var2: alpha
        X_DL_2 = data[:,[1]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'II.2.42' in name:
        # Transform data
        # Code order: kappa[0], T1[1], T2[2], A[3], d[4]
        # Y_char = A^0.5*T1*kappa
        Y_char = torch.sqrt(data[:,[3]]) * data[:,[1]] * data[:,[0]]
        y = labels / Y_char
        # var1: T2/T1
        X_DL_1 = data[:,[2]] / data[:,[1]]
        # var2: d/sqrt(A)
        X_DL_2 = data[:,[4]] / torch.sqrt(data[:,[3]])
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'II.6.15a' in name:
        # Transform data
        # Code order: eps[0], pd[1], z[2], x[3], y[4], r[5]
        # Y_char = pd/(eps*r^3)
        Y_char = data[:,[1]] / (data[:,[0]] * data[:,[5]]**3)
        y = labels / Y_char
        # var1: x/r
        X_DL_1 = data[:,[3]] / data[:,[5]]
        # var2: y/r
        X_DL_2 = data[:,[4]] / data[:,[5]]
        # var3: z/r
        X_DL_3 = data[:,[2]] / data[:,[5]]
        X = torch.cat([X_DL_1, X_DL_2, X_DL_3], dim=1)
        return X, y

    if 'II.11.17' in name:
        # Transform data
        # Code order: n0[0], pd[1], Ef[2], theta[3], kb[4], T[5]
        # Y_char = 1 (empty formula)
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: n0
        X_DL_1 = data[:,[0]]
        # var2: Ef*pd/(T*kb)
        X_DL_2 = (data[:,[2]] * data[:,[1]]) / (data[:,[5]] * data[:,[4]])
        # var3: theta
        X_DL_3 = data[:,[3]]
        X = torch.cat([X_DL_1, X_DL_2, X_DL_3], dim=1)
        return X, y

    if 'II.11.27' in name:
        # Transform data
        # Code order: n[0], alpha[1], eps[2], Ef[3]
        # Y_char = Ef*eps
        Y_char = data[:,[3]] * data[:,[2]]
        y = labels / Y_char
        # var1: n
        X_DL_1 = data[:,[0]]
        # var2: alpha
        X_DL_2 = data[:,[1]]
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'II.35.18' in name:
        # Transform data
        # Code order: n0[0], mu[1], B[2], kb[3], T[4]
        # Y_char = 1 (B^0.0 = 1)
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: n0
        X_DL_1 = data[:,[0]]
        # var2: B*mu/(T*kb)
        X_DL_2 = (data[:,[2]] * data[:,[1]]) / (data[:,[4]] * data[:,[3]])
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'II.36.38' in name:
        # Transform data
        # Code order: mu[0], B[1], kb[2], T[3], alpha[4], M[5], eps[6], c[7]
        # Y_char = 1
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: T*kb/(B*mu)  # Note: H in CSV is B in code
        X_DL_1 = (data[:,[3]] * data[:,[2]]) / (data[:,[1]] * data[:,[0]])
        # var2: alpha
        X_DL_2 = data[:,[4]]
        # var3: M/(B*c^2*eps)  # Note: H in CSV is B in code
        X_DL_3 = data[:,[5]] / (data[:,[1]] * data[:,[7]]**2 * data[:,[6]])
        X = torch.cat([X_DL_1, X_DL_2, X_DL_3], dim=1)
        return X, y

    if 'II.38.3' in name:
        # Transform data
        # Code order: Y[0], A[1], x[2], d[3]
        # Y_char = A*Y
        Y_char = data[:,[1]] * data[:,[0]]
        y = labels / Y_char
        # var1: d/sqrt(A)
        X_DL_1 = data[:,[3]] / torch.sqrt(data[:,[1]])
        # var2: x/sqrt(A)
        X_DL_2 = data[:,[2]] / torch.sqrt(data[:,[1]])
        X = torch.cat([X_DL_1, X_DL_2], dim=1)
        return X, y

    if 'III.9.52' in name:
        # Transform data
        # Code order: pd[0], Ef[1], t[2], hbar[3], omega[4], omega0[5]
        # Y_char = 1
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: hbar/(Ef*pd*t)  # Note: h in CSV is hbar in code
        X_DL_1 = data[:,[3]] / (data[:,[1]] * data[:,[0]] * data[:,[2]])
        # var2: omega*t
        X_DL_2 = data[:,[4]] * data[:,[2]]
        # var3: omega0*t
        X_DL_3 = data[:,[5]] * data[:,[2]]
        X = torch.cat([X_DL_1, X_DL_2, X_DL_3], dim=1)
        return X, y

    if 'III.17.37' in name:
        # Transform data
        # Code order: beta[0], alpha[1], theta[2]
        # Y_char = 1 (empty formula)
        Y_char = torch.ones_like(labels)
        y = labels / Y_char
        # var1: beta
        X_DL_1 = data[:,[0]]
        # var2: alpha
        X_DL_2 = data[:,[1]]
        # var3: theta
        X_DL_3 = data[:,[2]]
        X = torch.cat([X_DL_1, X_DL_2, X_DL_3], dim=1)
        return X, y

# Modified from https://github.com/KindXiaoming/pykan
def get_feynman_dataset(name):
    
    global symbols
    
    tpi = torch.tensor(torch.pi)
    
    if name == 'test':
        symbol = x, y = symbols('x, y')
        expr = (x+y) * sin(exp(2*y))
        f = lambda x: (x[:,[0]] + x[:,[1]])*torch.sin(torch.exp(2*x[:,[1]]))
        ranges = [-1,1]

    # DIMENSIONLESS FORMULAS AND REASONABLE RANGE GUESSES

    # This is the old way to do the feynman dataset. It involves guessing on the ranges and manually coding the dimensionless formulas
    if name == 'I.6.2_dimensionless':
        symbol = theta, sigma = symbols('theta sigma')
        expr = exp(-theta**2/(2*sigma**2))/sqrt(2*pi*sigma**2)
        f = lambda x: torch.exp(-x[:,[0]]**2/(2*x[:,[1]]**2))/torch.sqrt(2*tpi*x[:,[1]]**2)
        ranges = [[-1,1],[0.5,2]] # domain is the same!

    if name == 'I.6.2b_dimensionless':
        symbol = theta, theta1, sigma = symbols('theta theta1 sigma')
        expr = exp(-(theta-theta1)**2/(2*sigma**2))/sqrt(2*pi*sigma**2)
        f = lambda x: torch.exp(-(x[:,[0]]-x[:,[1]])**2/(2*x[:,[2]]**2))/torch.sqrt(2*tpi*x[:,[2]]**2)
        ranges = [[-1.5,1.5],[-1.5,1.5],[0.5,2]] # domain is the same!

    if name == 'I.9.18_dimensionless':
        symbol = a, b, c, d, e, f = symbols('a, b, c, d, e, f')
        expr = a/((b-1)**2+(c-d)**2+(e-f)**2)
        f = lambda x: x[:,[0]]/((x[:,[1]]-1)**2+(x[:,[2]]-x[:,[3]])**2+(x[:,[4]]-x[:,[5]])**2)
        ranges = [[-1,1],[2,3],[0.5,1],[-1,-0.5],[0.5,1],[-1,-0.5]] # previous modified ranges
        # ranges = [[-1,1],[-1,0.],[0.5,1],[-1,-0.5],[0.5,1],[-1,-0.5]]

    if name == 'I.12.11_dimensionless':
        symbol = a, theta = symbols('a theta')
        expr = 1 + a*sin(theta)
        f = lambda x: 1 + x[:,[0]]*torch.sin(x[:,[1]])
        # ranges = [[-1,1],[0,2*tpi]] # Simply pick -1, 1 as th range if it doesn't cause problems
        ranges = [[-3,1],[0,2*tpi]] # Previous ranges

    if name == 'I.13.12_dimensionless':
        symbol = a, b = symbols('a b')
        expr = a*(1/b-1)
        f = lambda x: x[:,[0]]*(1/x[:,[1]]-1)
        # ranges = [[0,1],[0.5,2]]
        ranges = [[0,2],[0.25,4]] # Previous ranges

    if name == 'I.15.3x_dimensionless':
        symbol = a, b = symbols('a b')
        expr = (1 - a)/sqrt(1-b**2)
        f = lambda x: (1 - x[:,[0]])/torch.sqrt(1-x[:,[1]]**2)
        # ranges = [[-1,1],[-.9,.9]]
        ranges = [[-1,3],[-.9,.9]] # Previous ranges

    if name == 'I.16.6_dimensionless':
        symbol = a, b = symbols('a b')
        expr = (a+b)/(1+a*b)
        f = lambda x: (x[:,[0]]+x[:,[1]])/(1+x[:,[0]]*x[:,[1]])
        ranges = [[-0.8,0.8],[-0.8,0.8]]

    if name == 'I.18.4_dimensionless':
        symbol = a, b = symbols('a b')
        expr = (1 + a*b)/(1 + a)
        f = lambda x: (1 + x[:,[0]]*x[:,[1]])/(1 + x[:,[0]])
        # ranges = [[0.5, 1], [-1, 1]]
        ranges = [[0,1],[-3,1]] # Previous ranges

    if name == 'I.26.2_dimensionless':
        symbol = n, theta2 = symbols('n theta2')
        expr = asin(n*sin(theta2))
        f = lambda x: torch.arcsin(x[:,[0]]*torch.sin(x[:,[1]]))
        ranges = [[0,0.99],[0,2*tpi]]

    if name == 'I.27.6_dimensionless':
        symbol = a, b = symbols('a b')
        expr = 1/(1 + a*b)
        f = lambda x: 1/(1+x[:,[0]]*x[:,[1]])
        ranges = [[0,1],[0,3]]

    if name == 'I.29.16_dimensionless':
        symbol = a, theta1, theta2 = symbols('a theta1 theta2')
        expr = sqrt(1+a**2-2*a*cos(theta1-theta2))
        f = lambda x: torch.sqrt(1+x[:,[0]]**2-2*x[:,[0]]*torch.cos(x[:,[1]]-x[:,[2]]))
        ranges = [[-1,1],[0,2*tpi],[0,2*tpi]] #check range of a to make sure it works
        
    if name == 'I.30.3_dimensionless':
        symbol = n, theta = symbols('n theta')
        expr = sin(n*theta/2)**2 / sin(theta/2) ** 2
        f = lambda x: torch.sin(x[:,[0]]*x[:,[1]]/2)**2 / torch.sin(x[:,[1]]/2)**2
        ranges = [[0,4],[0.4*tpi,1.6*tpi]] # Fix this range so we don't have divide by zero issues

    if name == 'I.30.5_dimensionless':
        symbol = a, n = symbols('a n')
        expr = asin(a/n)
        f = lambda x: torch.arcsin(x[:,[0]]/x[:,[1]])
        ranges = [[-1,1],[1,2.25]]
        
    if name == 'I.37.4_dimensionless':
        symbol = a, delta = symbols('a delta')
        expr = 1 + a + 2*sqrt(a)*cos(delta)
        f = lambda x: 1 + x[:,[0]] +2*torch.sqrt(x[:,[0]])*torch.cos(x[:,[1]])
        ranges = [[0.01,1],[0,2*tpi]]
        
    if name == 'I.40.1_dimensionless':
        symbol = n0, a = symbols('n_0 a')
        expr = n0 * exp(-a)
        f = lambda x: x[:,[0]] * torch.exp(-x[:,[1]])
        ranges = [[0,1],[-1,1]]
        
    if name == 'I.44.4_dimensionless':
        symbol = n, a = symbols('n a')
        expr = n*log(a)
        f = lambda x: x[:,[0]]*torch.log(x[:,[1]])
        ranges = [[0,1],[0.25,4]]        
        
    if name == 'I.50.26_dimensionless':
        symbol = a, alpha = symbols('a alpha')
        expr = cos(a)+alpha*cos(a)**2
        f = lambda x: torch.cos(x[:,[0]])+x[:,[1]]*torch.cos(x[:,[0]])**2
        ranges = [[0,2*tpi],[0,1]]

    if name == 'II.2.42_dimensionless':
        symbol = a, b = symbols('a b')
        expr = (a-1)*b
        f = lambda x: (x[:,[0]]-1)*x[:,[1]]
        ranges = [[-1,3],[-1,1]] # Could also be [[-1,3],[0,1]]
        
    if name == 'II.6.15a_dimensionless':
        symbol = a, b, c = symbols('a b c')
        expr = 1/(4*pi)*c*sqrt(a**2+b**2)
        f = lambda x: 1/(4*tpi)*x[:,[2]]*torch.sqrt(x[:,[0]]**2+x[:,[1]]**2)
        ranges = [[0,1],[0,1],[0,4]] # Last range should be [0, 192], this happens because of r^5. Cap at 4
        
    if name == 'II.11.17_dimensionless':
        symbol = n_0, a, theta = symbols('n_0 a theta')
        expr = n_0*(1+a*cos(theta))
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*torch.cos(x[:,[2]]))
        ranges = [[0,1],[-4,4],[0,2*tpi]]
        
    if name == 'II.11.27_dimensionless':
        symbol = n, alpha = symbols('n alpha')
        expr = n*alpha/(1-n*alpha/3)
        f = lambda x: x[:,[0]]*x[:,[1]]/(1-x[:,[0]]*x[:,[1]]/3)
        ranges = [[0,1],[0,2]]
        
    if name == 'II.35.18_dimensionless':
        symbol = n0, a = symbols('n0 a')
        expr = n0/(exp(a)+exp(-a))
        f = lambda x: x[:,[0]]/(torch.exp(x[:,[1]])+torch.exp(-x[:,[1]]))
        ranges = [[0,1],[0,4]]
        
    if name == 'II.36.38_dimensionless':
        symbol = a, alpha, b = symbols('a alpha b')
        expr = a+alpha*b
        f = lambda x: x[:,[0]]+x[:,[1]]*x[:,[2]]
        ranges = [[0,1],[0,1],[0,4]] # Last range should be [0,16] because of large divisor in original formula cap at 4
        
    if name == 'II.38.3_dimensionless':
        symbol = a, b = symbols('a b')
        expr = a/b
        f = lambda x: x[:,[0]]/x[:,[1]]
        ranges = [[0,1],[0.5,2]]

    if name == 'III.9.52_dimensionless':
        symbol = a, b, c = symbols('a b c')
        expr = a*sin((b-c)/2)**2/((b-c)/2)**2
        f = lambda x: x[:,[0]]*torch.sin((x[:,[1]]-x[:,[2]])/2)**2/((x[:,[1]]-x[:,[2]])/2)**2
        ranges = [[0,1],[0,tpi],[-tpi,-0.1]]
        
    if name == 'III.10.19_dimensionless':
        symbol = a, b = symbols('a b')
        expr = sqrt(1+a**2+b**2)
        f = lambda x: torch.sqrt(1+x[:,[0]]**2+x[:,[1]]**2)
        ranges = [[0,1],[0,1]]

    if name == 'III.17.37_dimensionless':
        symbol = beta, alpha, theta = symbols('beta alpha theta')
        expr = beta * (1+alpha*cos(theta))
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*torch.cos(x[:,[2]]))
        ranges = [[0,1],[0,1],[0,2*tpi]]

    # ORIGINAL FORMULAS FROM https://space.mit.edu/home/tegmark/aifeynman.html

    if name == 'I.6.2_orig':
        symbol = theta, sigma = symbols('theta sigma')
        expr = exp(-theta**2/(2*sigma**2))/sqrt(2*pi*sigma**2)
        f = lambda x: torch.exp(-x[:,[0]]**2/(2*x[:,[1]]**2))/torch.sqrt(2*tpi*x[:,[1]]**2)
        ranges = [[1,3],[1,3]]

    if name == 'I.6.2b_orig':
        symbol = theta, theta1, sigma = symbols('theta theta1 sigma')
        expr = exp(-(theta-theta1)**2/(2*sigma**2))/sqrt(2*pi*sigma**2)
        f = lambda x: torch.exp(-(x[:,[0]]-x[:,[1]])**2/(2*x[:,[2]]**2))/torch.sqrt(2*tpi*x[:,[2]]**2)
        ranges = [[1,3],[1,3],[1,3]]

    if name == 'I.9.18_orig':
        symbol = G, m1, m2, x1, x2, y1, y2, z1, z2 = symbols('G m1 m2 x1 x2 y1 y2 z1 z2')
        expr = G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/((x[:,[3]]-x[:,[4]])**2+(x[:,[5]]-x[:,[6]])**2+(x[:,[7]]-x[:,[8]])**2)
        ranges = [[1,2],[1,2],[1,2],[3,4],[1,2],[3,4],[1,2],[3,4],[1,2]]

    if name == 'I.12.11_orig':
        symbol = q, Ef, B, v, theta = symbols('q E_f B v theta')
        expr = q*(Ef + B*v*sin(theta))
        f = lambda x: x[:,[0]]*(x[:,[1]]+x[:,[2]]*x[:,[3]]*torch.sin(x[:,[4]]))
        ranges = [[1,5],[1,5],[1,5],[1,5],[1,5]]

    if name == 'I.13.12_orig':
        symbol = G, m1, m2, r1, r2 = symbols('G m1 m2 r1 r2')
        expr = G*m1*m2*(1/r2-1/r1)
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]*(1/x[:,[4]]-1/x[:,[3]])
        ranges = [[1,5],[1,5],[1,5],[1,5],[1,5]]

    if name == 'I.15.3x_orig':
        symbol = x, u, t, c = symbols('x u t c')
        expr = (x-u*t)/sqrt(1-u**2/c**2)
        f = lambda x: (x[:,[0]] - x[:,[1]]*x[:,[2]])/torch.sqrt(1-x[:,[1]]**2/x[:,[3]]**2)
        ranges = [[5,10],[1,2],[1,2],[3,20]]

    if name == 'I.16.6_orig':
        symbol = u, v, c = symbols('u v c')
        expr = (u+v)/(1+u*v/c**2)
        f = lambda x: (x[:,[0]]+x[:,[1]])/(1+x[:,[0]]*x[:,[1]]/x[:,[2]]**2)
        ranges = [[1,5],[1,5],[1,5]]

    if name == 'I.18.4_orig':
        symbol = m1, r1, m2, r2 = symbols('m1 r1 m2 r2')
        expr = (m1*r1+m2*r2)/(m1+m2)
        f = lambda x: (x[:,[0]]*x[:,[1]]+x[:,[2]]*x[:,[3]])/(x[:,[0]]+x[:,[2]])
        ranges = [[1,5],[1,5],[1,5],[1,5]]

    if name == 'I.27.6_orig':
        symbol = d1, d2, n = symbols('d1 d2 n')
        expr = 1/(1/d1+n/d2)
        f = lambda x: 1/(1/x[:,[0]]+x[:,[2]]/x[:,[1]])
        ranges = [[1,5],[1,5],[1,5]]

    if name == 'I.26.2_orig':
        symbol = n, theta2 = symbols('n theta2')
        expr = asin(n*sin(theta2))
        f = lambda x: torch.arcsin(x[:,[0]]*torch.sin(x[:,[1]]))
        ranges = [[0,1],[1,5]]

    if name == 'I.29.16_orig':
        symbol = x1, x2, theta1, theta2 = symbols('x1 x2 theta1 theta2')
        expr = sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))
        f = lambda x: torch.sqrt(x[:,[0]]**2+x[:,[1]]**2-2*x[:,[0]]*x[:,[1]]*torch.cos(x[:,[2]]-x[:,[3]]))
        ranges = [[1,5],[1,5],[1,5],[1,5]]

    if name == 'I.30.3_orig':
        symbol = I0, n, theta = symbols('I_0 n theta')
        expr = I0 * sin(n*theta/2)**2 / sin(theta/2) ** 2
        f = lambda x: x[:,[0]] * torch.sin(x[:,[1]]*x[:,[2]]/2)**2 / torch.sin(x[:,[2]]/2)**2
        ranges = [[1,5],[1,5],[1,5]]

    if name == 'I.30.5_orig':
        symbol = lamb, n, d = symbols('lambda n d')
        expr = asin(lamb/(n*d))
        f = lambda x: torch.arcsin(x[:,[0]]/(x[:,[1]]*x[:,[2]]))
        ranges = [[1,2],[1,5],[2,5]]

    if name == 'I.37.4_orig':
        symbol = I1, I2, delta = symbols('I_1 I_2 delta')
        expr = I1 + I2 + 2*sqrt(I1*I2)*cos(delta)
        f = lambda x: x[:,[0]] + x[:,[1]] + 2*torch.sqrt(x[:,[0]]*x[:,[1]])*torch.cos(x[:,[2]])
        ranges = [[1,5],[1,5],[1,5]]

    if name == 'I.40.1_orig':
        symbol = n0, m, g, x, kb, T = symbols('n_0 m g x k_b T')
        expr = n0 * exp(-m*g*x/(kb*T))
        f = lambda x: x[:,[0]] * torch.exp(-x[:,[1]]*x[:,[2]]*x[:,[3]]/(x[:,[4]]*x[:,[5]]))
        ranges = [[1,5],[1,5],[1,5],[1,5],[1,5],[1,5]]

    if name == 'I.44.4_orig':
        symbol = n, kb, T, V1, V2 = symbols('n k_b T V_1 V_2')
        expr = n*kb*T*log(V2/V1)
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]*torch.log(x[:,[4]]/x[:,[3]])
        ranges = [[1,5],[1,5],[1,5],[1,5],[1,5]]

    if name == 'I.50.26_orig':
        symbol = x1, alpha, omega, t = symbols('x_1 alpha omega t')
        expr = x1*(cos(omega*t)+alpha*cos(omega*t)**2)
        f = lambda x: x[:,[0]]*(torch.cos(x[:,[2]]*x[:,[3]])+x[:,[1]]*torch.cos(x[:,[2]]*x[:,[3]])**2)
        ranges = [[1,3],[1,3],[1,3],[1,3]]

    if name == 'II.2.42_orig':
        symbol = kappa, T1, T2, A, d = symbols('kappa T_1 T_2 A d')
        expr = kappa*(T2-T1)*A/d
        f = lambda x: x[:,[0]]*(x[:,[2]]-x[:,[1]])*x[:,[3]]/x[:,[4]]
        ranges = [[1,5],[1,5],[1,5],[1,5],[1,5]]

    if name == 'II.6.15a_orig':
        symbol = eps, pd, z, x, y, r = symbols('epsilon p_d z x y r')
        expr = 3/(4*pi*eps)*pd*z/r**5*sqrt(x**2+y**2)
        f = lambda x: 3/(4*tpi*x[:,[0]])*x[:,[1]]*x[:,[2]]/x[:,[5]]**5*torch.sqrt(x[:,[3]]**2+x[:,[4]]**2)
        ranges = [[1,3],[1,3],[1,3],[1,3],[1,3],[1,3]]

    if name == 'II.11.17_orig':
        symbol = n0, pd, Ef, theta, kb, T = symbols('n_0 p_d E_f theta k_b T')
        expr = n0*(1+pd*Ef*cos(theta)/(kb*T))
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*x[:,[2]]*torch.cos(x[:,[3]])/(x[:,[4]]*x[:,[5]]))
        ranges = [[1,3],[1,3],[1,3],[1,3],[1,3],[1,3]]

    if name == 'II.11.27_orig':
        symbol = n, alpha, eps, Ef = symbols('n alpha epsilon E_f')
        expr = n*alpha/(1-n*alpha/3)*eps*Ef
        f = lambda x: x[:,[0]]*x[:,[1]]/(1-x[:,[0]]*x[:,[1]]/3)*x[:,[2]]*x[:,[3]]
        ranges = [[0,1],[0,1],[1,2],[1,2]]

    if name == 'II.35.18_orig':
        symbol = n0, mu, B, kb, T = symbols('n0 mu B k_b T')
        expr = n0/(exp(mu*B/(kb*T))+exp(-mu*B/(kb*T)))
        f = lambda x: x[:,[0]]/(torch.exp(x[:,[1]]*x[:,[2]]/(x[:,[3]]*x[:,[4]]))+torch.exp(-x[:,[1]]*x[:,[2]]/(x[:,[3]]*x[:,[4]])))
        ranges = [[1,3],[1,3],[1,3],[1,3],[1,3]]

    if name == 'II.36.38_orig':
        symbol = mu, B, kb, T, alpha, M, eps, c = symbols('mu B k_b T alpha M epsilon c')
        expr = mu*B/(kb*T) + mu*alpha*M/(eps*c**2*kb*T)
        f = lambda x: x[:,[0]]*x[:,[1]]/(x[:,[2]]*x[:,[3]]) + x[:,[0]]*x[:,[4]]*x[:,[5]]/(x[:,[6]]*x[:,[7]]**2*x[:,[2]]*x[:,[3]])
        ranges = [[1,3],[1,3],[1,3],[1,3],[1,3],[1,3],[1,3],[1,3]]

    if name == 'II.38.3_orig':
        symbol = Y, A, x, d = symbols('Y A x d')
        expr = Y*A*x/d
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]
        ranges = [[1,5],[1,5],[1,5],[1,5]]

    if name == 'III.9.52_orig':
        symbol = pd, Ef, t, hbar, omega, omega0 = symbols('p_d E_f t hbar omega omega_0')
        expr = pd*Ef*t/hbar*sin((omega-omega0)*t/2)**2/((omega-omega0)*t/2)**2
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]*torch.sin((x[:,[4]]-x[:,[5]])*x[:,[2]]/2)**2/((x[:,[4]]-x[:,[5]])*x[:,[2]]/2)**2
        ranges = [[1,3],[1,3],[1,3],[1,3],[1,5],[1,5]]

    if name == 'III.17.37_orig':
        symbol = beta, alpha, theta = symbols('beta alpha theta')
        expr = beta * (1+alpha*cos(theta))
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*torch.cos(x[:,[2]]))
        ranges = [[1,5],[1,5],[1,5]]


    # FORMULAS AND RANGES ACCORDING TO KAN PAPER

    if name == 'I.6.2' or name == 2:
        symbol = theta, sigma = symbols('theta sigma')
        expr = exp(-theta**2/(2*sigma**2))/sqrt(2*pi*sigma**2)
        f = lambda x: torch.exp(-x[:,[0]]**2/(2*x[:,[1]]**2))/torch.sqrt(2*tpi*x[:,[1]]**2)
        ranges = [[-1,1],[0.5,2]]

    if name == 'I.6.2b' or name == 3:
        symbol = theta, theta1, sigma = symbols('theta theta1 sigma')
        expr = exp(-(theta-theta1)**2/(2*sigma**2))/sqrt(2*pi*sigma**2)
        f = lambda x: torch.exp(-(x[:,[0]]-x[:,[1]])**2/(2*x[:,[2]]**2))/torch.sqrt(2*tpi*x[:,[2]]**2)
        ranges = [[-1.5,1.5],[-1.5,1.5],[0.5,2]]

    if name == 'I.9.18' or name == 5:
        symbol = G, m1, m2, x1, x2, y1, y2, z1, z2 = symbols('G m1 m2 x1 x2 y1 y2 z1 z2')
        expr = G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/((x[:,[3]]-x[:,[4]])**2+(x[:,[5]]-x[:,[6]])**2+(x[:,[7]]-x[:,[8]])**2)
        ranges = [[-1,1],[-1,1],[-1,1],[-1,-0.5],[0.5,1],[-1,-0.5],[0.5,1],[-1,-0.5],[0.5,1]]

    if name == 'I.12.11' or name == 12:
        symbol = q, Ef, B, v, theta = symbols('q E_f B v theta')
        expr = q*(Ef + B*v*sin(theta))
        f = lambda x: x[:,[0]]*(x[:,[1]]+x[:,[2]]*x[:,[3]]*torch.sin(x[:,[4]]))
        ranges = [[-1,1],[-1,1],[-1,1],[-1,1],[0,2*tpi]]

    if name == 'I.13.12' or name == 14:
        symbol = G, m1, m2, r1, r2 = symbols('G m1 m2 r1 r2')
        expr = G*m1*m2*(1/r2-1/r1)
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]*(1/x[:,[4]]-1/x[:,[3]])
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2]]

    if name == 'I.15.3x' or name == 17:
        symbol = x, u, t, c = symbols('x u t c')
        expr = (x-u*t)/sqrt(1-u**2/c**2)
        f = lambda x: (x[:,[0]] - x[:,[1]]*x[:,[2]])/torch.sqrt(1-x[:,[1]]**2/x[:,[3]]**2)
        ranges = [[-1,1],[-1,1],[-1,1],[1,2]]

    if name == 'I.16.6' or name == 20:
        symbol = u, v, c = symbols('u v c')
        expr = (u+v)/(1+u*v/c**2)
        f = lambda x: (x[:,[0]] + x[:,[1]])/(1+x[:,[0]]*x[:,[1]]/x[:,[2]]**2)
        ranges = [[-0.8,0.8],[-0.8,0.8],[1,2]]

    if name == 'I.18.4' or name == 22:
        symbol = r, F, theta = symbols('r F theta')
        expr = r*F*sin(theta)
        f = lambda x: x[:,[0]]*x[:,[1]]*torch.sin(x[:,[2]])
        ranges = [[-1,1],[-1,1],[0,2*tpi]]

    if name == 'I.26.2' or name == 26:
        symbol = n, theta2 = symbols('n theta2')
        expr = asin(n*sin(theta2))
        f = lambda x: torch.arcsin(x[:,[0]]*torch.sin(x[:,[1]]))
        ranges = [[0,0.99],[0,2*tpi]]

    if name == 'I.27.6' or name == 27:
        symbol = d1, d2, n = symbols('d1 d2 n')
        expr = 1/(1/d1+n/d2)
        f = lambda x: 1/(1/x[:,[0]]+x[:,[2]]/x[:,[1]])
        ranges = [[0.5,2],[1,2],[0.5,2]]

    if name == 'I.29.16' or name == 29:
        symbol = x1, x2, theta1, theta2 = symbols('x1 x2 theta1 theta2')
        expr = sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))
        f = lambda x: torch.sqrt(x[:,[0]]**2+x[:,[1]]**2-2*x[:,[0]]*x[:,[1]]*torch.cos(x[:,[2]]-x[:,[3]]))
        ranges = [[-1,1],[-1,1],[0,2*tpi],[0,2*tpi]]

    if name == 'I.30.3' or name == 30:
        symbol = I0, n, theta = symbols('I_0 n theta')
        expr = I0 * sin(n*theta/2)**2 / sin(theta/2) ** 2
        f = lambda x: x[:,[0]] * torch.sin(x[:,[1]]*x[:,[2]]/2)**2 / torch.sin(x[:,[2]]/2)**2
        ranges = [[0,1],[0,4],[0.4*tpi,1.6*tpi]]

    if name == 'I.30.5' or name == 31:
        symbol = lamb, n, d = symbols('lambda n d')
        expr = asin(lamb/(n*d))
        f = lambda x: torch.arcsin(x[:,[0]]/(x[:,[1]]*x[:,[2]]))
        ranges = [[-1,1],[1,1.5],[1,1.5]]

    if name == 'I.37.4' or name == 38:
        symbol = I1, I2, delta = symbols('I_1 I_2 delta')
        expr = I1 + I2 + 2*sqrt(I1*I2)*cos(delta)
        f = lambda x: x[:,[0]] + x[:,[1]] + 2*torch.sqrt(x[:,[0]]*x[:,[1]])*torch.cos(x[:,[2]])
        ranges = [[0.1,1],[0.1,1],[0,2*tpi]]

    if name == 'I.40.1' or name == 43:
        symbol = n0, m, g, x, kb, T = symbols('n_0 m g x k_b T')
        expr = n0 * exp(-m*g*x/(kb*T))
        f = lambda x: x[:,[0]] * torch.exp(-x[:,[1]]*x[:,[2]]*x[:,[3]]/(x[:,[4]]*x[:,[5]]))
        ranges = [[0,1],[-1,1],[-1,1],[-1,1],[1,2],[1,2]]

    if name == 'I.44.4' or name == 48:
        symbol = n, kb, T, V1, V2 = symbols('n k_b T V_1 V_2')
        expr = n*kb*T*log(V2/V1)
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]*torch.log(x[:,[4]]/x[:,[3]])
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2]]

    if name == 'I.50.26' or name == 51:
        symbol = x1, alpha, omega, t = symbols('x_1 alpha omega t')
        expr = x1*(cos(omega*t)+alpha*cos(omega*t)**2)
        f = lambda x: x[:,[0]]*(torch.cos(x[:,[2]]*x[:,[3]])+x[:,[1]]*torch.cos(x[:,[2]]*x[:,[3]])**2)
        ranges = [[0,1],[0,1],[0,2*tpi],[0,1]]
        
    if name == 'II.2.42' or name == 52:
        symbol = kappa, T1, T2, A, d = symbols('kappa T_1 T_2 A d')
        expr = kappa*(T2-T1)*A/d
        f = lambda x: x[:,[0]]*(x[:,[2]]-x[:,[1]])*x[:,[3]]/x[:,[4]]
        ranges = [[0,1],[0,1],[0,1],[0,1],[0.5,2]]

    if name == 'II.6.15a' or name == 56:
        symbol = eps, pd, z, x, y, r = symbols('epsilon p_d z x y r')
        expr = 3/(4*pi*eps)*pd*z/r**5*sqrt(x**2+y**2)
        f = lambda x: 3/(4*tpi*x[:,[0]])*x[:,[1]]*x[:,[2]]/x[:,[5]]**5*torch.sqrt(x[:,[3]]**2+x[:,[4]]**2)
        ranges = [[0.5,2],[0,1],[0,1],[0,1],[0,1],[0.5,2]]

    if name == 'II.11.17' or name == 62:
        symbol = n0, pd, Ef, theta, kb, T = symbols('n_0 p_d E_f theta k_b T')
        expr = n0*(1+pd*Ef*cos(theta)/(kb*T))
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*x[:,[2]]*torch.cos(x[:,[3]])/(x[:,[4]]*x[:,[5]]))
        ranges = [[0,1],[-1,1],[-1,1],[0,2*tpi],[0.5,2],[0.5,2]]

    if name == 'II.35.18' or name == 80:
        symbol = n0, mu, B, kb, T = symbols('n0 mu B k_b T')
        expr = n0/(exp(mu*B/(kb*T))+exp(-mu*B/(kb*T)))
        f = lambda x: x[:,[0]]/(torch.exp(x[:,[1]]*x[:,[2]]/(x[:,[3]]*x[:,[4]]))+torch.exp(-x[:,[1]]*x[:,[2]]/(x[:,[3]]*x[:,[4]])))
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2]]

    if name == 'II.38.3' or name == 84:
        symbol = Y, A, x, d = symbols('Y A x d')
        expr = Y*A*x/d
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]
        ranges = [[0,1],[0,1],[0,1],[0.5,2]]

    if name == 'III.9.52' or name == 90:
        symbol = pd, Ef, t, hbar, omega, omega0 = symbols('p_d E_f t hbar omega omega_0')
        expr = pd*Ef*t/hbar*sin((omega-omega0)*t/2)**2/((omega-omega0)*t/2)**2
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]*torch.sin((x[:,[4]]-x[:,[5]])*x[:,[2]]/2)**2/((x[:,[4]]-x[:,[5]])*x[:,[2]]/2)**2
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0,tpi],[0,tpi]]
        
    if name == 'III.10.19' or name == 91:
        symbol = mu, Bx, By, Bz = symbols('mu B_x B_y B_z')
        expr = mu*sqrt(Bx**2+By**2+Bz**2)
        f = lambda x: x[:,[0]]*torch.sqrt(x[:,[1]]**2+x[:,[2]]**2+x[:,[3]]**2)
        ranges = [[0,1],[0,1],[0,1],[0,1]]

    if name == 'III.17.37' or name == 98:
        symbol = beta, alpha, theta = symbols('beta alpha theta')
        expr = beta * (1+alpha*cos(theta))
        f = lambda x: x[:,[0]]*(1+x[:,[1]]*torch.cos(x[:,[2]]))
        ranges = [[0,1],[0,1],[0,2*tpi]]


    # REST OF THE FORMULAS
    
    if name == 'I.6.20a' or name == 1:
        symbol = theta = symbols('theta')
        symbol = [symbol]
        expr = exp(-theta**2/2)/sqrt(2*pi)
        f = lambda x: torch.exp(-x[:,[0]]**2/2)/torch.sqrt(2*tpi)
        ranges = [[-3,3]]

    if name == 'I.8.4' or name == 4:
        symbol = x1, x2, y1, y2 = symbols('x1 x2 y1 y2')
        expr = sqrt((x2-x1)**2+(y2-y1)**2)
        f = lambda x: torch.sqrt((x[:,[1]]-x[:,[0]])**2+(x[:,[3]]-x[:,[2]])**2)
        ranges = [[-1,1],[-1,1],[-1,1],[-1,1]]
        
    if name == 'I.10.7' or name == 6:
        symbol = m0, v, c = symbols('m0 v c')
        expr = m0/sqrt(1-v**2/c**2)
        f = lambda x: x[:,[0]]/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)
        ranges = [[0,1],[0,1],[1,2]]
        
    if name == 'I.11.19' or name == 7:
        symbol = x1, y1, x2, y2, x3, y3 = symbols('x1 y1 x2 y2 x3 y3')
        expr = x1*y1 + x2*y2 + x3*y3
        f = lambda x: x[:,[0]]*x[:,[1]] + x[:,[2]]*x[:,[3]] + x[:,[4]]*x[:,[5]]
        ranges = [-1,1]
    
    if name == 'I.12.1' or name == 8:
        symbol = mu, Nn = symbols('mu N_n')
        expr = mu * Nn
        f = lambda x: x[:,[0]]*x[:,[1]]
        ranges = [-1,1]
        
    if name == 'I.12.2' or name == 9:
        symbol = q1, q2, eps, r = symbols('q1 q2 epsilon r')
        expr = q1*q2/(4*pi*eps*r**2)
        f = lambda x: x[:,[0]]*x[:,[1]]/(4*tpi*x[:,[2]]*x[:,[3]]**2)
        ranges = [[-1,1],[-1,1],[0.5,2],[0.5,2]]
        
    if name == 'I.12.4' or name == 10:
        symbol = q1, eps, r = symbols('q1 epsilon r')
        expr = q1/(4*pi*eps*r**2)
        f = lambda x: x[:,[0]]/(4*tpi*x[:,[1]]*x[:,[2]]**2)
        ranges = [[-1,1],[0.5,2],[0.5,2]]
        
    if name == 'I.12.5' or name == 11:
        symbol = q2, Ef = symbols('q2, E_f')
        expr = q2*Ef
        f = lambda x: x[:,[0]]*x[:,[1]]
        ranges = [-1,1]
        
    
        
    if name == 'I.13.4' or name == 13:
        symbol = m, v, u, w = symbols('m u v w')
        expr = 1/2*m*(v**2+u**2+w**2)
        f = lambda x: 1/2*x[:,[0]]*(x[:,[1]]**2+x[:,[2]]**2+x[:,[3]]**2)
        ranges = [[-1,1],[-1,1],[-1,1],[-1,1]]
        
    
        
    if name == 'I.14.3' or name == 15:
        symbol = m, g, z = symbols('m g z')
        expr = m*g*z
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]
        ranges = [[0,1],[0,1],[-1,1]]
        
    if name == 'I.14.4' or name == 16:
        symbol = ks, x = symbols('k_s x')
        expr = 1/2*ks*x**2
        f = lambda x: 1/2*x[:,[0]]*x[:,[1]]**2
        ranges = [[0,1],[-1,1]]
        
    
        
    if name == 'I.15.3t' or name == 18:
        symbol = t, u, x, c = symbols('t u x c')
        expr = (t-u*x/c**2)/sqrt(1-u**2/c**2)
        f = lambda x: (x[:,[0]] - x[:,[1]]*x[:,[2]]/x[:,[3]]**2)/torch.sqrt(1-x[:,[1]]**2/x[:,[3]]**2)
        ranges = [[-1,1],[-1,1],[-1,1],[1,2]]
        
    if name == 'I.15.10' or name == 19:
        symbol = m0, v, c = symbols('m0 v c')
        expr = m0*v/sqrt(1-v**2/c**2)
        f = lambda x: x[:,[0]]*x[:,[1]]/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)
        ranges = [[-1,1],[-0.9,0.9],[1.1,2]]
        
    
        
    if name == 'I.18.4' or name == 21:
        symbol = m1, r1, m2, r2 = symbols('m1 r1 m2 r2')
        expr = (m1*r1+m2*r2)/(m1+m2)
        f = lambda x: (x[:,[0]]*x[:,[1]]+x[:,[2]]*x[:,[3]])/(x[:,[0]]+x[:,[2]])
        ranges = [[0.5,1],[-1,1],[0.5,1],[-1,1]]
        
    
        
    if name == 'I.18.16' or name == 23:
        symbol = m, r, v, theta = symbols('m r v theta')
        expr = m*r*v*sin(theta)
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]*torch.sin(x[:,[3]])
        ranges = [[-1,1],[-1,1],[-1,1],[0,2*tpi]]
        
    if name == 'I.24.6' or name == 24:
        symbol = m, omega, omega0, x = symbols('m omega omega_0 x')
        expr = 1/4*m*(omega**2+omega0**2)*x**2
        f = lambda x: 1/4*x[:,[0]]*(x[:,[1]]**2+x[:,[2]]**2)*x[:,[3]]**2
        ranges = [[0,1],[-1,1],[-1,1],[-1,1]]
        
    if name == 'I.25.13' or name == 25:
        symbol = q, C = symbols('q C')
        expr = q/C
        f = lambda x: x[:,[0]]/x[:,[1]]
        ranges = [[-1,1],[0.5,2]]
        
    
        
    
    
    if name == 'I.29.4' or name == 28:
        symbol = omega, c = symbols('omega c')
        expr = omega/c
        f = lambda x: x[:,[0]]/x[:,[1]]
        ranges = [[0,1],[0.5,2]]
        
    
        
    
        
    
        
    if name == 'I.32.5' or name == 32:
        symbol = q, a, eps, c = symbols('q a epsilon c')
        expr = q**2*a**2/(eps*c**3)
        f = lambda x: x[:,[0]]**2*x[:,[1]]**2/(x[:,[2]]*x[:,[3]]**3)
        ranges = [[-1,1],[-1,1],[0.5,2],[0.5,2]]
        
    if name == 'I.32.17' or name == 33:
        symbol = eps, c, Ef, r, omega, omega0 = symbols('epsilon c E_f r omega omega_0')
        expr = nsimplify((1/2*eps*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega0**2)**2))
        f = lambda x: (1/2*x[:,[0]]*x[:,[1]]*x[:,[2]]**2)*(8*tpi*x[:,[3]]**2/3)*(x[:,[4]]**4/(x[:,[4]]**2-x[:,[5]]**2)**2)
        ranges = [[0,1],[0,1],[-1,1],[0,1],[0,1],[1,2]]
        
    if name == 'I.34.8' or name == 34:
        symbol = q, V, B, p = symbols('q V B p')
        expr = q*V*B/p
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]
        ranges = [[-1,1],[-1,1],[-1,1],[0.5,2]]
        
    if name == 'I.34.10' or name == 35:
        symbol = omega0, v, c = symbols('omega_0 v c')
        expr = omega0/(1-v/c)
        f = lambda x: x[:,[0]]/(1-x[:,[1]]/x[:,[2]])
        ranges = [[0,1],[0,0.9],[1.1,2]]
        
    if name == 'I.34.14' or name == 36:
        symbol = omega0, v, c = symbols('omega_0 v c')
        expr = omega0 * (1+v/c)/sqrt(1-v**2/c**2)
        f = lambda x: x[:,[0]]*(1+x[:,[1]]/x[:,[2]])/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)
        ranges = [[0,1],[-0.9,0.9],[1.1,2]]
        
    if name == 'I.34.27' or name == 37:
        symbol = hbar, omega = symbols('hbar omega')
        expr = hbar * omega
        f = lambda x: x[:,[0]]*x[:,[1]]
        ranges = [[-1,1],[-1,1]]
        
    
        
    if name == 'I.38.12' or name == 39:
        symbol = eps, hbar, m, q = symbols('epsilon hbar m q')
        expr = 4*pi*eps*hbar**2/(m*q**2)
        f = lambda x: 4*tpi*x[:,[0]]*x[:,[1]]**2/(x[:,[2]]*x[:,[3]]**2)
        ranges = [[0,1],[0,1],[0.5,2],[0.5,2]]
        
    if name == 'I.39.10' or name == 40:
        symbol = pF, V = symbols('p_F V')
        expr = 3/2 * pF * V
        f = lambda x: 3/2 * x[:,[0]] * x[:,[1]]
        ranges = [[0,1],[0,1]]
        
    if name == 'I.39.11' or name == 41:
        symbol = gamma, pF, V = symbols('gamma p_F V')
        expr = pF * V/(gamma - 1)
        f = lambda x: 1/(x[:,[0]]-1) * x[:,[1]] * x[:,[2]]
        ranges = [[1.5,3],[0,1],[0,1]]
        
    if name == 'I.39.22' or name == 42:
        symbol = n, kb, T, V = symbols('n k_b T V')
        expr = n*kb*T/V
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]
        ranges = [[0,1],[0,1],[0,1],[0.5,2]]
        
    
        
    if name == 'I.41.16' or name == 44:
        symbol = hbar, omega, c, kb, T = symbols('hbar omega c k_b T')
        expr = hbar * omega**3/(pi**2*c**2*(exp(hbar*omega/(kb*T))-1))
        f = lambda x: x[:,[0]]*x[:,[1]]**3/(tpi**2*x[:,[2]]**2*(torch.exp(x[:,[0]]*x[:,[1]]/(x[:,[3]]*x[:,[4]]))-1))
        ranges = [[0.5,1],[0.5,1],[0.5,2],[0.5,2],[0.5,2]]
        
    if name == 'I.43.16' or name == 45:
        symbol = mu, q, Ve, d = symbols('mu q V_e d')
        expr = mu*q*Ve/d
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]
        ranges = [[0,1],[0,1],[0,1],[0.5,2]]
        
    if name == 'I.43.31' or name == 46:
        symbol = mu, kb, T = symbols('mu k_b T')
        expr = mu*kb*T
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]
        ranges = [[0,1],[0,1],[0,1]]
    
    if name == 'I.43.43' or name == 47:
        symbol = gamma, kb, v, A = symbols('gamma k_b v A')
        expr = kb*v/A/(gamma-1)
        f = lambda x: 1/(x[:,[0]]-1)*x[:,[1]]*x[:,[2]]/x[:,[3]]
        ranges = [[1.5,3],[0,1],[0,1],[0.5,2]]
        
    
        
    if name == 'I.47.23' or name == 49:
        symbol = gamma, p, rho = symbols('gamma p rho')
        expr = sqrt(gamma*p/rho)
        f = lambda x: torch.sqrt(x[:,[0]]*x[:,[1]]/x[:,[2]])
        ranges = [[0.1,1],[0.1,1],[0.5,2]]
        
    if name == 'I.48.20' or name == 50:
        symbol = m, v, c = symbols('m v c')
        expr = m*c**2/sqrt(1-v**2/c**2)
        f = lambda x: x[:,[0]]*x[:,[2]]**2/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)
        ranges = [[0,1],[-0.9,0.9],[1.1,2]]
        
    
        
    if name == 'II.3.24' or name == 53:
        symbol = P, r = symbols('P r')
        expr = P/(4*pi*r**2)
        f = lambda x: x[:,[0]]/(4*tpi*x[:,[1]]**2)
        ranges = [[0,1],[0.5,2]]
        
    if name == 'II.4.23' or name == 54:
        symbol = q, eps, r = symbols('q epsilon r')
        expr = q/(4*pi*eps*r)
        f = lambda x: x[:,[0]]/(4*tpi*x[:,[1]]*x[:,[2]])
        ranges = [[0,1],[0.5,2],[0.5,2]]
        
    if name == 'II.6.11' or name == 55:
        symbol = eps, pd, theta, r = symbols('epsilon p_d theta r')
        expr = 1/(4*pi*eps)*pd*cos(theta)/r**2
        f = lambda x: 1/(4*tpi*x[:,[0]])*x[:,[1]]*torch.cos(x[:,[2]])/x[:,[3]]**2
        ranges = [[0.5,2],[0,1],[0,2*tpi],[0.5,2]]
        
    
    
    if name == 'II.6.15b' or name == 57:
        symbol = eps, pd, r, theta = symbols('epsilon p_d r theta')
        expr = 3/(4*pi*eps)*pd/r**3*cos(theta)*sin(theta)
        f = lambda x: 3/(4*tpi*x[:,[0]])*x[:,[1]]/x[:,[2]]**3*torch.cos(x[:,[3]])*torch.sin(x[:,[3]])
        ranges = [[0.5,2],[0,1],[0.5,2],[0,2*tpi]]
        
    if name == 'II.8.7' or name == 58:
        symbol = q, eps, d = symbols('q epsilon d')
        expr = 3/5*q**2/(4*pi*eps*d)
        f = lambda x: 3/5*x[:,[0]]**2/(4*tpi*x[:,[1]]*x[:,[2]])
        ranges = [[0,1],[0.5,2],[0.5,2]]
        
    if name == 'II.8.31' or name == 59:
        symbol = eps, Ef = symbols('epsilon E_f')
        expr = 1/2*eps*Ef**2
        f = lambda x: 1/2*x[:,[0]]*x[:,[1]]**2
        ranges = [[0,1],[0,1]]
        
    if name == 'I.10.9' or name == 60:
        symbol = sigma, eps, chi = symbols('sigma epsilon chi')
        expr = sigma/eps/(1+chi)
        f = lambda x: x[:,[0]]/x[:,[1]]/(1+x[:,[2]])
        ranges = [[0,1],[0.5,2],[0,1]]
        
    if name == 'II.11.3' or name == 61:
        symbol = q, Ef, m, omega0, omega = symbols('q E_f m omega_o omega')
        expr = q*Ef/(m*(omega0**2-omega**2))
        f = lambda x: x[:,[0]]*x[:,[1]]/(x[:,[2]]*(x[:,[3]]**2-x[:,[4]]**2))
        ranges = [[0,1],[0,1],[0.5,2],[1.5,3],[0,1]]
        
    
        
        
    if name == 'II.11.20' or name == 63:
        symbol = n, pd, Ef, kb, T = symbols('n p_d E_f k_b T')
        expr = n*pd**2*Ef/(3*kb*T)
        f = lambda x: x[:,[0]]*x[:,[1]]**2*x[:,[2]]/(3*x[:,[3]]*x[:,[4]])
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2]]
        
    if name == 'II.11.27' or name == 64:
        symbol = n, alpha, eps, Ef = symbols('n alpha epsilon E_f')
        expr = n*alpha/(1-n*alpha/3)*eps*Ef
        f = lambda x: x[:,[0]]*x[:,[1]]/(1-x[:,[0]]*x[:,[1]]/3)*x[:,[2]]*x[:,[3]]
        ranges = [[0,1],[0,2],[0,1],[0,1]]
        
    if name == 'II.11.28' or name == 65:
        symbol = n, alpha = symbols('n alpha')
        expr = 1 + n*alpha/(1-n*alpha/3)
        f = lambda x: 1 + x[:,[0]]*x[:,[1]]/(1-x[:,[0]]*x[:,[1]]/3)
        ranges = [[0,1],[0,2]]
        
    if name == 'II.13.17' or name == 66:
        symbol = eps, c, l, r = symbols('epsilon c l r')
        expr = 1/(4*pi*eps*c**2)*(2*l/r)
        f = lambda x: 1/(4*tpi*x[:,[0]]*x[:,[1]]**2)*(2*x[:,[2]]/x[:,[3]])
        ranges = [[0.5,2],[0.5,2],[0,1],[0.5,2]]
        
    if name == 'II.13.23' or name == 67:
        symbol = rho, v, c = symbols('rho v c')
        expr = rho/sqrt(1-v**2/c**2)
        f = lambda x: x[:,[0]]/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)
        ranges = [[0,1],[0,1],[1,2]]
        
    if name == 'II.13.34' or name == 68:
        symbol = rho, v, c = symbols('rho v c')
        expr = rho*v/sqrt(1-v**2/c**2)
        f = lambda x: x[:,[0]]*x[:,[1]]/torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)
        ranges = [[0,1],[0,1],[1,2]]
        
    if name == 'II.15.4' or name == 69:
        symbol = muM, B, theta = symbols('mu_M B theta')
        expr = - muM * B * cos(theta)
        f = lambda x: - x[:,[0]]*x[:,[1]]*torch.cos(x[:,[2]])
        ranges = [[0,1],[0,1],[0,2*tpi]]
        
    if name == 'II.15.5' or name == 70:
        symbol = pd, Ef, theta = symbols('p_d E_f theta')
        expr = - pd * Ef * cos(theta)
        f = lambda x: - x[:,[0]]*x[:,[1]]*torch.cos(x[:,[2]])
        ranges = [[0,1],[0,1],[0,2*tpi]]
        
    if name == 'II.21.32' or name == 71:
        symbol = q, eps, r, v, c = symbols('q epsilon r v c')
        expr = q/(4*pi*eps*r*(1-v/c))
        f = lambda x: x[:,[0]]/(4*tpi*x[:,[1]]*x[:,[2]]*(1-x[:,[3]]/x[:,[4]]))
        ranges = [[0,1],[0.5,2],[0.5,2],[0,1],[1,2]]
        
    if name == 'II.24.17' or name == 72:
        symbol = omega, c, d = symbols('omega c d')
        expr = sqrt(omega**2/c**2-pi**2/d**2)
        f = lambda x: torch.sqrt(x[:,[0]]**2/x[:,[1]]**2-tpi**2/x[:,[2]]**2)
        ranges = [[1,1.5],[0.75,1],[1*tpi,1.5*tpi]]
        
    if name == 'II.27.16' or name == 73:
        symbol = eps, c, Ef = symbols('epsilon c E_f')
        expr = eps * c * Ef**2
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]**2
        ranges = [[0,1],[0,1],[-1,1]]
        
    if name == 'II.27.18' or name == 74:
        symbol = eps, Ef = symbols('epsilon E_f')
        expr = eps * Ef**2
        f = lambda x: x[:,[0]]*x[:,[1]]**2
        ranges = [[0,1],[-1,1]]
        
    if name == 'II.34.2a' or name == 75:
        symbol = q, v, r = symbols('q v r')
        expr = q*v/(2*pi*r)
        f = lambda x: x[:,[0]]*x[:,[1]]/(2*tpi*x[:,[2]])
        ranges = [[0,1],[0,1],[0.5,2]]
        
    if name == 'II.34.2' or name == 76:
        symbol = q, v, r = symbols('q v r')
        expr = q*v*r/2
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/2
        ranges = [[0,1],[0,1],[0,1]]
        
    if name == 'II.34.11' or name == 77:
        symbol = g, q, B, m = symbols('g q B m')
        expr = g*q*B/(2*m)
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]/(2*x[:,[3]])
        ranges = [[0,1],[0,1],[0,1],[0.5,2]]
        
    if name == 'II.34.29a' or name == 78:
        symbol = q, h, m = symbols('q h m')
        expr = q*h/(4*pi*m)
        f = lambda x: x[:,[0]]*x[:,[1]]/(4*tpi*x[:,[2]])
        ranges = [[0,1],[0,1],[0.5,2]]
        
    if name == 'II.34.29b' or name == 79:
        symbol = g, mu, B, J, hbar = symbols('g mu B J hbar')
        expr = g*mu*B*J/hbar
        f = lambda x: x[:,[0]]*x[:,[1]]*x[:,[2]]*x[:,[3]]/x[:,[4]]
        ranges = [[0,1],[0,1],[0,1],[0,1],[0.5,2]]
        
    
        
    if name == 'II.35.21' or name == 81:
        symbol = n, mu, B, kb, T = symbols('n mu B k_b T')
        expr = n*mu*tanh(mu*B/(kb*T))
        f = lambda x: x[:,[0]]*x[:,[1]]*torch.tanh(x[:,[1]]*x[:,[2]]/(x[:,[3]]*x[:,[4]]))
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2]]
        
    if name == 'II.36.38' or name == 82:
        symbol = mu, B, kb, T, alpha, M, eps, c = symbols('mu B k_b T alpha M epsilon c')
        expr = mu*B/(kb*T) + mu*alpha*M/(eps*c**2*kb*T)
        f = lambda x: x[:,[0]]*x[:,[1]]/(x[:,[2]]*x[:,[3]]) + x[:,[0]]*x[:,[4]]*x[:,[5]]/(x[:,[6]]*x[:,[7]]**2*x[:,[2]]*x[:,[3]])
        ranges = [[0,1],[0,1],[0.5,2],[0.5,2],[0,1],[0,1],[0.5,2],[0.5,2]]
        
    if name == 'II.37.1' or name == 83:
        symbol = mu, chi, B = symbols('mu chi B')
        expr = mu*(1+chi)*B
        f = lambda x: x[:,[0]]*(1+x[:,[1]])*x[:,[2]]
        ranges = [[0,1],[0,1],[0,1]]
        
    
        
    if name == 'II.38.14' or name == 85:
        symbol = Y, sigma = symbols('Y sigma')
        expr = Y/(2*(1+sigma))
        f = lambda x: x[:,[0]]/(2*(1+x[:,[1]]))
        ranges = [[0,1],[0,1]]
        
    if name == 'III.4.32' or name == 86:
        symbol = hbar, omega, kb, T = symbols('hbar omega k_b T')
        expr = 1/(exp(hbar*omega/(kb*T))-1)
        f = lambda x: 1/(torch.exp(x[:,[0]]*x[:,[1]]/(x[:,[2]]*x[:,[3]]))-1)
        ranges = [[0.5,1],[0.5,1],[0.5,2],[0.5,2]]
        
    if name == 'III.4.33' or name == 87:
        symbol = hbar, omega, kb, T = symbols('hbar omega k_b T')
        expr = hbar*omega/(exp(hbar*omega/(kb*T))-1)
        f = lambda x: x[:,[0]]*x[:,[1]]/(torch.exp(x[:,[0]]*x[:,[1]]/(x[:,[2]]*x[:,[3]]))-1)
        ranges = [[0,1],[0,1],[0.5,2],[0.5,2]]
        
    if name == 'III.7.38' or name == 88:
        symbol = mu, B, hbar = symbols('mu B hbar')
        expr = 2*mu*B/hbar
        f = lambda x: 2*x[:,[0]]*x[:,[1]]/x[:,[2]]
        ranges = [[0,1],[0,1],[0.5,2]]
        
    if name == 'III.8.54' or name == 89:
        symbol = E, t, hbar = symbols('E t hbar')
        expr = sin(E*t/hbar)**2
        f = lambda x: torch.sin(x[:,[0]]*x[:,[1]]/x[:,[2]])**2
        ranges = [[0,2*tpi],[0,1],[0.5,2]]
        
    
        
    if name == 'III.12.43' or name == 92:
        symbol = n, hbar = symbols('n hbar')
        expr = n * hbar
        f = lambda x: x[:,[0]]*x[:,[1]]
        ranges = [[0,1],[0,1]]
        
    if name == 'III.13.18' or name == 93:
        symbol = E, d, k, hbar = symbols('E d k hbar')
        expr = 2*E*d**2*k/hbar
        f = lambda x: 2*x[:,[0]]*x[:,[1]]**2*x[:,[2]]/x[:,[3]]
        ranges = [[0,1],[0,1],[0,1],[0.5,2]]
        
    if name == 'III.14.14' or name == 94:
        symbol = I0, q, Ve, kb, T = symbols('I_0 q V_e k_b T')
        expr = I0 * (exp(q*Ve/(kb*T))-1)
        f = lambda x: x[:,[0]]*(torch.exp(x[:,[1]]*x[:,[2]]/(x[:,[3]]*x[:,[4]]))-1)
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2]]
        
    if name == 'III.15.12' or name == 95:
        symbol = U, k, d = symbols('U k d')
        expr = 2*U*(1-cos(k*d))
        f = lambda x: 2*x[:,[0]]*(1-torch.cos(x[:,[1]]*x[:,[2]]))
        ranges = [[0,1],[0,2*tpi],[0,1]]
        
    if name == 'III.15.14' or name == 96:
        symbol = hbar, E, d = symbols('hbar E d')
        expr = hbar**2/(2*E*d**2)
        f = lambda x: x[:,[0]]**2/(2*x[:,[1]]*x[:,[2]]**2)
        ranges = [[0,1],[0.5,2],[0.5,2]]
        
    if name == 'III.15.27' or name == 97:
        symbol = alpha, n, d = symbols('alpha n d')
        expr = 2*pi*alpha/(n*d)
        f = lambda x: 2*tpi*x[:,[0]]/(x[:,[1]]*x[:,[2]])
        ranges = [[0,1],[0.5,2],[0.5,2]]
        
    
        
    if name == 'III.19.51' or name == 99:
        symbol = m, q, eps, hbar, n = symbols('m q epsilon hbar n')
        expr = - m * q**4/(2*(4*pi*eps)**2*hbar**2)*1/n**2
        f = lambda x: - x[:,[0]]*x[:,[1]]**4/(2*(4*tpi*x[:,[2]])**2*x[:,[3]]**2)*1/x[:,[4]]**2
        ranges = [[0,1],[0,1],[0.5,2],[0.5,2],[0.5,2]]
        
    if name == 'III.21.20' or name == 100:
        symbol = rho, q, A, m = symbols('rho q A m')
        expr = - rho*q*A/m
        f = lambda x: - x[:,[0]]*x[:,[1]]*x[:,[2]]/x[:,[3]]
        ranges = [[0,1],[0,1],[0,1],[0.5,2]]
        
    if name == 'Rutherforld scattering' or name == 101:
        symbol = Z1, Z2, alpha, hbar, c, E, theta = symbols('Z_1 Z_2 alpha hbar c E theta')
        expr = (Z1*Z2*alpha*hbar*c/(4*E*sin(theta/2)**2))**2
        f = lambda x: (x[:,[0]]*x[:,[1]]*x[:,[2]]*x[:,[3]]*x[:,[4]]/(4*x[:,[5]]*torch.sin(x[:,[6]]/2)**2))**2
        ranges = [[0,1],[0,1],[0,1],[0,1],[0,1],[0.5,2],[0.1*tpi,0.9*tpi]]
        
    if name == 'Friedman equation' or name == 102:
        symbol = G, rho, kf, c, af = symbols('G rho k_f c a_f')
        expr = sqrt(8*pi*G/3*rho-kf*c**2/af**2)
        f = lambda x: torch.sqrt(8*tpi*x[:,[0]]/3*x[:,[1]] - x[:,[2]]*x[:,[3]]**2/x[:,[4]]**2)
        ranges = [[1,2],[1,2],[0,1],[0,1],[1,2]]
        
    if name == 'Compton scattering' or name == 103:
        symbol = E, m, c, theta = symbols('E m c theta')
        expr = E/(1+E/(m*c**2)*(1-cos(theta)))
        f = lambda x: x[:,[0]]/(1+x[:,[0]]/(x[:,[1]]*x[:,[2]]**2)*(1-torch.cos(x[:,[3]])))
        ranges = [[0,1],[0.5,2],[0.5,2],[0,2*tpi]]
        
    if name == 'Radiated gravitational wave power' or name == 104:
        symbol = G, c, m1, m2, r = symbols('G c m_1 m_2 r')
        expr = -32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5
        f = lambda x: -32/5*x[:,[0]]**4/x[:,[1]]**5*(x[:,[2]]*x[:,[3]])**2*(x[:,[2]]+x[:,[3]])/x[:,[4]]**5
        ranges = [[0,1],[0.5,2],[0,1],[0,1],[0.5,2]]
        
    if name == 'Relativistic aberration' or name == 105:
        symbol = theta2, v, c = symbols('theta_2 v c')
        expr = acos((cos(theta2)-v/c)/(1-v/c*cos(theta2)))
        f = lambda x: torch.arccos((torch.cos(x[:,[0]])-x[:,[1]]/x[:,[2]])/(1-x[:,[1]]/x[:,[2]]*torch.cos(x[:,[0]])))
        ranges = [[0,tpi],[0,1],[1,2]]
        
    if name == 'N-slit diffraction' or name == 106:
        symbol = I0, alpha, delta, N = symbols('I_0 alpha delta N')
        expr = I0 * (sin(alpha/2)/(alpha/2)*sin(N*delta/2)/sin(delta/2))**2
        f = lambda x: x[:,[0]] * (torch.sin(x[:,[1]]/2)/(x[:,[1]]/2)*torch.sin(x[:,[3]]*x[:,[2]]/2)/torch.sin(x[:,[2]]/2))**2
        ranges = [[0,1],[0.1*tpi,0.9*tpi],[0.1*tpi,0.9*tpi],[0.5,1]]
        
    if name == 'Goldstein 3.16' or name == 107:
        symbol = m, E, U, L, r = symbols('m E U L r')
        expr = sqrt(2/m*(E-U-L**2/(2*m*r**2)))
        f = lambda x: torch.sqrt(2/x[:,[0]]*(x[:,[1]]-x[:,[2]]-x[:,[3]]**2/(2*x[:,[0]]*x[:,[4]]**2)))
        ranges = [[1,2],[2,3],[0,1],[0,1],[1,2]]
        
    if name == 'Goldstein 3.55' or name == 108:
        symbol = m, kG, L, E, theta1, theta2 = symbols('m k_G L E theta_1 theta_2')
        expr = m*kG/L**2*(1+sqrt(1+2*E*L**2/(m*kG**2))*cos(theta1-theta2))
        f = lambda x: x[:,[0]]*x[:,[1]]/x[:,[2]]**2*(1+torch.sqrt(1+2*x[:,[3]]*x[:,[2]]**2/(x[:,[0]]*x[:,[1]]**2))*torch.cos(x[:,[4]]-x[:,[5]]))
        ranges = [[0.5,2],[0.5,2],[0.5,2],[0,1],[0,2*tpi],[0,2*tpi]]
        
    if name == 'Goldstein 3.64 (ellipse)' or name == 109:
        symbol = d, alpha, theta1, theta2 = symbols('d alpha theta_1 theta_2')
        expr = d*(1-alpha**2)/(1+alpha*cos(theta2-theta1))
        f = lambda x: x[:,[0]]*(1-x[:,[1]]**2)/(1+x[:,[1]]*torch.cos(x[:,[2]]-x[:,[3]]))
        ranges = [[0,1],[0,0.9],[0,2*tpi],[0,2*tpi]]
        
    if name == 'Goldstein 3.74 (Kepler)' or name == 110:
        symbol = d, G, m1, m2 = symbols('d G m_1 m_2')
        expr = 2*pi*d**(3/2)/sqrt(G*(m1+m2))
        f = lambda x: 2*tpi*x[:,[0]]**(3/2)/torch.sqrt(x[:,[1]]*(x[:,[2]]+x[:,[3]]))
        ranges = [[0,1],[0.5,2],[0.5,2],[0.5,2]]
        
    if name == 'Goldstein 3.99' or name == 111:
        symbol = eps, E, L, m, Z1, Z2, q = symbols('epsilon E L m Z_1 Z_2 q')
        expr = sqrt(1+2*eps**2*E*L**2/(m*(Z1*Z2*q**2)**2))
        f = lambda x: torch.sqrt(1+2*x[:,[0]]**2*x[:,[1]]*x[:,[2]]**2/(x[:,[3]]*(x[:,[4]]*x[:,[5]]*x[:,[6]]**2)**2))
        ranges = [[0,1],[0,1],[0,1],[0.5,2],[0.5,2],[0.5,2],[0.5,2]]
        
    if name == 'Goldstein 8.56' or name == 112:
        symbol = p, q, A, c, m, Ve = symbols('p q A c m V_e')
        expr = sqrt((p-q*A)**2*c**2+m**2*c**4) + q*Ve
        f = lambda x: torch.sqrt((x[:,[0]]-x[:,[1]]*x[:,[2]])**2*x[:,[3]]**2+x[:,[4]]**2*x[:,[3]]**4) + x[:,[1]]*x[:,[5]]
        ranges = [0,1]
        
    if name == 'Goldstein 12.80' or name == 113:
        symbol = m, p, omega, x, alpha, y = symbols('m p omega x alpha y')
        expr = 1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*y/x))
        f = lambda x: 1/(2*x[:,[0]]) * (x[:,[1]]**2+x[:,[0]]**2*x[:,[2]]**2*x[:,[3]]**2*(1+x[:,[4]]*x[:,[3]]/x[:,[5]]))
        ranges = [[0.5,2],[0,1],[0,1],[0,1],[0,1],[0.5,2]]
        
    if name == 'Jackson 2.11' or name == 114:
        symbol = q, eps, y, Ve, d = symbols('q epsilon y V_e d')
        expr = q/(4*pi*eps*y**2)*(4*pi*eps*Ve*d-q*d*y**3/(y**2-d**2)**2)
        f = lambda x: x[:,[0]]/(4*tpi*x[:,[1]]*x[:,x[:,[2]]]**2)*(4*tpi*x[:,[1]]*x[:,[3]]*x[:,[4]]-x[:,[0]]*x[:,[4]]*x[:,[2]]**3/(x[:,[2]]**2-x[:,[4]]**2)**2)
        ranges = [[0,1],[0.5,2],[1,2],[0,1],[0,1]]
        
    if name == 'Jackson 3.45' or name == 115:
        symbol = q, r, d, alpha = symbols('q r d alpha')
        expr = q/sqrt(r**2+d**2-2*d*r*cos(alpha))
        f = lambda x: x[:,[0]]/torch.sqrt(x[:,[1]]**2+x[:,[2]]**2-2*x[:,[1]]*x[:,[2]]*torch.cos(x[:,[3]]))
        ranges = [[0,1],[0,1],[0,1],[0,2*tpi]]
        
    if name == 'Jackson 4.60' or name == 116:
        symbol = Ef, theta, alpha, d, r = symbols('E_f theta alpha d r')
        expr = Ef * cos(theta) * ((alpha-1)/(alpha+2) * d**3/r**2 - r)
        f = lambda x: x[:,[0]] * torch.cos(x[:,[1]]) * ((x[:,[2]]-1)/(x[:,[2]]+2) * x[:,[3]]**3/x[:,[4]]**2 - x[:,[4]])
        ranges = [[0,1],[0,2*tpi],[0,2],[0,1],[0.5,2]]
        
    if name == 'Jackson 11.38 (Doppler)' or name == 117:
        symbol = omega, v, c, theta = symbols('omega v c theta')
        expr = sqrt(1-v**2/c**2)/(1+v/c*cos(theta))*omega
        f = lambda x: torch.sqrt(1-x[:,[1]]**2/x[:,[2]]**2)/(1+x[:,[1]]/x[:,[2]]*torch.cos(x[:,[3]]))*x[:,[0]]
        ranges = [[0,1],[0,1],[1,2],[0,2*tpi]]
        
    if name == 'Weinberg 15.2.1' or name == 118:
        symbol = G, c, kf, af, H = symbols('G c k_f a_f H')
        expr = 3/(8*pi*G)*(c**2*kf/af**2+H**2)
        f = lambda x: 3/(8*tpi*x[:,[0]])*(x[:,[1]]**2*x[:,[2]]/x[:,[3]]**2+x[:,[4]]**2)
        ranges = [[0.5,2],[0,1],[0,1],[0.5,2],[0,1]]
        
    if name == 'Weinberg 15.2.2' or name == 119:
        symbol = G, c, kf, af, H, alpha = symbols('G c k_f a_f H alpha')
        expr = -1/(8*pi*G)*(c**4*kf/af**2+c**2*H**2*(1-2*alpha))
        f = lambda x: -1/(8*tpi*x[:,[0]])*(x[:,[1]]**4*x[:,[2]]/x[:,[3]]**2 + x[:,[1]]**2*x[:,[4]]**2*(1-2*x[:,[5]]))
        ranges = [[0.5,2],[0,1],[0,1],[0.5,2],[0,1],[0,1]]
        
    if name == 'Schwarz 13.132 (Klein-Nishina)' or name == 120:
        symbol = alpha, hbar, m, c, omega0, omega, theta = symbols('alpha hbar m c omega_0 omega theta')
        expr = pi*alpha**2*hbar**2/m**2/c**2*(omega0/omega)**2*(omega0/omega+omega/omega0-sin(theta)**2)
        f = lambda x: tpi*x[:,[0]]**2*x[:,[1]]**2/x[:,[2]]**2/x[:,[3]]**2*(x[:,[4]]/x[:,[5]])**2*(x[:,[4]]/x[:,[5]]+x[:,[5]]/x[:,[4]]-torch.sin(x[:,[6]])**2)
        ranges = [[0,1],[0,1],[0.5,2],[0.5,2],[0.5,2],[0.5,2],[0,2*tpi]]
        
    return symbol, expr, f, ranges