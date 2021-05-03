import matplotlib.pyplot as plt
import numpy as np

def y(x,xi):
    """
    x = np.arange(-1,2.1,0.1)
    plt.plot(x,np.exp(-x))
    plt.plot(x,y(x,0.5))
    plt.show()
    
    xi = np.arange(-1,2.1,0.1)
    x_0 = 0.5
    plt.plot(x,y(x_0,xi))
    plt.show()    
    """
    return np.exp(-xi)-np.exp(-xi)*(x-xi)

def sigmoid(z):
    """
    x = np.arange(-20,10,0.1)
    plt.plot(x,np.log(sigmoid(x)))
    plt.show()
    """
    return 1/(1+np.exp(-z))

def _lambda(xi):#xi=xis; xi=Phi.dot(w); xi = 0
    """
    Cunado xi == 0 quiero que devuelva 0
    """
    div0 = np.divide(1, (2*xi),where=xi!=0)
    return (div0 * (sigmoid(xi)-(1/2)))

def upper_bound(x,xi):
    """
    x = np.arange(-5,5,0.1)
    plt.plot(x,sigmoid(x))
    plt.plot(x,upper_bound(x,2))
    plt.plot(x,upper_bound(x,-2))
    plt.ylim(0,1)
    plt.show()    
    """
    return np.exp(-np.log(1+np.exp(-xi)) + (np.exp(-xi)/(1+np.exp(-xi)))*(x-xi))

def plot_sigmoid2():
    """
    plot_sigmoid2()
    """
    x = np.arange(-5,5,0.1)
    plt.plot(x,sigmoid(x**2))
    plt.show()

def logistic_lower_bound(z,xi):
    return sigmoid(xi)*np.exp( (z-xi)/2 - _lambda(xi)*(z**2 - xi**2) )

def figure_10_12_b():
    """
    Replica la figura 10.12 right de bishop
    Gaussian lower bound for the logistic
    """
    xs = np.arange(-6,6,0.1)
    plt.plot(xs,sigmoid(xs))
    x=2.5
    plt.scatter(x,sigmoid(x))
    plt.plot(xs,logistic_lower_bound(xs,x))
    plt.show()

def polynomial_basis_function(x, degree=1):
    return x ** degree

def likelihood(ts,w,Phi):
    """
    result == (sigmoid(a)**ts) * ((1-sigmoid(a))**(1-ts) )
    """
    a = Phi.dot(w)
    return np.exp(a*ts)*sigmoid(-a) 

def variational_likelihood(ts,w,Phi):
    aes = Phi.dot(w)
    s = logistic_lower_bound(-aes,aes)
    return np.exp(aes*ts)*s

def posterior(ts,Phi,xis,mu0=None,S0=None,alpha=1e8):
    N, D = Phi.shape
    mu0 = mu0 if not mu0 is None else np.zeros(D).reshape((D,1)) 
    S0 = S0 if not S0 is None else  alpha * np.eye(D)
    S0_inv = np.linalg.inv(S0)
    #SN_inv = S0_inv + 2*sum([_lambda(xis[i])*Phi[i,:].reshape((D,1)).dot(Phi[i,:].reshape((1,D))) for i in range(N)] )
    SN_inv = S0_inv + 2*Phi.T.dot(_lambda(xis)*Phi) # es correcto
    SN = np.linalg.inv(SN_inv)
    #SN.dot(S0_inv.dot(mu0) + sum([(ts[i,:]-1/2)*Phi[i,:] for i in range(N)] ).reshape((D,1)))
    muN = SN.dot(S0_inv.dot(mu0) + (ts - 1/2).T.dot(Phi).T ) # es correcto
    return muN, SN    

def new_xis(muN, SN, Phi):
    N, D = Phi.shape
    return np.array([np.sqrt(float(Phi[i,:].reshape((1,D)).dot((SN+muN.dot(muN.T)).dot(Phi[i,].reshape((D,1)))))) for i in range(N) ]).reshape((N,1))

def EM(ts,Phi,xis,mu0=None,S0=None,alpha=1e8):
    divergence = np.inf
    i = 0
    while ((abs(divergence) > 0.01) and (i < 100)): 
        muN, SN = posterior(ts,Phi,xis,mu0,S0,alpha) 
        xis_new = new_xis(muN, SN, Phi)
        delta = float(max(xis - xis_new))
        xis = xis_new
        print(delta)
        divergence = min(delta,divergence)
        i = i + 1
    return muN, SN, xis


"""
El siguiente codigo debe ser migrado a test.py
"""

edades_grilla = np.arange(0,100,1).reshape((100,1))
Phi_grilla = polynomial_basis_function(edades_grilla,np.array(range(2)))

N=330
slope = 0.1
media = 65
w1 = slope
w0 = -slope*media
w = np.array([w0,w1]).reshape((2,1))

edades =np.random.randint(1,100,N).reshape((N,1))
xis = edades
Phi = polynomial_basis_function(edades,np.array(range(2)))
ts = np.random.binomial(1,sigmoid(Phi.dot(w)))
muN, SN, xis_fin = EM(ts,Phi,xis)

plt.scatter(edades,ts)
plt.plot(edades_grilla,sigmoid(Phi_grilla.dot(w)))
plt.plot(edades_grilla,sigmoid(Phi_grilla.dot(muN)))
plt.show()
