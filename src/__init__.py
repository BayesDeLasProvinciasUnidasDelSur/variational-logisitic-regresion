import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def _lambda(xi):#xi=xis; xi=Phi.dot(w)
    """
    Cunado xi == 0 quiero que devuelva 0
    """
    div0 = np.divide(1, (2*xi), out=np.zeros_like(xi), where=xi!=0)
    return (-div0 * (sigmoid(xi)-(1/2)))

def logistic_lower_bound(z,xi):
    """
    Notar que hay un sumando (+) en vez de un restando (-) en el exponente
    En alg\'un logar hay un signo invertido (en bishop o ac\'a)
    """
    return sigmoid(xi)*np.exp( (z-xi)/2 + _lambda(xi)*(z**2 - xi**2) )

def figure_10_12_b():
    """
    Replica la figura 10.12 right de bishop
    Gaussian lower bound for the logistic
    TODO:
        [] Agregar par\'ametro bool para imprimir figura
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

def posterior(ts,Phi,mu0=None,S0=None,alpha=1e8):
    N, D = Phi.shape
    mu0 = mu0 if not mu0 is None else np.zeros(D).reshape((D,1)) 
    S0 = S0 if not S0 is None else  alpha * np.eye(D)   
  
    """
    TODO:
        xis = EM procedure
        - Bayesian parameter estimation via variational methods
        - Bishop 10.6.2
        see:
            https://github.com/zhengqigao/PRML-Solution-Manual/blob/master/Solution%20Manual%20For%20PRML.pdf
    """

    S0_inv = np.linalg.inv(S0)
    SN_inv = S0_inv + 2*Phi.T.dot(_lambda(xis)*Phi)
    SN = np.linalg.inv(SN_inv)
    muN = SN.dot(S0_inv.dot(mu0) + (ts - 1/2).T.dot(Phi).T )
    return muN, SN    




"""
El siguiente codigo debe ser migrado a test.py
"""

"""
El likelihood variacional y el likelihood son exactamente iguales
"""
np.sum(np.log(likelihood(ts,w,Phi)))
np.sum(np.log(variational_likelihood(ts,w,Phi)))



edades_grilla = np.arange(0,100,1).reshape((100,1))
Phi_grilla = polynomial_basis_function(edades_grilla,np.array(range(2)))
slope = 0.1
media = 65
w1 = slope
w0 = -slope*media
w = np.array([w0,w1]).reshape((2,1))
plt.plot(edades_grilla,sigmoid(Phi_grilla.dot(w)))
Phi_xi = np.array([1,85])
plt.plot(edades_grilla,logistic_lower_bound(Phi_grilla.dot(w),Phi_xi.dot(w)))

z = Phi_grilla.dot(w)



plt.plot(Phi_grilla.dot(w))
plt.show()
plt.close()

N=1000
edades =np.random.randint(1,100,N).reshape((N,1))
Phi = polynomial_basis_function(edades,np.array(range(2)))
ts = np.random.binomial(1,sigmoid(Phi.dot(w)))
plt.scatter(edades,ts)


muN, SN = posterior(ts,Phi)
plt.plot(edades_grilla,sigmoid(Phi_grilla.dot(muN)))
plt.show()




