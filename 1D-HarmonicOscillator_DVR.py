#parameter
n0: int = 10        #matrix size
m: float = 10       #mass/a.m.u.
kzz: float = 500    #Morse parameter/cm-1
az: float = 1.5     #Morse parameter/Å^-1

""""------------------------------------------------------------------
以下計算プログラム
------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt

#constant
c: float = 2.998*10**(8)
hbar: float = 1.054571817*10**(-34)
N_A: float = 6.02214076*10**(23)
m_kg: float = m/N_A/10**3
omega: float = np.sqrt(2*kzz*az*az *1.98645*10**(-23) *10**(20)/m_kg)  #J, m　単位
hbaromega: float = omega/(2*np.pi)/c/100

def DVR():
    x: np.ndarray = np.zeros((n0 + 1, n0 + 1))
    Ri: np.ndarray = np.zeros((n0 + 1))
    Ri_vec: np.ndarray = np.zeros((n0 + 1, n0 + 1))
    for i in range (1, n0 + 1):
        x[i, i - 1] = np.sqrt(i)    #creation operator
        x[i - 1, i] = np.sqrt(i)    #annihilation operator
    x = np.sqrt(hbaromega/(4*kzz*az*az))*x
    Ri, Ri_vec = np.linalg.eigh(x)
    Ri_vec = Ri_vec.real
    return Ri, Ri_vec

#対角化を何度もしなくないためgridとユニタリー行列はグローバル変数で定義しておく．
Ri = DVR()[0]
Ri_vec = DVR()[1]

class MatrixElement:
    def __init__(self, n0row, n0column):
        self.n0row: int = n0row
        self.n0column: int = n0column
    
    def Ts_DVRbasis_1(self):
        Ts_1: float = 0
        if self.n0row == self.n0column:
            Ts_1 = Ri[self.n0column]**2
        else:
            Ts_1 = 0
        
        return Ts_1

    def Ts_DVRbasis_2(self):
        Ts_2: float = 0
        Ts_3: float = 0
        for nn in range(0, n0 + 1):
            Ts_2 += nn*Ri_vec.transpose()[self.n0row, nn]*Ri_vec[nn, self.n0column]
        
        if self.n0row == self.n0column:
            Ts_3 = 1
        else:
            Ts_3 = 0
        return 4*Ts_2 + 2*Ts_3

    def Vs_DVRbasis(self):
        Vs: float = 0
        if self.n0row == self.n0column:
            Vs = kzz*(1 - np.exp(-az*Ri[self.n0column]))**2
        return Vs
    
    def Ts_HObasis(self):
        Ts: float = 0
        if self.n0row == self.n0column:
            Ts = 2*self.n0column + 1
        elif self.n0row == self.n0column + 2:
            Ts = - np.sqrt((self.n0column + 1)*(self.n0column + 2))
        elif self.n0row == self.n0column - 2:
            Ts = - np.sqrt(self.n0column*(self.n0column - 1))
        else:
            Ts = 0
        return Ts

    def Vs_HObasis(self):
        #unitary matrix product
        Vs: float = 0
        for i in range(0, n0 + 1):
            Vs += (kzz*(1 - np.exp(-az*Ri[i]))**2)*Ri_vec[self.n0row, i]*Ri_vec.transpose()[i, self.n0column]
        return Vs
    
#DVR basisにおけるHamiltonian行列
def H_DVR():
    #print("coefficient, ", round(- m_kg*omega*omega/2 *10**(-20) *5.03412*10**(22), 5), round(hbaromega/4, 5))
    result: np.ndarray = np.array([[+ MatrixElement(n0row, n0column).Ts_DVRbasis_1() *(- m_kg*omega*omega/2 *10**(-20) *5.03412*10**(22))
                                    + MatrixElement(n0row, n0column).Ts_DVRbasis_2() *hbaromega/4
                                    + MatrixElement(n0row, n0column).Vs_DVRbasis() 
                        for n0column in range(0, n0 + 1)
                    ]
                    for n0row in range(0, n0 + 1)
                ])
    result = result.astype(np.float64) 
    return result

#harmonic oscillator basisにおけるHamiltonian行列
def H_HO():
    result: np.ndarray = np.array([[+ MatrixElement(n0row, n0column).Ts_HObasis() *hbaromega/4
                                  + MatrixElement(n0row, n0column).Vs_HObasis() 
                        for n0column in range(0, n0 + 1)
                    ]
                    for n0row in range(0, n0 + 1)
                ])
    result = result.astype(np.float64)
    return result

def diagonalization(Hamiltonian):
    size: int = len(Hamiltonian)
    eig_val: np.ndarray = np.zeros((size))
    eig_vec: np.ndarray = np.zeros((size, size))
    #diagonalization
    eig_val, eig_vec = np.linalg.eigh(Hamiltonian) 
    return [eig_val, eig_vec]

def Morse_anal():
    #analytical solution
    analE: list = []
    for n in range(0, n0):
        E: float = hbaromega*(n + 1/2) - (hbaromega*(n + 1/2))**2/(4*kzz)
        analE.append(E)
    print('analytical eigen value\n', analE - analE[0])

def make_graph():
    #visualization
    plt.figure(dpi = 1000, figsize=(4, 3))
    plt.rcParams["font.size"] = 10
    p: list = np.linspace( -0.7, 0.7, 70)
    q: list = kzz*(1 - np.exp(-az*p))**2
    p1 = plt.plot(p, q)
    for i in range(0, n0 + 1):    
        p2 = plt.plot(Ri[i], kzz*(1 - np.exp(-az*Ri[0][i]))**2, marker='o',c='orange')
    plt.legend((p1[0],   p2[0]), ("potential surface", "DVR grid"))
    plt.xlabel("Distance/$\mathrm{\AA}$")
    plt.ylabel("Energy/cm$^{-1}$")
    plt.show() 

def main ():
    #print DVR grid
    print('Ri eigen value (Gauss-Hermite grid)')
    for ii in range(0, n0 + 1):
        print(round(Ri[ii], 3), end='\t')
    print('\n')
    
    #diagonalization
    val_DVR: np.ndarray = np.zeros((n0 + 1))
    vec_DVR: np.ndarray = np.zeros((n0 + 1, n0 + 1))
    val_HO: np.ndarray = np.zeros((n0 + 1))
    vec_HO: np.ndarray = np.zeros((n0 + 1, n0 + 1))
    
    val_DVR, vec_DVR = diagonalization(H_DVR())
    val_HO, vec_HO = diagonalization(H_HO())
    print('DVR basis eigen value\n{}\n'.format(val_DVR - val_DVR[0]))
    print('HO basis eigen value\n{}\n'.format(val_HO - val_HO[0]))
    Morse_anal()
    make_graph()
    return

main()