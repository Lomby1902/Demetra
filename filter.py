import numpy as np
from math import sqrt, log
import matplotlib.pyplot as plt

class Gaussian_filter:
    def __init__(self,lambda_min,lambda_max,constrain_min, constrain_max):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.constrain_min = constrain_min
        self.constrain_max = constrain_max

    def _gaussian(self,r,mu,sigma):
        return [(1/(sigma*np.sqrt(2*np.pi)))*np.exp((-1/2)*((x-mu)/sigma)**2) for x in r] 
    
    def _read_signals(self,x,n_signals):
        signals = x[:n_signals,:]
        return signals
    
    def _create_filters(self):
        if self._plot:     
            self._fig.suptitle(f"{self.n_bands} band, fwhm = {self.fwhm}")
            ax_left = self._fig.add_subplot(1, 2, 1)
        self._filters = []
        self.sigma = self.fwhm/(2*sqrt(2*log(2)))
        self._mu_list= np.linspace(self.constrain_min,self.constrain_max,self.n_bands)
        '''Define Filters'''
        for mu in self._mu_list:
            x = np.arange(self.lambda_min, self.lambda_max + 1, dtype=int) #1nm delta
            y = self._gaussian(x,mu,self.sigma)
            self._filters.append(y)
            if self._plot:
                ax_left.set_xlim(self.lambda_min, self.constrain_max + 50)
                ax_left.plot(x,y)
    
    def apply_filters(self,x,n_signals,n_bands,fwhm, plot):
        self.n_bands = n_bands
        self.fwhm = fwhm
        self._plot = plot
        if self._plot:
            self._fig = plt.figure(figsize=(10, 5))
        self._create_filters()
        filtered_signals = []
        if self._plot:
            ax_right = self._fig.add_subplot(1, 2, 2)

        '''Apply filters'''
        signals = self._read_signals(x,n_signals)
        for signal in signals:
            filtered_signal = []
            for filter in self._filters: 
                integrand = np.multiply(filter,signal)
                point = sum(integrand) #1nm delta lambda
                filtered_signal.append(point)
            filtered_signals.append(filtered_signal)

        if self._plot:
            colors = ["r","g","b","k"]

            x = np.arange(self.lambda_min, self.lambda_max + 1, dtype=int) #1nm delta

            i = 0
            for signal, filtered_signal in zip(signals,filtered_signals):
                ax_right.plot(x, signal,"--",color=colors[i], label=f"Signal {i+1}")
                ax_right.plot(self._mu_list,filtered_signal,color=colors[i],label=f"Filtered Signal {i+1}")
                i+=1

            ax_right.legend()
            ax_right.set_xlabel("λ")
            ax_right.set_ylabel("Integrated R(λ)")
            ax_right.set_xlim(self.constrain_min-10,self.constrain_max+10)
            plt.tight_layout()
            plt.show()
            
        print(f"Filtered signals(n_bands={self.n_bands}, fwhm={self.fwhm}):",filtered_signals)
        return filtered_signals
    
        