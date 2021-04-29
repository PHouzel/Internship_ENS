from numpy import linspace, hstack, array
from datetime import datetime
from numpy import sin, cos, pi, loadtxt, mod, array, zeros
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numba import jit, float64, types, int64


@jit([float64[:](float64[:], float64, float64, float64)], nopython=True) 
def stuartLandau(z, t, a, alpha):
	x, y = z
	fx = alpha*x*(1 - (x**2 + y**2)) - y*(1 + alpha*a*(x**2 + y**2))
	fy = alpha*y*(1 - (x**2 + y**2)) + x*(1 + alpha*a*(x**2 + y**2))
	return array([fx, fy])


def findMaxs(x):
	for i in reversed(range(0,len(x)-1)):
		if x[i] > x[i+1] and x[i] > x[i-1]:  
			return i
		      

def main():
  
	startTime = datetime.now()
	a, alpha = [0., 1.]; period = 2*pi/(1 + alpha*a)
	N = 20; phaseArray = linspace(0, 1, N)
	
	xCycle = cos(2*pi*phaseArray); yCycle = sin(2*pi*phaseArray)
	amp = -2; prc = zeros(N)
	
	for i in range(0, N):
		# We integrate the perturbed trajectory
		ic = array([xCycle[i] + amp,  yCycle[i]])
		timeSteps = 2000.; t = linspace(0, 2*period, timeSteps + 1)
		args=(a, alpha);
		pertOrbits = odeint(stuartLandau, ic, t, args, rtol=1e-8, atol=1e-8)
		
		# We integrate the unperturbed trajectory
		ic = array([xCycle[i], yCycle[i]])
		timeSteps = 2000.; t = linspace(0, 2*period, timeSteps + 1)
		args=(a, alpha);
		unpertOrbits = odeint(stuartLandau, ic, t, args, rtol=1e-8, atol=1e-8)
		
		# We obtain the index of the maximas
		ix_perturbed = findMaxs(pertOrbits[:,0]); iy_perturbed = findMaxs(pertOrbits[:,1])
		ix_free = findMaxs(unpertOrbits[:,0]); iy_free = findMaxs(unpertOrbits[:,1])
		
		# And we compute the phase diferences
		delta_thetaX = (t[ix_perturbed] - t[ix_free])/period
		delta_thetaY = (t[iy_perturbed] - t[iy_free])/period
		prc[i] = delta_thetaX
	
	print('\tiniTime: %s\n\tendTime: %s' % (startTime, datetime.now()))
	plt.plot(phaseArray, prc, 'r.')
	plt.show()
	
	
if __name__ == '__main__':
	main()
