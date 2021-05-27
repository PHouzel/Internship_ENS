from datetime import datetime
import matplotlib.pyplot as plt
from numpy import array, linspace, arange, zeros, matlib, reshape, sqrt, pi, transpose, diag, log, cos, sin, random, mean, arctan2, mod, loadtxt, meshgrid, hstack, exp, arctan, savetxt, vstack, ones
from scipy import interpolate
from scipy.integrate import odeint
from numba import jit, float64, types, int64
import sys
from tqdm import tqdm


@jit(types.Tuple((float64, float64))(float64, float64, float64, float64[:], float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64),nopython=True)
def integrador(xIni, yIni, tIni, noiseX, noiseY, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, h, gx, gy, nSteps):

	for i in range(0, nSteps):

		Se, Si = pulse*exp(-tIni/tau_stim)
		a_1 = gee*xIni + gei*yIni + Se + ae
		a_2 = gie*xIni + gii*yIni + Si + ai
		fx = (1/tau_e)*(-xIni + (1/(1 + exp(-a_1))))
		fy = (1/tau_i)*(-yIni + (1/(1 + exp(-a_2))))
		
		#noiseX = random.normal(0, 1)*sqrt(h); noiseY = random.normal(0, 1)*sqrt(h); 
		xTilda = xIni + fx*h + noiseX[i]*gx
		yTilda = yIni + fy*h + noiseY[i]*gy
		tTilda = tIni + h
		
		Se, Si = pulse*exp(-tTilda/tau_stim)
		a_1 = gee*xTilda + gei*yTilda + Se + ae
		a_2 = gie*xTilda + gii*yTilda + Si + ai
		fxTilda = (1/tau_e)*(-xTilda + (1/(1 + exp(-a_1))))
		fyTilda = (1/tau_i)*(-yTilda + (1/(1 + exp(-a_2))))
		
		xIni = xIni + 0.5*(fx*h + fxTilda*h + 2*noiseX[i]*gx)
		yIni = yIni + 0.5*(fy*h + fyTilda*h + 2*noiseY[i]*gy)
		tIni = tIni + h

	return xIni, yIni

@jit(types.Tuple((float64[:], float64[:]))(int64),nopython=True)
def box_muller_sample(n):
    
    u1 = random.rand(n); u2 = random.rand(n)
    
    r = sqrt(-2*log(u1)); thetaTerm = 2*pi*u2
    x = cos(thetaTerm); y = sin(thetaTerm)
    z1 = r*x; z2 = r*y
    
    return z1, z2

def computePhaseShift(xTrajectory, yTrajectory, nPoints, nSteps, nTimes, f_real, f_imag, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, h, gx, gy):

	thetaPert = zeros(nPoints) 
	for i in tqdm(arange(0, nPoints)):
		print(i)
		meanRe = 0; meanIm = 0; 
		for j in range(0, nTimes):
			noiseX, noiseY = box_muller_sample(nSteps)
			#print(xTrajectory[i])
			xFinal, yFinal = integrador(xTrajectory[i], yTrajectory[i], 0, noiseX*sqrt(h), noiseY*sqrt(h), pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, h, gx, gy, nSteps)
			try:
				reSto = f_real(xFinal, yFinal)[-1]; imSto = f_imag(xFinal, yFinal)[-1]
				meanRe = meanRe + reSto
				meanIm = meanIm + imSto
			except:
				print('Unexpected error:', sys.exc_info()[0]) 
		thetaPert[i] = arctan2(meanIm, meanRe);
	
	return thetaPert

def main():
	
	startTime = datetime.now()
	D = 0.1; 
	tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, gx1, gx2, gy1, gy2  = [3., 8., 6., 10., -10, 12., -10, -2., -3.5, D, 0, 0, D]; 
	landa = -1.8977290789720564; period = 33.126711029860324; totalTime = period
	S0 = 10.; theta = 85*(pi/180)
	pulse = array([S0*cos(theta), S0*sin(theta)])

	# Vale, ahora hay que ver que ampStocastica les corresponde
	str2read = './data/WilsonCowan/realValuesD%s' % D; realData = loadtxt(str2read); M, N = realData.shape
	str2read = './data/WilsonCowan/imagValuesD%s' % D; imagData = loadtxt(str2read); M, N = imagData.shape
	yp = 0.75; ym = -0.25; y = linspace(ym, yp, 100)
	xp = 1.0;  xm = -0.25; x = linspace(xm, xp, 100)
	f_real = interpolate.interp2d(x, y, realData, kind='cubic')
	f_imag = interpolate.interp2d(x, y, imagData, kind='cubic')
	
	nTimes = 150; nSteps = 5000; h = totalTime/nSteps; 
	print('TimeStep %s' % h)
	limCycle = loadtxt('./data/WilsonCowan/limCycleData0.1')
    #X and y of the limit cycle
	xLim = limCycle[:,0]; yLim = limCycle[:,1]; thetaArray = limCycle[:,2];
	nPoints = len(xLim)
	print(nPoints)

	thetaPert = computePhaseShift(xLim, yLim, nPoints, nSteps, nTimes, f_real, f_imag, pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, h, gx1/tau_e, gy2/tau_i)
	thetaFree = computePhaseShift(xLim, yLim, nPoints, nSteps, nTimes, f_real, f_imag, 0*pulse, tau_e, tau_i, tau_stim, gee, gei, gie, gii, ae, ai, h, gx1/tau_e, gy2/tau_i)
		
	saveArray = zeros((nPoints, 3))
	saveArray[:,0] = thetaArray
	saveArray[:,1] = thetaPert
	saveArray[:,2] = thetaFree
	#saveArray[:,3] = directPRC

	#saveStr = './data/prc_D%s' % D
	#savetxt(saveStr, saveArray)

	plt.plot(thetaArray, thetaPert, 'r+', ms='15')
	plt.plot(thetaArray, thetaFree, 'b+', ms='15')
	plt.plot(thetaArray, thetaPert-thetaFree, 'g+', ms='15')
	plt.show()
	

if __name__ == '__main__':
	main()
