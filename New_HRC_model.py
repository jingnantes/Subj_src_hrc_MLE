#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:09:38 2018

@author: jingli
"""



import numpy as np
#from scipy.optimize import root
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#from scipy.stats.stats import pearsonr
#from scipy.optimize import SR1
from scipy.optimize import Bounds
#from scipy.optimize import BFGS



""" generate simulated data """
global pvs_num
global obs_num
global src_num

#iteration_time = 3 # run iteration_time optimization function

"""Zhi's MLE synthetic data setup
synthetic_result={
            'quality_scores': np.random.uniform(1, 5, 79),
            'observer_bias': np.random.normal(0, 1, 30),
            'observer_inconsistency': np.abs(np.random.uniform(0.0, 0.4, 30)),
            'content_bias': np.random.normal(0, 0.00001, 9),
            'content_ambiguity': np.abs(np.random.uniform(0.4, 0.6, 9)),
        }

"""


""" simulation parameter setup """
src_num = 15
hrc_num = 20
pvs_num = src_num*hrc_num
obs_num = 28

## do not change this part
""" hrc true score"""
xh = np.random.uniform(1,5,hrc_num)
""" observer bias """
bs = np.random.uniform(-2,2,obs_num)
#bs = np.random.normal(0, 1, obs_num)
""" observer inconsistency """
vs = np.random.uniform(0.1,0.5,obs_num)
""" content ambiguity """
ae = np.random.uniform(0.4,0.6,src_num)
""" hrc ambiguity"""
ha = np.random.uniform(0.1,0.5,hrc_num)

""" hrc-src ambicuity covariance"""
#hs = np.random.uniform(0,0,pvs_num)


""" synthesized observed score """
global xes
xes =[[0 for i in range(pvs_num)]for i in range(obs_num)]
xes = np.array(xes)

data = np.zeros((src_num,hrc_num,obs_num))

np.random.seed(0)
pvs = -1
for src in range(0,src_num):
    for hrc in range(0,hrc_num):
        pvs = pvs+1
        for obs in range(0,obs_num):
            mu = xh[hrc]+bs[obs]
            #sigma = np.sqrt(np.power(vs[obs],2)+np.power(ae[src],2)+np.power(ha[hrc],2)-2*hs[pvs])
            sigma = np.sqrt(np.power(vs[obs],2)+np.power(ae[src],2)+np.power(ha[hrc],2))
            
            xes[obs,pvs] = np.random.normal(mu, sigma, 1)
            data[src,hrc,obs] = xes[obs,pvs]
# finish the synthesis process

# calculate src mos
srcmos = [[0 for i in range(hrc_num*obs_num)]for i in range(src_num)]
srcmos = np.array(srcmos)
pvs = -1
for src in range(0,src_num):
    n = -1
    for hrc in range(0,hrc_num):
        pvs = pvs+1
        for obs in range(0,obs_num):
            n=n+1;
            srcmos[src,n]=xes[obs,pvs]
            
# calculate hrc mos
hrcmos = [[0 for i in range(hrc_num*obs_num)]for i in range(hrc_num)]
hrcmos = np.array(hrcmos)

for hrc in range(0,hrc_num):
    n = -1
    for src in range(0,src_num):
        for obs in range(0,obs_num):
            n=n+1;
            hrcmos[hrc,n]=data[src,hrc,obs]            

""" likelihood function and its first-order derivatives"""

def Lfunction(x):
    """ the log likelihood function for MLE subjective model"""
    
    xh = x[range(0,hrc_num)]
    bs = x[range(hrc_num,hrc_num+obs_num)]
    ha = x[range(hrc_num+obs_num,2*hrc_num+obs_num)]
    ae = x[range(2*hrc_num+obs_num,2*hrc_num+obs_num+src_num)]
    vs = x[range(2*hrc_num+obs_num+src_num,2*hrc_num+2*obs_num+src_num)]
    #hs = x[range(2*hrc_num+2*obs_num+src_num,2*hrc_num+2*obs_num+src_num+pvs_num)]
    
    res = 0
    pvs = -1
    
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            pvs = pvs+1
            for obs in range(0,obs_num):
              #  res = res + np.log(vs[obs]**2+ae[src]**2+ha[hrc]**2-2*hs[pvs])+np.power((xes[obs,pvs]-xe[hrc]-bs[obs]),2)/(vs[obs]**2+ae[src]**2+ha[hrc]**2-2*hs[pvs])
                res = res + np.log(vs[obs]**2+ae[src]**2+ha[hrc]**2)+np.power((xes[obs,pvs]-xh[hrc]-bs[obs]),2)/(vs[obs]**2+ae[src]**2+ha[hrc]**2)
    
    return res
        
def L_dev(x):
    ## need to be change accordingly
    xh = x[range(0,hrc_num)]
    bs = x[range(hrc_num,hrc_num+obs_num)]
    ha = x[range(hrc_num+obs_num,2*hrc_num+obs_num)]
    ae = x[range(2*hrc_num+obs_num,2*hrc_num+obs_num+src_num)]
    vs = x[range(2*hrc_num+obs_num+src_num,2*hrc_num+2*obs_num+src_num)]
    
    dxh = np.zeros_like(xh)
    dbs = np.zeros_like(bs)
    dha = np.zeros_like(ha)
    dae = np.zeros_like(ae)
    dvs = np.zeros_like(vs)
    
   
    #xx = np.zeros((src_num,hrc_num))
    #tmp = -1
    #for src in range(0,src_num):
    #    for hrc in range(0,hrc_num):
    #        tmp = tmp+1
    #        xx[src,hrc] = xh[tmp]
            
    
    
    for hrc in range(0,hrc_num):
        init = 0
        for src in range(0,src_num):
            for s in range(0,obs_num):
                init = init -2*(data[src,hrc,s]-xh[hrc]-bs[s])/(vs[s]**2+ae[src]**2+ha[hrc]**2)
        dxh[hrc] = init
           
    for s in range(0,obs_num):
        init = 0
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                init = init -2*(data[src,hrc,s]-xh[hrc]-bs[s])/(vs[s]**2+ae[src]**2+ha[hrc]**2)
        dbs[s] = init  
           
    for hrc in range(0,hrc_num):
        init = 0
        for src in range(0,src_num):
            for s in range(0,obs_num):
                init = init + 2*ha[hrc]/(vs[s]**2+ae[src]**2+ha[hrc]**2) -2*ha[hrc]*np.power((data[src,hrc,s]-xh[hrc]-bs[s])/(vs[s]**2+ae[src]**2+ha[hrc]**2),2)
        dha[hrc] = init
           
    for src in range(0,src_num):
        init = 0
        for hrc in range(0,hrc_num):
            for s in range(0,obs_num):                 
                init = init + 2*ae[src]/(vs[s]**2+ae[src]**2+ha[hrc]**2) -2*ae[src]*np.power((data[src,hrc,s]-xh[hrc]-bs[s])/(vs[s]**2+ae[src]**2+ha[hrc]**2),2)
        dae[src] = init 
           
    for s in range(0,obs_num):
        init = 0
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                init = init + 2*vs[s]/(vs[s]**2+ae[src]**2+ha[hrc]**2) -2*vs[s]*np.power((data[src,hrc,s]-xh[hrc]-bs[s])/(vs[s]**2+ae[src]**2+ha[hrc]**2),2)
        dvs[s] = init  
            
    
            
            
    der = np.r_[dxh,dbs,dha,dae,dvs]
    return der

""" Test the proposed subject model """   
    
""" Initialization value for xe,bs,vs, ae """
xh0 = np.mean(hrcmos,1) # calculate column mean
bs0 = np.zeros(obs_num)
ha0 = np.std(hrcmos,axis=1,ddof=1)
ae0 = np.std(srcmos,axis=1,ddof=1)
vs0 = np.std(xes,axis=1,ddof=1)
#hs0 = np.zeros(pvs_num)



#xinput = np.r_[xe0,bs0,ha0,ae0,vs0,hs0]
xinput = np.r_[xh0,bs0,ha0,ae0,vs0]

""" set lower and upper bounds for each parameter """
tempstd = np.std(hrcmos,axis=1)
hrcusestd = np.max(tempstd)

tempstd = np.std(srcmos,axis=1)
srcusestd = np.max(tempstd)

tempstd = np.std(xes,axis=1)
obsusestd = np.max(tempstd)


xelb = np.zeros(hrc_num)
xeub = 6*np.ones(hrc_num)
bslb = -2*np.ones(obs_num)
bsub = 2*np.ones(obs_num)
halb = 0.0001*np.ones(hrc_num)
haub = hrcusestd*np.ones(hrc_num)
aelb = 0.001*np.ones(src_num)
aeub = srcusestd*np.ones(src_num)
vslb = 0.001*np.ones(obs_num)
vsub = obsusestd*np.ones(obs_num)
#hslb =-0.05*np.ones(pvs_num)
#hsub =-0.01*np.ones(pvs_num)

lb = np.r_[xelb,bslb,halb,aelb,vslb] #,hslb]
ub = np.r_[xeub,bsub,haub,aeub,vsub] #,hsub]

bounds = Bounds(lb, ub)

""" set nonlinear constraints that ha^2+ae^2-2*hs > 0"""


"""using minimize method to find MLE estimates """
## nelder-mead method is worse than BFGS
#res = minimize(Lfunction, xinput, method='nelder-mead',options={'xtol':1e-8, 'disp':True})
#res = minimize(Lfunction, xinput, method='BFGS',options={'disp':True})
#res = minimize(Lfunction, xinput, method='Newton-CG',jac = L_dev,options={'disp':True})

""" with constraint that vs and ae >0 """
""" This method is significantly better than the method without constraints """
#res = minimize(Lfunction, xinput, method='trust-constr',  jac=L_dev, hess = BFGS(), options={'verbose': 1}, bounds=bounds)
res = minimize(Lfunction, xinput, method='SLSQP', jac=L_dev, options={'ftol': 1e-9, 'disp': True}, bounds=bounds)
#res = minimize(Lfunction, xinput, method='SLSQP', jac='2-point', options={'ftol': 1e-9, 'disp': True}, bounds=bounds)

print res
############# end of the optimization process##############

xhe = res.x[range(0,hrc_num)]
bse = res.x[range(hrc_num,hrc_num+obs_num)]
hae = res.x[range(hrc_num+obs_num,2*hrc_num+obs_num)]
aee = res.x[range(2*hrc_num+obs_num,2*hrc_num+obs_num+src_num)]
vse = res.x[range(2*hrc_num+obs_num+src_num,2*hrc_num+2*obs_num+src_num)]
#hse = res.x[range(2*hrc_num+2*obs_num+src_num,2*hrc_num+2*obs_num+src_num+pvs_num)]
    

rmse_observe = np.sqrt(np.mean((xh-xh0)**2)) 
rmse_quality = np.sqrt(np.mean((xh-xhe)**2))
rmse_bias = np.sqrt(np.mean((bs-bse)**2)) 
rmse_inconsistency = np.sqrt(np.mean((vs-vse)**2))
rmse_content = np.sqrt(np.mean((ae-aee)**2)) 
rmse_hrcambiguity = np.sqrt(np.mean((ha-hae)**2)) 
#rmse_srchrc = np.sqrt(np.mean(hs-hse)**2)
print 'RMSE between observed score xes and the gt score xe is',rmse_observe
print 'RMSE between estimated score xee and the gt score xe is',rmse_quality
print 'RMSE of bias', rmse_bias
print 'RMSE of inconsistency', rmse_inconsistency
print 'RMSE of content ambiguity', rmse_content
print 'RMSE of hrc ambiguity', rmse_hrcambiguity
#print 'RMSE of srchrc covariance', rmse_srchrc

plt.scatter(xh,xh0,marker=r'$\clubsuit$')
plt.xlabel("ground truth hrc score, xh")
plt.ylabel("observed hrc quality score")
plt.show()

plt.scatter(xh,xhe,marker=r'$\clubsuit$')
plt.xlabel("ground truth hrc score, xh")
plt.ylabel("estimated xhe")
plt.show()

plt.scatter(ha,hae,marker=r'$\clubsuit$')
plt.xlabel("ground truth hrc ambiguity, ha")
plt.ylabel("estimated hae")
plt.show()

plt.scatter(bs,bse,marker=r'$\clubsuit$')
plt.xlabel("ground truth observer bias, bs")
plt.ylabel("estimated bse")
plt.show()

plt.scatter(vs,vse,marker=r'$\clubsuit$')
plt.xlabel("ground truth observer inconsistency, vs")
plt.ylabel("estimated vse")
plt.show()

plt.scatter(ae,aee,marker=r'$\clubsuit$')
plt.xlabel("ground truth content ambiguity, ae")
plt.ylabel("estimated aee")
plt.show()

#plt.scatter(hs,hse,marker=r'$\clubsuit$')
#plt.xlabel("ground truth srchrc covariance, hs")
#plt.ylabel("estimated hse")
#plt.show()


"""use the output as the input of next round, to check if the performance will be improved"""
#for loopn in range(iteration_time):
#    res = minimize(Lfunction, res.x, method='SLSQP', jac=L_dev, options={'ftol': 1e-9, 'disp': True}, bounds=bounds)
#
#print res
#
#xhe = res.x[range(0,hrc_num)]
#bse = res.x[range(hrc_num,hrc_num+obs_num)]
#hae = res.x[range(hrc_num+obs_num,2*hrc_num+obs_num)]
#aee = res.x[range(2*hrc_num+obs_num,2*hrc_num+obs_num+src_num)]
#vse = res.x[range(2*hrc_num+obs_num+src_num,2*hrc_num+2*obs_num+src_num)]
##hse = res.x[range(2*hrc_num+2*obs_num+src_num,2*hrc_num+2*obs_num+src_num+pvs_num)]
#    
#
#rmse_observe = np.sqrt(np.mean((xh-xh0)**2)) 
#rmse_quality = np.sqrt(np.mean((xh-xhe)**2))
#rmse_bias = np.sqrt(np.mean((bs-bse)**2)) 
#rmse_inconsistency = np.sqrt(np.mean((vs-vse)**2))
#rmse_content = np.sqrt(np.mean((ae-aee)**2)) 
#rmse_hrcambiguity = np.sqrt(np.mean((ha-hae)**2)) 
##rmse_srchrc = np.sqrt(np.mean(hs-hse)**2)
#print 'RMSE between observed score xes and the gt score xe is',rmse_observe
#print 'RMSE between estimated score xee and the gt score xe is',rmse_quality
#print 'RMSE of bias', rmse_bias
#print 'RMSE of inconsistency', rmse_inconsistency
#print 'RMSE of content ambiguity', rmse_content
#print 'RMSE of hrc ambiguity', rmse_hrcambiguity
##print 'RMSE of srchrc covariance', rmse_srchrc
#
#
#plt.scatter(xh,xhe,marker=r'$\clubsuit$')
#plt.xlabel("ground truth hrc score, xh")
#plt.ylabel("estimated xhe")
#plt.show()
#
#plt.scatter(ha,hae,marker=r'$\clubsuit$')
#plt.xlabel("ground truth hrc ambiguity, ha")
#plt.ylabel("estimated hae")
#plt.show()
#
#plt.scatter(bs,bse,marker=r'$\clubsuit$')
#plt.xlabel("ground truth observer bias, bs")
#plt.ylabel("estimated bse")
#plt.show()
#
#plt.scatter(vs,vse,marker=r'$\clubsuit$')
#plt.xlabel("ground truth observer inconsistency, vs")
#plt.ylabel("estimated vse")
#plt.show()
#
#plt.scatter(ae,aee,marker=r'$\clubsuit$')
#plt.xlabel("ground truth content ambiguity, ae")
#plt.ylabel("estimated aee")
#plt.show()
#
##plt.scatter(hs,hse,marker=r'$\clubsuit$')
##plt.xlabel("ground truth srchrc covariance, hs")
##plt.ylabel("estimated hse")
##plt.show()
