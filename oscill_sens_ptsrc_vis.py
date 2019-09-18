
# This Python script was translated from Matlab codes developed and provided as
# part of the following publication: Cardiff, M. and Barrash, W. (2015). 
# Analytical and semi-analytical tools for the design of oscillatory pumping 
# tests. Groundwater, 53(6), 896-907. The original Matlab codes are copyright
# of Michael Cardiff (University of Wisconsin-Madison), 2013-2014.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # not called directly but required


# Black and Kipp (1981) solution:
def phasor_ptsrc_soln(Qpeak, om, K, Ss, xw, yw, zw, x, y, z):
    return Qpeak/(4.*np.pi*K*((x-xw)**2.+(y-yw)**2.+(z-zw)**2.)**0.5)*\
           np.exp(-(1.+1j)/(2.**0.5)*((om*Ss*((x-xw)**2.+(y-yw)**2.+\
           (z-zw)**2.))/K)**0.5)


def oscill_sens_generator_ptsrc(Qpeak, period, Ss, K, domain_def, pump_loc, 
                                obs_loc):
    # Setup discretisation and plotting parameters
    om = 2.*np.pi/period
    x_w   = pump_loc[0]
    y_w   = pump_loc[1]
    z_w   = pump_loc[2]
    x_o   = obs_loc[0]
    y_o   = obs_loc[1]
    z_o   = obs_loc[2]
    xmin  = domain_def[0, 0]
    xstep = domain_def[0, 1]
    xmax  = domain_def[0, 2]
    ymin  = domain_def[1, 0]
    ystep = domain_def[1, 1]
    ymax  = domain_def[1, 2]
    zmin  = domain_def[2, 0]
    zstep = domain_def[2, 1]
    zmax  = domain_def[2, 2]
    xv    = np.arange(xmin, xmax, xstep)
    yv    = np.arange(ymin, ymax, ystep)
    zv    = np.arange(zmin, zmax, zstep)
    xg, yg, zg = np.meshgrid(xv, yv, zv)
    
    # Calculate the value of the phasor at the observation location
    # for use in the source term for the phasor solution
    phasor_at_obs = phasor_ptsrc_soln(Qpeak, om, Ss, K, x_w, y_w, z_w, x_o, 
                                      y_o, z_o)
    obs_adjsrc_val = -1./phasor_at_obs
    
    # Calculate the phasor solution
    phasor_soln = phasor_ptsrc_soln(Qpeak, om, Ss, K, x_w, y_w, z_w, xg, yg, 
                                    zg)
    
    # Calculate the adjoint solution
    adj_soln = phasor_ptsrc_soln(obs_adjsrc_val, om, Ss, K, x_o, y_o, z_o, xg, 
                                 yg, zg)
    
    # Calculate sensitivity of metrics to ln(Ss). Note this is why the extra
    # factor of Ss appears at the end of the sensitivity calculation
    logSs_sens = phasor_soln*adj_soln*1j*om*Ss
    logmag_logSs_sens = np.real(logSs_sens)
    phase_logSs_sens  = np.imag(logSs_sens)
    
    # First calculate the spatial gradients of the two solutions (needed for
    # calculating K sensitivities). Then do sensitivity calculation. 
    # Note, extra factor of K comes from calculating sensitivity with respect 
    # to ln(K) instead of to K.
    phasorgradx, phasorgrady, phasorgradz = np.gradient(phasor_soln, xstep, 
                                                        ystep, zstep)
    adjgradx, adjgrady, adjgradz = np.gradient(adj_soln, xstep, ystep, zstep)
    logK_sens = K*(phasorgradx*adjgradx+phasorgrady*adjgrady)
    logmag_logK_sens = np.real(logK_sens)
    phase_logK_sens = np.imag(logK_sens)
    return (xg, yg, zg, logmag_logK_sens, logmag_logSs_sens, phase_logK_sens,  
            phase_logSs_sens)


Qpeak = 1e-3
period_list = [0.1, 1., 10., 100.]
Ss = 1e-4
K  = 1e-4

# Domain definition - to avoid issues with point sources, use spacings that
# will not exactly land on the locations of wells
domain_def = np.array([[-5., 0.22, 5.],
                       [-5., 0.22, 5.],
                       [-5., 0.22, 5.]])
pump_loc = [-2.5, 0.0, 0.0]
obs_loc  = [ 2.5, 0.0, 0.0]
lims = [domain_def[0,0], domain_def[0,2], 
        domain_def[1,0], domain_def[1,2],
        domain_def[2,0], domain_def[2,2]]

num_periods = len(period_list)


f = plt.figure(figsize=[16.00/2.54, 32.00/2.54])
f.suptitle('ln(amplitude) sensitivity maps')
s = np.reshape([f.add_subplot(4, 2, sp, projection='3d') for sp in range(1,9)], 
               [4,2])

for p in range(num_periods): 
    (xg, yg, zg, logmag_logK_sens, logmag_logSs_sens, phase_logK_sens, 
     phase_logSs_sens) = oscill_sens_generator_ptsrc(Qpeak, period_list[p], 
     Ss, K, domain_def, pump_loc, obs_loc)

    start_idx = int(np.shape(yg)[1]/2)
    thresh = np.abs(logmag_logK_sens)[:,start_idx:,:]>1e-3
    s[p,0].scatter3D(yg[:,start_idx:,:][thresh], 
                     xg[:,start_idx:,:][thresh], 
                     zg[:,start_idx:,:][thresh], 
                     c=logmag_logK_sens[:,start_idx:,:][thresh], 
                     s=1., cmap='bwr', vmin=-0.1, vmax=0.1, alpha=0.5)  
    
    s[p,0].set_xlim(domain_def[0,0], domain_def[0,2])
    s[p,0].set_ylim(domain_def[1,0], domain_def[1,2])
    s[p,0].set_zlim(domain_def[2,0], domain_def[2,2])
    s[p,0].set_xlabel('x (m)')
    s[p,0].set_ylabel('y (m)')
    s[p,0].set_zlabel('z (m)')
    
    
    start_idx = int(np.shape(yg)[1]/2)
    thresh = np.abs(logmag_logSs_sens)[:,start_idx:,:]>1e-3
    s[p,1].scatter3D(yg[:,start_idx:,:][thresh], 
                     xg[:,start_idx:,:][thresh], 
                     zg[:,start_idx:,:][thresh], 
                     c=logmag_logSs_sens[:,start_idx:,:][thresh], 
                     s=1., cmap='bwr', vmin=-0.1, vmax=0.1, alpha=0.5)  

    s[p,1].set_xlim(domain_def[0,0], domain_def[0,2])
    s[p,1].set_ylim(domain_def[1,0], domain_def[1,2])
    s[p,1].set_zlim(domain_def[2,0], domain_def[2,2])
    s[p,1].set_xlabel('x (m)')
    s[p,1].set_ylabel('y (m)')
    s[p,1].set_zlabel('z (m)')

    if p==0:
        s[p,0].set_title('to ln(T)\n')
        s[p,1].set_title('to ln(S)\n')
    plt.gcf().text(0.025, 1.07-float(p+1)*0.21, 
            'Period = '+str(period_list[p])+' s', rotation=90)

mpl.colorbar.ColorbarBase(ax=f.add_axes([0.08, 0.04, 0.86, 0.018]),
                          cmap=plt.get_cmap('bwr'),
                          norm=mpl.colors.Normalize(vmin=-0.1, vmax=0.1),
                          ticks=np.arange(-0.1, 0.15, 0.05), 
                          orientation='horizontal',
                          label='Parameter sensitivity')

plt.tight_layout()
f.subplots_adjust(right=0.93, top=0.93, bottom=0.1, wspace=0.1)
plt.savefig('oscill_sens_ptsrc_vis_Amplitude.png', dpi=500)
plt.close(f)


f = plt.figure(figsize=[16.00/2.54, 32.00/2.54])
f.suptitle('ln(phase) sensitivity maps')
s = np.reshape([f.add_subplot(4, 2, sp, projection='3d') for sp in range(1,9)], 
               [4,2])

for p in range(num_periods): 
    (xg, yg, zg, logmag_logK_sens, logmag_logSs_sens, phase_logK_sens, 
     phase_logSs_sens) = oscill_sens_generator_ptsrc(Qpeak, period_list[p], 
     Ss, K, domain_def, pump_loc, obs_loc)

    start_idx = int(np.shape(yg)[1]/2)
    thresh = np.abs(phase_logK_sens)[:,start_idx:,:]>1e-3
    s[p,0].scatter3D(yg[:,start_idx:,:][thresh], 
                     xg[:,start_idx:,:][thresh], 
                     zg[:,start_idx:,:][thresh], 
                     c=phase_logK_sens[:,start_idx:,:][thresh], 
                     s=1., cmap='bwr', vmin=-0.1, vmax=0.1)  
    
    s[p,0].set_xlim(domain_def[0,0], domain_def[0,2])
    s[p,0].set_ylim(domain_def[1,0], domain_def[1,2])
    s[p,0].set_zlim(domain_def[2,0], domain_def[2,2])
    s[p,0].set_xlabel('x (m)')
    s[p,0].set_ylabel('y (m)')
    s[p,0].set_zlabel('z (m)')
    
    
    start_idx = int(np.shape(yg)[1]/2)
    thresh = np.abs(phase_logSs_sens)[:,start_idx:,:]>1e-3
    s[p,1].scatter3D(yg[:,start_idx:,:][thresh], 
                     xg[:,start_idx:,:][thresh], 
                     zg[:,start_idx:,:][thresh], 
                     c=phase_logSs_sens[:,start_idx:,:][thresh], 
                     s=1., cmap='bwr', vmin=-0.1, vmax=0.1)  

    s[p,1].set_xlim(domain_def[0,0], domain_def[0,2])
    s[p,1].set_ylim(domain_def[1,0], domain_def[1,2])
    s[p,1].set_zlim(domain_def[2,0], domain_def[2,2])
    s[p,1].set_xlabel('x (m)')
    s[p,1].set_ylabel('y (m)')
    s[p,1].set_zlabel('z (m)')

    if p==0:
        s[p,0].set_title('to ln(T)\n')
        s[p,1].set_title('to ln(S)\n')
    plt.gcf().text(0.025, 1.08-float(p+1)*0.23, 
            'Period = '+str(period_list[p])+' s', rotation=90)

mpl.colorbar.ColorbarBase(ax=f.add_axes([0.08, 0.04, 0.86, 0.018]),
                          cmap='bwr',
                          norm=mpl.colors.Normalize(vmin=-0.1, vmax=0.1),
                          ticks=np.arange(-0.1, 0.15, 0.05), 
                          orientation='horizontal',
                          label='Parameter sensitivity')

plt.tight_layout()
f.subplots_adjust(right=0.93, top=0.93, bottom=0.1, wspace=0.1)
plt.savefig('oscill_sens_ptsrc_vis_Phase.png', dpi=500)
plt.close(f)
