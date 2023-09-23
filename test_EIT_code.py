# %%
import os
import gc
from turtle import width
import h5py
os.environ['NUMBA_ENABLE_AVX'] = '1'
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '0'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

import numpy as np
from numba import njit, set_num_threads
from numpy import zeros, arange, ones, pi, meshgrid, sqrt, eye, exp, conj
from numpy.linalg import inv
import matplotlib.pylab as plt
from datetime import datetime
date = datetime.strftime(datetime.now(),'%Y%m%d')
import imageio
from io import BytesIO
from scipy import interpolate

# %%
@njit(nogil=True, fastmath=True, cache=False)
def simulate():
    set_num_threads(2)
    k_Cs = 2*pi/894.59e-6 # 1/mm
    r31 = 2*pi*5.2*1e6 # 1/s
    r21 = 0
    Ac = 60*1e6
    Ap = 1
    tend = 0.0005
    #xend_r=2
    xend = 2  ###
    yend = 2  ####
    xslp = xend #xslp>=0.5*xend
    OD_x = 400 #1000#400*100  #5
    OD_y = OD_x*yend/xend
    #vg_x=Ac**2*xend/r31/OD_x
    #vg_y=Ac**2*yend/r31/OD_y
    dt = 0.3e-7/100 #0.3e-7/100 
    dx = xend/40
    dy = yend/40
    
    ### parameters setting
    x=arange(-xend/2,xend/2+dx/2,dx)
    y=arange(-yend/2,yend/2+dy/2,dy)
    t=arange(0,tend+dt/2,dt)
    dense=ones((len(y),len(x)))
    
    #beta=sqrt(1+((8*OD_y*r31**2)/(tau_t**2*Ac**4)))
    #vg=Ac**2*yend/r31/OD_y
    eta_x=OD_x*r31/2/xend
    eta_y=OD_y*r31/2/yend

    ### switch function
    #input_time=1e-6 # input wp (middle of switching time) 
    cut1=0.00002 # storage
    cut2 = cut1 + 0.0002 # release
    pulse_peak = 0.00005
    pulse_wid = 0.0001/8
    guassian = np.exp(-(t - pulse_peak)**2/2/pulse_wid**2)
    # cut3=cut2+0.25e-3 # Turn on Landau Gauge potential
    #cut4=cut3+5e-5 # Turn on ePA/m perturbation
    # cut1=-1#1e-4*2.5 # storage
    # cut2=-1 # release
    # cut3=-1 # Turn on Landau Gauge potential
    # cut4=-1 # Turn on ePA/m perturbation

    switch_Y_value=1/100 # definition of start & -1 of y value
    switch_X_value=np.arctanh(1-2*switch_Y_value)*2 # To find x value in normalized tanh function
    #switch_timei=5e-6/dt # total switch time
    switch_time1=5e-6/dt
    #switch_time2=2.5e-5/dt
    #switch_time3=2.5e-5/dt
    #switch_time4=2.5e-5/dt
    #switch_len=xend/20
    
    ratc1=1-(1-np.tanh((t-cut2)/dt*switch_X_value/switch_time1))/2
    ratc1c=(1-np.tanh((t-cut1)/dt*switch_X_value/switch_time1))/2
    # ratc2=(1-np.tanh((t-cut2)/dt*switch_X_value/switch_time1))/2
    # ratc3=1-(1-np.tanh((t-cut2)/dt*switch_X_value/switch_time1))/2

    video_interval=round(tend/dt/500)#200*5
    video_total=(len(t)-1)//video_interval

    ###l_B
    l_B=0.5#0.55
    
    ###detunning
    X = np.repeat(x,len(y)).reshape(len(x),len(y)).T # [X,Y]=meshgrid(x,y)
    
    #detu_c_coff=Ac**2/(16*detu_p_coff)
    detu_p_coff=l_B**2*eta_y/4/xslp
    ### ky & mode
    #n_ky=0
    #ky=n_ky*2*pi/yend
    #mode=0
    
    ### e/m*P*A
    wB=Ac**2/2/xslp/eta_x
    
    # dense_t = np.linspace(-np.pi,np.pi,20)
    # dense_t = (1/2)*np.tanh(dense_t)+1/2
    
    #### the range of the obstacle 
    # dense[0:8,0:10]=0

    ### Crank-Nicolson Parameters
    ### x direction

    idadx=OD_x*r31/xend/2*dx
    alphax=1j/2/k_Cs*dx/dy**2

    #### input conditions
    wk = wB  #####
    kk = (xend/2/l_B**2) #*kk_Arr(idx_kk)
    
    #clockwise use -1j*kk*y
    # 如果要做打一邊的wr改成後面註解的波形即可
    input_wr1 = np.outer((exp(-((y-0)/0.4)**2)*exp(-1j*kk*y)), exp(-1j*wk*t))#np.outer( exp(-1j*kk*y).T, exp(-1j*wk*t) )#np.outer( exp(-1j*kk*y).T, exp(-1j*wk*t) )#
    # input_wr2 = np.outer( (exp(-((y-0)/0.4)**2)*exp( 1j*kk*y)),exp(-1j*wk*t))
    # input_wl = np.outer( exp( 1j*kk*y).T, exp(-1j*wk*t) )
    
    I=eye(len(y))
    Ax_matrix=I*2*(1-dy**2/dx**2*alphax+alphax)
    Bx_matrix=I*2*(1-2*dy**2/dx**2*alphax-alphax)

    for i in range(1,len(y)):
        Ax_matrix[i-1,i]=-alphax
        Bx_matrix[i-1,i]=alphax
        Ax_matrix[i,i-1]=-alphax
        Bx_matrix[i,i-1]=alphax

    Axinv_matrix=inv(Ax_matrix)

    ### y direction
    # idady=OD_y*r31/yend/2*dy
    # alphay=1j/2/k_Cs*dy/dx**2
    # I=eye(len(x))
    # Ay_matrix=I*2*(1-dx**2/dy**2*alphay+alphay)
    # By_matrix=I*2*(1-2*dx**2/dy**2*alphay-alphay)
    # for i in range(1,len(x)):################
    #     Ay_matrix[i-1,i]=-alphay
    #     By_matrix[i-1,i]=alphay
    #     Ay_matrix[i,i-1]=-alphay
    #     By_matrix[i,i-1]=alphay

    # Ayinv_matrix=inv(Ay_matrix)

    wp_sum_ini=np.abs(Ap)**2*xend*yend
    
    #### pre-allocation
    rk4_cache = np.complex128(0)
    k1 = zeros((len(y),len(x)), dtype = np.cdouble)
    k2 = zeros((len(y),len(x)), dtype = np.cdouble)
    k3 = zeros((len(y),len(x)), dtype = np.cdouble)
    k4 = zeros((len(y),len(x)), dtype = np.cdouble)

    p21=zeros((len(y),len(x)), dtype = np.cdouble)

    p31_r=zeros((len(y),len(x)), dtype = np.cdouble)
    p31_l=zeros((len(y),len(x)), dtype = np.cdouble)
    p31_f=zeros((len(y),len(x)), dtype = np.cdouble)
    p31_b=zeros((len(y),len(x)), dtype = np.cdouble)

    wp_r = np.asfortranarray( zeros((len(y),len(x)), dtype = np.cdouble) )
    wp_l = np.asfortranarray( zeros((len(y),len(x)), dtype = np.cdouble) )
    wp_f = zeros((len(y),len(x)), dtype = np.cdouble)
    wp_b = zeros((len(y),len(x)), dtype = np.cdouble)

    wc_r = zeros((len(y),len(x)), dtype = np.cdouble)
    wc_l = zeros((len(y),len(x)), dtype = np.cdouble)
    wc_f = zeros((len(y),len(x)), dtype = np.cdouble)
    wc_b = zeros((len(y),len(x)), dtype = np.cdouble)

    ER=zeros(len(t))
    p21_record=zeros((len(y),len(x),video_total), dtype = np.cdouble)

    j=0

    ### detuning
    detu_p = detu_p_coff
    detu_c = detu_p - Ac**2/(16*detu_p_coff)*((X**2)/xslp**2)

    ### Couple

    r1 = []
    for n in range(len(t)):
        # wc_f = Ac/sqrt(2)*sqrt(1+X/xslp)*(ratc1[n]*ratc2[n]) + Ac/sqrt(2)*sqrt(1-X/xslp)*ratc3[n]#1-Xforclockwise Ac/sqrt(2)*sqrt(1-X/xslp)*ratc1[n]#
        # wc_b = Ac/sqrt(2)*sqrt(1-X/xslp)*(ratc1[n]*ratc2[n]) + Ac/sqrt(2)*sqrt(1+X/xslp)*ratc3[n]#Ac/sqrt(2)*sqrt(1+X/xslp)*ratc1[n]#
        wc_r = Ac*(ratc1c[n] + ratc1[n])  #+Ac/sqrt(2)*ratc1[n]
        # wc_l = Ac/sqrt(2)*ratc1[n]

        ### input probe
        wp_r[:,0] = Ap*(input_wr1[:,n]*guassian[n]) #ratc2[n]+input_wr1[:,n]*ratc3[n])#Ap*input_wr1[:,n]#
        # wp_l[:,-1] = Ap*input_wl[:,n]
        
        ### wp by Crank Nicoson
        # wp_l[:,-1] = Axinv_matrix@(1j*idadx*dense[:,-1]*p31_l[:,-1])
        # wp_b[-1,:] = Ayinv_matrix@(1j*idady*dense_t[-1,:].T*p31_b[-1,:].T)
        # wp_f[0,:] = Ayinv_matrix@(1j*idady*dense_t[0,:].T*p31_f[0,:].T)
        
        wp_r[:,1] =Axinv_matrix@(Bx_matrix@wp_r[:,0]+1j*idadx*dense[:,0]*(p31_r[:,1]+p31_r[:,0]))
        wp_l[:,-2]=Axinv_matrix@(Bx_matrix@wp_l[:,-1]+1j*idadx*dense[:,-1]*(p31_l[:,-2]+p31_l[:,-1]))
        # wp_f[1,:]  =Ayinv_matrix@(By_matrix@wp_f[0,:].T+1j*idady*dense_t[0,:].T*(p31_f[1,:].T+p31_f[0,:].T))
        # wp_b[-2,:] =Ayinv_matrix@(By_matrix@wp_b[-1,:].T+1j*idady*dense_t[-1,:].T*(p31_b[-2,:].T+p31_b[-1,:].T))
        
        for m in range(1,len(x)-1):
            wp_r[:,m+1] =Axinv_matrix@(Bx_matrix@wp_r[:,m]+2*dy**2/dx**2*alphax*wp_r[:,m-1]+1j*idadx*dense[:,m]*(p31_r[:,m+1]+p31_r[:,m]))
            wp_l[:,-m-2]=Axinv_matrix@(Bx_matrix@wp_l[:,-m-1]+2*dy**2/dx**2*alphax*wp_l[:,-m]+1j*idadx*dense[:,-m-1]*(p31_l[:,-m-2]+p31_l[:,-m-1]))
        
        # for m in range(1,len(y)-1):
        #     wp_f[m+1,:]=Ayinv_matrix@(By_matrix@wp_f[m,:].T+2*dx**2/dy**2*alphay*wp_f[m-1,:].T+1j*idady*dense_t[m,:].T*(p31_f[m+1,:].T+p31_f[m,:].T))
        #     wp_b[-m-2,:]=Ayinv_matrix@(By_matrix@wp_b[-m-1,:].T+2*dx**2/dy**2*alphay*wp_b[-m,:].T+1j*idady*dense_t[-m-1,:].T*(p31_b[-m-2,:].T+p31_b[-m-1,:].T))
        
        ### p21, p31 by Range Kutta 4th
        ## p21
        
        rk4_cache = -1j*(detu_p-detu_c)*p21 + 1j/2*(conj(wc_r)*p31_r + conj(wc_l)*p31_l + conj(wc_f)*p31_f + conj(wc_b)*p31_b)
        k1 = rk4_cache - r21* p21
        k2 = rk4_cache - r21*(p21 + dt/2 * k1)
        k3 = rk4_cache - r21*(p21 + dt/2 * k2)
        k4 = rk4_cache - r21*(p21 + dt   * k3)
        p21 = p21 + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        ## p31
        rk4_cache = 1j/2*wp_r + 1j/2*wc_r*p21 - r31/2*p31_r
        k1 = -1j*detu_p*(p31_r            ) + rk4_cache
        k2 = -1j*detu_p*(p31_r + dt/2 * k1) + rk4_cache
        k3 = -1j*detu_p*(p31_r + dt/2 * k2) + rk4_cache
        k4 = -1j*detu_p*(p31_r + dt   * k3) + rk4_cache
        p31_r = p31_r + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        rk4_cache = 1j/2*wp_l + 1j/2*wc_l*p21 - r31/2*p31_l
        k1 = -1j*detu_p*(p31_l            ) + rk4_cache
        k2 = -1j*detu_p*(p31_l + dt/2 * k1) + rk4_cache
        k3 = -1j*detu_p*(p31_l + dt/2 * k2) + rk4_cache
        k4 = -1j*detu_p*(p31_l + dt   * k3) + rk4_cache
        p31_l = p31_l + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        rk4_cache = 1j/2*wp_f + 1j/2*wc_f*p21 - r31/2*p31_f
        k1 = -1j*detu_p*(p31_f            ) + rk4_cache
        k2 = -1j*detu_p*(p31_f + dt/2 * k1) + rk4_cache
        k3 = -1j*detu_p*(p31_f + dt/2 * k2) + rk4_cache
        k4 = -1j*detu_p*(p31_f + dt   * k3) + rk4_cache
        p31_f = p31_f + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        rk4_cache = 1j/2*wp_b + 1j/2*wc_b*p21 - r31/2*p31_b
        k1 = -1j*detu_p*(p31_b            ) + rk4_cache
        k2 = -1j*detu_p*(p31_b + dt/2 * k1) + rk4_cache
        k3 = -1j*detu_p*(p31_b + dt/2 * k2) + rk4_cache
        k4 = -1j*detu_p*(p31_b + dt   * k3) + rk4_cache
        p31_b = p31_b + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        ### Energy ratio
        wp_Abs2 = np.abs(wp_r)**2+np.abs(wp_l)**2+np.abs(wp_f)**2+np.abs(wp_b)**2
        ER[n] = np.sum(wp_Abs2)*dx*dy/wp_sum_ini
        
        if (n%video_interval)==1:
            ### Save
            p21_record[...,j] = p21
            r1.append(ratc1c[n] + ratc1[n])
            j+=1

    return p21_record,tend,dt,x,y,(wk/wB),detu_p, r1

# %%
#=============================
# 讀取
#=============================

p21_record,tend,dt,x,y,nwB,dp,r1 = simulate()

# %%

# tend = 0.0005
# t = np.linspace(0, tend, 500)
# plt.plot(t, p21_record[19][19]**2)
# plt.plot(t, 1e-16*np.array(r1[1:]))

frame = 220
plt.plot(y, abs(p21_record[19, ..., frame])**2)

X, Y = meshgrid(x, y)
fig, ax = plt.subplots()
plt.pcolor(X, Y, np.abs(p21_record[..., frame])**2,shading='auto',cmap="turbo")
plt.colorbar()


# %%
