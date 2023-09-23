# %%
# 匯入套件

'''

[目標]: SLOW LIGHT;wc e^(i*Y**2)
[時間]: 

'''

from io import BytesIO
import os

import h5py
os.environ['NUMBA_ENABLE_AVX'] = '1'
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '0'
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
import time
import numpy as np
from numba import njit
from numpy import zeros, arange, ones, meshgrid, eye, exp, conj
from numpy.linalg import inv
import matplotlib.pylab as plt
from datetime import datetime
import imageio
import imageio.v2 as imageio
from datetime import datetime
date = datetime.strftime(datetime.now(),'%Y / %m / %d / (%H: %M: %S)')

# %%
# 控制臺

start_time = time.time()

'''

Parameter

'''
tend = 10e-6  # Total Time (s)
dt = 4e-10*1.06/(5.085/5)
t = arange( 0, tend + dt / 2, dt )

shrink = 10

xend = 5 # lx (mm)
yend = 5/shrink # ly (mm)
Nx = 50*(5.085/5)
Ny = 80
dx = xend / Nx
dy = yend / Ny 
x = arange( - xend / 2, xend / 2 + dx / 2, dx )
y = arange( - yend / 2, yend / 2 + dy / 2, dy )
dense = ones( ( len( y ), len( x ) ) )
X = np.repeat( x, len( y ) ).reshape( len(x), len(y) ).T
Y = np.repeat( y, len( x ) ).reshape( len(y), len(x) )

k_Cs = 2*np.pi/(780*10**-6) # (1/mm)

r31 = 2*np.pi*6*10**6 # Gamma (1/s)
r21 = 0

OD_x = 80
OD_y = OD_x * yend / xend
eta_x = OD_x * r31 / 2 / xend
eta_y = OD_y * r31 / 2 / yend

frames = 100*8 
# video_interval = round( (xend/Vx/dt/10))
video_interval = round( tend / dt / frames)
video_total = ( len( t ) - 1 ) // video_interval
progress_bar_length = 50

'''

Couple Light

'''
Ac1_r = 1 * r31
Ac2_r = 10 * r31
Ac2_l = 10 * r31

eta_x = OD_x * r31 / 2 / xend
Vx = Ac1_r**2/(2*eta_x)
t_mid = (xend/2)/Vx

l_B = 0.5/shrink # l_B eigenstate width/2
A = 2/(l_B**2)
w = A/(eta_x*k_Cs)*(Ac2_r**2)
fr = w/(2*np.pi)
print('xend/Vx =', xend/Vx, 's')
print('w =', w, 'Hz')
print('f =', fr, 'Hz')
print('T =', 1/fr, 's')

hbar = 1.05457266*10**-34
k_B = 1.38*10**-23
print('temperature:', hbar*w/k_B, 'K')

# 在這邊改沒有效果，直接到simulate()裡面改
# w_c

# wc_r = Ac * np.exp( 1j*A*(Y)**2 )
# wc_r = Ac 
# wc_l = 0.01 * Ac * np.exp( 1j*(np.pi/2) )
# wc_l = Ac
  
'''

Probe Light

'''
Ap = 1

state_n = 1
coff = (1/(np.sqrt(2**state_n*np.math.factorial(state_n))))

# wp_r

# input_wr = np.outer( np.ones( len( y ) ), exp( - ( t - tmid_p )**2 / ( 2 * ( tsig**2 ) ) ) )
input_wr = np.outer( exp( - ( (y-0.5/shrink)/ l_B )**2) , np.ones(len(t)))
# input_wr = np.outer( coff*(2/(np.pi*l_B**2))**(1/4)*exp( - ( y / l_B )**2)*2*((np.sqrt(2)/l_B)*y) , np.ones(len(t)))


'''

Slow Light

'''
# tmid_p = 7.5e-6 # 入射 Probe 光脈衝峰值時間 (usually used in EIT test where we use gaussian probe as initital condition) 
# tsig = 1.5e-6 # 入射 Probe 光脈衝寬度

'''

detunning

'''

detu_p_coff = 0 #2*r31#l_B**2 * eta_y / 4 / xslp 
# detu_c_coff = Ac**2 / ( 16 * detu_p_coff )
detu_p = detu_p_coff
detu_c = detu_p #- Ac**2 / ( 16 * detu_p_coff ) * ( ( X**2 ) / xslp**2 )

'''

Switch 

'''

# # rising & declining slope period
# tou = 1000

# t_s1 = 2000                    # rising slope center of probe
# t_s2 = (t_mid/dt) + t_s1       # declining slope center of probe
# t_s3 = 18500                   # declining slope center of couple
# # t_s4 = t_s3 + 6000             # rising slope center of couple
# t_s4 = 20000

# def switch_p_f(n):
#     switch_p = 0.5*(np.tanh((n-(t_s))/(0.25*tou))-np.tanh((n-((t_mid/dt)+t_s))/(0.25*tou)))
#     return switch_p

# def switch_c(n):
#     switch_c = -0.5*(np.tanh((n-(t_mid/dt+t_s))/(0.25*tou))-np.tanh((n-((t_mid/dt)+t_s+6000))/(0.25*tou)))+1
#     return switch_c
# switch_p = np.exp((n-25000))*0.5*(1-np.tanh((n-25000)/(0.25*1000)))
# switch_p = exp( - (( n -  25000)/2500)**2)
'''

Crank-Nicolson Parameters 

'''

# x direction  
idadx = OD_x * r31 / xend / 2 * dx
alphax = 1j / 2 / k_Cs * dx / dy**2

I = eye( len( y ) )
Ax_matrix = I * 2 * ( 1 - dy**2 / dx**2 * alphax + alphax )
Bx_matrix = I * 2 * ( 1 - 2 * dy**2 / dx**2 * alphax - alphax )
for i in range( 1, len( y ) ):
    Ax_matrix[i-1,i] = - alphax
    Bx_matrix[i-1,i] = alphax
    Ax_matrix[i,i-1] = - alphax
    Bx_matrix[i,i-1] = alphax

Axinv_matrix = inv( Ax_matrix )

# y direction  
idady = OD_y * r31 / yend / 2 * dy
alphay = 1j / 2 / k_Cs * dy / dx**2


I = eye( len( x ) )
Ay_matrix = I * 2 * ( 1 - dx**2 / dy**2 * alphay + alphay )
By_matrix = I * 2 * ( 1 - 2 * dx**2 / dy**2 * alphay - alphay )


for i in range( 1, len( x ) ):
    Ay_matrix[i-1,i] = - alphay
    By_matrix[i-1,i] = alphay
    Ay_matrix[i,i-1] = - alphay
    By_matrix[i,i-1] = alphay

Ayinv_matrix = inv( Ay_matrix )

wp_sum_ini = np.abs( Ap )**2 * xend * yend

# #%%
# # switch 時序 (interval = dt)

# # rising & declining slope period
# tou = 4e-7 # (s)

# t_s1 = 8e-7            # (s)   # rising slope center of probe
# t_s2 = 5e-6            # (s)   # declining slope center of probe
# t_s3 = 5e-6            # (s)   # declining slope center of couple
# t_s4 = 8e-6            # (s)   # rising slope center of couple


# def switch_p_f(n):
#     # return exp( - (( n -  5e-6)/1e-6)**2)
#     return 0.5*(np.tanh((n-(t_s1))/(0.25*tou))-np.tanh((n-(t_s2))/(0.25*tou)))

# def switch_c_f(n):
#     return -0.5*(np.tanh((n-(t_s3))/(0.25*tou))-np.tanh((n-(t_s4))/(0.25*tou)))+1

# def switch_c_l_f(n):
#     return 0.5*(np.tanh((n-(t_s4))/(0.25*tou))+1)

# switch_p = switch_p_f(t)
# switch_c = switch_c_f(t)
# switch_c_l = switch_c_l_f(t)

# fig, ax = plt.subplots(figsize=(8,6))
# ax.set_xlabel('t (us)')
# # ax.set_ylabel('value')
# ax.set_title('Switch')
# ax.grid()
# ax.plot(t*10**6, switch_p, linestyle='-', color='green')#, marker='^', markersize=1)
# ax.plot(t*10**6, switch_c, linestyle='-')
# ax.plot(t*10**6, switch_c_l, linestyle='--')
# ax.legend(['probe', 'couple_r', 'couple_l'])

#%%
# switch 時序 (interval = dt)

# rising & declining slope period
tou = 4e-7 # (s)

t_s1 = 8e-7            # (s)   # rising slope center of probe
t_s2 = 4.9e-6            # (s)   # declining slope center of probe
t_s3 = 5e-6            # (s)   # declining slope center of couple
t_s4 = 8e-6            # (s)   # rising slope center of couple


def switch_p_f(n):
    # return exp( - (( n -  5e-6)/1e-6)**2)
    return 0.5*(np.tanh((n-t_s1)/(0.25*tou))-np.tanh((n-t_s2)/(0.25*tou)))

def switch_off_f(n):
    return 0.5*(1-np.tanh((n-t_s3)/(0.25*tou)))

def switch_on_f(n):
    return 0.5*(1+np.tanh((n-t_s4)/(0.25*tou)))

switch_p = switch_p_f(t)
switch_on = switch_on_f(t)
switch_off = switch_off_f(t)

fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel('t (us)')
# ax.set_ylabel('value')
ax.set_title('Switch')
ax.grid()
ax.plot(t*10**6, switch_p, linestyle='-', color='green')#, marker='^', markersize=1)
ax.plot(t*10**6, switch_on, linestyle='-')
ax.plot(t*10**6, switch_off, linestyle='--')
ax.legend(['probe', 'wc_r2', 'wc_r1'])

# %%
# 程式

@njit(nogil=True, fastmath=True, cache=False)
def simulate():

    '''
    
    === 資料矩陣 =====================================================================
    
    '''
    #### pre-allocation
    rk4_cache = np.complex128(0)
    k1 = zeros((len(y),len(x)), dtype = np.cdouble)
    k2 = zeros((len(y),len(x)), dtype = np.cdouble)
    k3 = zeros((len(y),len(x)), dtype = np.cdouble)
    k4 = zeros((len(y),len(x)), dtype = np.cdouble)
    p21 = zeros((len(y),len(x)), dtype = np.cdouble)
    p31_r = zeros((len(y),len(x)), dtype = np.cdouble)
    p31_l = zeros((len(y),len(x)), dtype = np.cdouble)
    p31_f = zeros((len(y),len(x)), dtype = np.cdouble)
    p31_b = zeros((len(y),len(x)), dtype = np.cdouble)
    wp_r = zeros((len(y),len(x)), dtype = np.cdouble)
    wp_l = zeros((len(y),len(x)), dtype = np.cdouble)
    wp_f = zeros((len(y),len(x)), dtype = np.cdouble)
    wp_b = zeros((len(y),len(x)), dtype = np.cdouble)
    wc_r = zeros((len(y),len(x)), dtype = np.cdouble)
    wc_l = zeros((len(y),len(x)), dtype = np.cdouble)
    wc_f = zeros((len(y),len(x)), dtype = np.cdouble)
    wc_b = zeros((len(y),len(x)), dtype = np.cdouble)


    ER = zeros( len( t ) )
    
    p21_record = zeros( (len( y ), len( x ), video_total ), dtype = np.cdouble)
    wc_f_record = zeros( (len( y ), len( x ), video_total ), dtype = np.cdouble)
    wc_b_record = zeros( (len( y ), len( x ), video_total ), dtype = np.cdouble)
    # wc_r_record = zeros( (len( y ), len( x ), video_total ), dtype = np.cdouble)
    wc_r1_record = zeros( (len( y ), len( x ), video_total ), dtype = np.cdouble)
    wc_r2_record = zeros( (len( y ), len( x ), video_total ), dtype = np.cdouble)
    wc_l_record = zeros( (len( y ), len( x ), video_total ), dtype = np.cdouble)    
    wp_r_record = zeros( (len( y ), len( x ), video_total ), dtype = np.cdouble)
    t_record = zeros( (video_total), dtype = np.cdouble)

    '''
    
    === 運算 =======================================================================================
    
    '''
    # wc_r = r31 * np.exp( 1j*A*(Y)**2 )
    wc_r1 = Ac1_r
    wc_r2 = Ac2_r * np.exp( 1j*A*(Y)**2 )
    # wc_r = Ac 
    # wc_l = 0.01 * Ac * np.exp( 1j*(np.pi/2) )
    wc_l = Ac2_l
    
    j = 0
    
    for n in range( len( t ) ):

        # switch_p = 0.5*(np.tanh((n-(1000))/(0.25*1000))-np.tanh((n-((t_mid/dt)+1000))/(0.25*1000)))
        # switch_p = 0.5*(np.tanh((n-(t_s1))/(0.25*tou))-np.tanh((n-(t_s2))/(0.25*tou)))
        # switch_p = exp( - (( n -  12500)/2500)**2)
        # switch_p = exp( - (( n -  12500)/2500)**2)*0.5*(1-np.tanh((n-12500)/(0.25*1000)))
        
        # switch_p = np.exp((n-37500)/5000)*0.5*(1-np.tanh((n-37500)/(0.25*1000)))    

        #　initial condition
        wp_r[:,0] = Ap * input_wr[:,n] * switch_p[n]
        
        #  wp by Crank Nicoson
        # wp_r[:,1] = Axinv_matrix@(1j*idadx*dense[:,1]*p31_r[:,1])
        wp_l[:,-1] = Axinv_matrix@(1j*idadx*dense[:,-1]*p31_l[:,-1])
        wp_b[-1,:] = Ayinv_matrix@(1j*idady*dense[-1,:].T*p31_b[-1,:].T)
        wp_f[0,:]  = Ayinv_matrix@(1j*idady*dense[0,:].T*p31_f[0,:].T)
        
        wp_r[:,1]  = Axinv_matrix@(Bx_matrix@wp_r[:,0]+1j*idadx*dense[:,0]*(p31_r[:,1]+p31_r[:,0]))
        wp_l[:,-2] = Axinv_matrix@(Bx_matrix@wp_l[:,-1]+1j*idadx*dense[:,-1]*(p31_l[:,-2]+p31_l[:,-1]))
        wp_f[1,:]  = Ayinv_matrix@(By_matrix@wp_f[0,:].T+1j*idady*dense[0,:].T*(p31_f[1,:].T+p31_f[0,:].T))
        wp_b[-2,:] = Ayinv_matrix@(By_matrix@wp_b[-1,:].T+1j*idady*dense[-1,:].T*(p31_b[-2,:].T+p31_b[-1,:].T))
        
        for m in range(1,len(x)-1):
            wp_r[:,m+1] = Axinv_matrix@(Bx_matrix@wp_r[:,m]+2*dy**2/dx**2*alphax*wp_r[:,m-1]+1j*idadx*dense[:,m]*(p31_r[:,m+1]+p31_r[:,m]))
            wp_l[:,-m-2] = Axinv_matrix@(Bx_matrix@wp_l[:,-m-1]+2*dy**2/dx**2*alphax*wp_l[:,-m]+1j*idadx*dense[:,-m-1]*(p31_l[:,-m-2]+p31_l[:,-m-1]))
        
        for m in range(1,len(y)-1):
            wp_f[m+1,:] = Ayinv_matrix@(By_matrix@wp_f[m,:].T+2*dx**2/dy**2*alphay*wp_f[m-1,:].T+1j*idady*dense[m,:].T*(p31_f[m+1,:].T+p31_f[m,:].T))
            wp_b[-m-2,:] = Ayinv_matrix@(By_matrix@wp_b[-m-1,:].T+2*dx**2/dy**2*alphay*wp_b[-m,:].T+1j*idady*dense[-m-1,:].T*(p31_b[-m-2,:].T+p31_b[-m-1,:].T))
        
        # p21, p31 by Range Kutta 4th
        # switch_c = -0.5*(np.tanh((n-(t_mid/dt+1000))/(0.25*1000))-np.tanh((n-((t_mid/dt)+7000))/(0.25*1000)))+1
        # switch_c = -0.5*(np.tanh((n-(t_s3))/(0.25*tou))-np.tanh((n-(t_s4))/(0.25*tou)))+1
        # switch_c_l = 0.5*(np.tanh((n-(t_s4))/(0.25*tou))+1)
        # switch_c_l = 0        
        # switch_c = 1
        # if n == ((t_s3+t_s4)/2/dt):
        #     wc_l = Ac * np.exp( 1j*(np.pi/2) )
        #     wc_r = Ac * np.exp( 1j*A*(Y)**2 )
            # switch_c[n] = switch_c[n]*10
            # switch_c_l[n] = switch_c_l[n]*10
        # rk4_cache = -1j*( detu_p - detu_c ) * p21 + 1j/2 * ( conj( wc_r*switch_r[n] ) * p31_r + conj( wc_l * switch_l[n] ) * p31_l + conj( wc_f * switch_f[n] ) * p31_f + conj( wc_b * switch_b[n] ) * p31_b )
        rk4_cache = -1j*( detu_p - detu_c ) * p21 + 1j/2 * ( conj( wc_r1 * switch_off[n] + wc_r2 * switch_on[n] ) * p31_r + conj( wc_l * switch_on[n] ) * p31_l + conj( wc_f ) * p31_f + conj( wc_b ) * p31_b )
        k1 = rk4_cache - r21* p21
        k2 = rk4_cache - r21*(p21 + dt/2 * k1)
        k3 = rk4_cache - r21*(p21 + dt/2 * k2)
        k4 = rk4_cache - r21*(p21 + dt   * k3)
        p21 = p21 + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # p31
        rk4_cache = 1j/2 * wp_r + 1j/2 * (wc_r1 * switch_off[n] + wc_r2 * switch_on[n] ) * p21 - r31 / 2 * p31_r
        k1 = -1j * detu_p * (p31_r            ) + rk4_cache
        k2 = -1j * detu_p * (p31_r + dt/2 * k1) + rk4_cache
        k3 = -1j * detu_p * (p31_r + dt/2 * k2) + rk4_cache
        k4 = -1j * detu_p * (p31_r + dt   * k3) + rk4_cache
        p31_r = p31_r + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        rk4_cache = 1j/2 * wp_l + 1j/2 * (wc_l * switch_on[n]) * p21 - r31 / 2 * p31_l
        k1 = -1j * detu_p * (p31_l            ) + rk4_cache
        k2 = -1j * detu_p * (p31_l + dt/2 * k1) + rk4_cache
        k3 = -1j * detu_p * (p31_l + dt/2 * k2) + rk4_cache
        k4 = -1j * detu_p * (p31_l + dt   * k3) + rk4_cache
        p31_l = p31_l + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # rk4_cache = 1j/2 * wp_f + 1j/2 * wc_f * switch_f[n] * p21 - r31 / 2 * p31_f
        rk4_cache = 1j/2 * wp_f + 1j/2 * wc_f * p21 - r31 / 2 * p31_f
        k1 = -1j * detu_p * (p31_f            ) + rk4_cache
        k2 = -1j * detu_p * (p31_f + dt/2 * k1) + rk4_cache
        k3 = -1j * detu_p * (p31_f + dt/2 * k2) + rk4_cache
        k4 = -1j * detu_p * (p31_f + dt   * k3) + rk4_cache
        p31_f = p31_f + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # rk4_cache = 1j/2 * wp_b + 1j/2 * wc_b * switch_b[n] * p21 - r31 / 2 * p31_b
        rk4_cache = 1j/2 * wp_b + 1j/2 * wc_b * p21 - r31 / 2 * p31_b
        k1 = -1j * detu_p * (p31_b            ) + rk4_cache
        k2 = -1j * detu_p * (p31_b + dt/2 * k1) + rk4_cache
        k3 = -1j * detu_p * (p31_b + dt/2 * k2) + rk4_cache
        k4 = -1j * detu_p * (p31_b + dt   * k3) + rk4_cache
        p31_b = p31_b + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Energy ratio
        wp_Abs2 = np.abs( wp_r )**2 + np.abs( wp_l )**2 + np.abs( wp_f )**2 + np.abs( wp_b )**2
        ER[n] = np.sum( wp_Abs2 ) * dx * dy / wp_sum_ini
        
        
        if ( n % video_interval ) == 1:
            # Save
            p21_record[...,j] = p21
            wc_f_record[...,j] = wc_f
            wc_b_record[...,j] = wc_b
            wc_r1_record[...,j] = wc_r1 * switch_off[n]
            wc_r2_record[...,j] = wc_r2 * switch_on[n]
            wc_l_record[...,j] = wc_l * switch_on[n]
            wp_r_record[...,j] = wp_r
            t_record[j] = n*dt
                
            j += 1
            
    return p21_record, wc_f_record, wc_b_record, wc_r1_record, wc_r2_record, wc_l_record, input_wr, wp_r_record, t_record

# %%
# 計算

p21_record, wc_f_record, wc_b_record, wc_r1_record, wc_r2_record, wc_l_record, input_wr, wp_r_record, t_record = simulate()

cal_time = time.time() - start_time

#%% 
# 繪圖

# Normalization (ALL)
# p21_record = p21_record / np.max(np.abs(p21_record))
# wc_f_record = wc_f_record / np.max(np.abs(wc_f_record))
# wc_b_record = wc_b_record / np.max(np.abs(wc_b_record))

[X,Y] = meshgrid( x, y )

# plt.figure( figsize = ( 5, 5 * yend / xend - 1), dpi = 100 )
plt.figure( figsize = ( 5, 5 ), dpi = 100 )
# plt.pcolor( X, Y, np.abs( p21_record[...,-1] )**2 / np.max( np.abs( p21_record[...,-1] )**2 ), shading = 'auto', cmap = "afmhot", vmin = 0, vmax = 1)
plt.pcolor( X, Y, np.abs( wp_r_record[...,0] )**2 , shading = 'auto', cmap = "turbo")
plt.xlabel('x(mm)')
plt.ylabel('y(mm)')
plt.title( f'OD: {OD_x:.2f}, $\Omega$c: {Ac2_r/r31:.2f} $\Gamma$, Time: {tend*1e6:.2f} $\mu$s, Cal_Time: {int(cal_time // 60)} m {int(cal_time % 60)} s')
plt.colorbar()

#%%
# the Numerical sol. & Analytical sol. 

# plt.plot( Y, (np.abs( p21_record[...,-1] )**2))
# plt.show()
input_wr_test = exp( - ( (y)/ l_B )**2) 
# test1 = (np.abs( p21_record[...,-1] )**2)
# plt.plot( y, test1.T[0])
# plt.plot( y, test1.T[-1])
# plt.show()
plt.figure(figsize=(10,6))
plt.xlabel( "y (mm)" )
plt.ylabel( 'Normaling Intensity' )
plt.title('Section at x = 0 (mm)')
# plt.plot(y,np.abs(p21_record[:,25,-1])**2/np.max(np.abs(p21_record[:,25,-1])**2),'r',y,input_wr[:,0]**2/np.max(input_wr[:,0]**2),'b--')
plt.plot(y,np.abs(p21_record[:,25,-1])**2/np.max(np.abs(p21_record[:,25,-1])**2),'r',y,input_wr_test**2/np.max(input_wr_test**2),'b--')
plt.legend(['Numerical sol.', 'Analytical sol.'], loc='upper right')
plt.show()

#%%
# peak trackor

fig, ax = plt.subplots(figsize=(16,6))
ax.set_xlabel('t (us)')
ax.set_ylabel('y (mm)')
ax.set_title('Coherent state')
ax.grid()

peak = np.array([])

for i in range(0, np.size( p21_record, 2 )):
# for i in range(600,750):
    p21_square = np.abs(p21_record[:,25,i])**2
    p21_max_yidx = np.where(p21_square==np.max(p21_square))[0]
    p21_max_yval = (- yend / 2 + p21_max_yidx*dy)
    # ax.scatter(t_record[i]*10**6, p21_max_yval)
    peak = np.append(peak, p21_max_yval)

plt.plot(t_record*10**6, peak, linestyle='--', color='sienna', marker='o')
# plt.plot( peak, linestyle='--', marker='o')

#%%

test1, test2, test3, test4, test5, test6, test7, test8, test9, test10 = p21_record, wc_f_record, wc_b_record, wc_r1_record, wc_r2_record, wc_l_record, input_wr, wp_r_record, t_record, test 
#%%

fig, ax = plt.subplots(figsize=(16,6))
ax.set_xlabel('t (us)')
ax.set_ylabel('y (mm)')
ax.set_title('Coherent state')
ax.grid()

peak1 = np.array([])
peak = np.array([])

N1 = 625
N2 = 799
# for i in range(0, np.size( test1, 2 )):
for i in range(N1, N2):
    p21_square1 = np.abs(test1[:,25,i])**2
    p21_max_yidx1 = np.where(p21_square1==np.max(p21_square1))[0]
    p21_max_yval1 = (- yend / 2 + p21_max_yidx1*dy)
    # ax.scatter(t_record[i]*10**6, p21_max_yval)
    peak1 = np.append(peak1, p21_max_yval1)
    
# for i in range(0, np.size( p21_record, 2 )):
for i in range(N1, N2):
    p21_square = np.abs(p21_record[:,25,i])**2
    p21_max_yidx = np.where(p21_square==np.max(p21_square))[0]
    p21_max_yval = (- yend / 2 + p21_max_yidx*dy)
    # ax.scatter(t_record[i]*10**6, p21_max_yval)
    peak = np.append(peak, p21_max_yval)
    
# plt.plot(t_record*10**6, peak, linestyle='--', color='sienna', marker='o')
plt.plot( t_record[N1:N2]*10**6, peak, linestyle='--', marker='o')
# plt.plot( t_record[N1:N2]*10**6, peak1, linestyle='--', marker='o')
plt.legend(['Nx = 50', 'Nx = 500'])

#%%

'''
P21 - Phase (wp_r_record)
'''
# rep21 = np.real( wc_r_record )
# imp21 = np.imag( wc_r_record )
# phase = np.arctan( imp21 / rep21 )

# plt.xlabel( "x (mm)" )
# plt.ylabel( "y (mm)" )
# plt.title( r"$\omega_{c}$" + " [ Phase ]" )

# plt.pcolor( X, Y, phase[...,-1], shading = 'auto', cmap = 'turbo')
# cbar = plt.colorbar()
# cbar.set_ticks( np.linspace( np.min( phase[...,-1] ), np.max( phase[...,-1] ), 5 ) )
# cbar.set_ticklabels( ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'] )
#%%

'''

Output

'''
file_path = '/home/hsuchunyen/LG/0818data2/'
test = 'Nx500'
filename = f'{test}_Lx-{xend}mm, Ly-{yend}mm, OD-{OD_x:.2f}_Ac-{Ac2_r/r31:.2f}Gamma_Time-{tend*1e6:.2f}us, Nx-{Nx}' # File Name

'''

=== p21.h5 =========================================================================

'''
# with h5py.File(f'{file_path}{filename}.h5', 'w') as f:
with h5py.File('/home/hsuchunyen/LG/0818data2/Nx500.h5', 'w') as f:
    f['p21_record'] = p21_record[...]
    f['x'] = x
    f['y'] = y
    f['t'] = t
    f['wc_r1_record'] = wc_r1_record[...]
    f['wc_r2_record'] = wc_r2_record[...]
    f['wc_l_record'] = wc_l_record[...]
    f['input_wr'] = input_wr[...]
    f['wp_r_record'] = wp_r_record[...]
    
#%%

with h5py.File('/home/hsuchunyen/LG/0818data2/Nx500.h5', 'r') as f1:
    output = f1['x'][...]
    
#%%
# 輸出

[X,Y] = meshgrid( x, y )

Np = np.size( p21_record, 2 )

t_lb = [ round( ( i * video_interval + 1 ) * dt*10**6, 5 ) for i in range( Np ) ]

# '''

# === 2D.mp4 ==========================================================================

# '''
# def MP4_2D(p21_record):
    
#     files = list()
    
#     for j in range( p21_record.shape[-1] ):
        
#         files.append( BytesIO() )

        
#         # 繪製40張折線圖
#         rep21 = np.real( p21_record )
#         imp21 = np.imag( p21_record )
#         phase = np.arctan( imp21 / rep21 )
        
#         '''
#             P21
            
#         '''
#         plt.figure( figsize = ( 8, 16 * yend / xend - 2 ), dpi = 200 )
#         plt.subplot( 2, 1, 1 )
#         plt.pcolor( X, Y, np.abs( p21_record[...,j] )**2 / np.max( np.abs( p21_record[...,j] )**2 ), shading = 'auto', cmap = "turbo", vmin = 0, vmax = 1)
#         # plt.pcolor(X[:,20:],Y[:,20:],np.abs(p21_record[:,20:,j])**2,shading='auto',cmap="turbo")
#         # plt.pcolor(X[5:,:],Y[5:,:],np.abs(p21_record[5:,:,j])**2,shading='auto',cmap="turbo")
#         plt.xlabel( 'x(mm)' )
#         plt.ylabel( 'y(mm)' )
#         plt.colorbar()
#         # plt.title(str(t_lb[j])+"ms, "+str(format(nwB,'.2f'))+"wB")
#         # plt.title( str( t_lb[j] ) + "ms, " + "nwB = " + str(nwB) )
        
        
#         '''
#             Phase
            
#         '''
#         plt.subplot( 2, 1, 2 )
#         plt.pcolor( X, Y, phase[...,j], shading = 'auto', cmap = "turbo" )
#         # plt.clim(0,np.pi)
#         plt.title( "phase" )
#         plt.xlabel( "x(mm)" )
#         plt.ylabel( "y(mm)" )
#         plt.colorbar()
        
#         plt.tight_layout()

#         # 保存圖片文件
#         plt.savefig(files[j])
#         plt.close()

#     # 生成 mp4
#     with imageio.get_writer(f'{file_path}{filename}_2D.mp4', mode='I') as writer:
#         for f in files:
#             image = imageio.imread(f)
#             writer.append_data(image)
            
#     return None

'''

=== 3D.mp4 ==========================================================================

'''
def MP4_3D():
    
    files = list()
    
    # for j in range( 0, 100 ):
    for j in range( 0, p21_record.shape[-1] ):
        
        files.append( BytesIO() )
        fig = plt.figure( figsize=(16, 12), dpi = 200 )
        fig.suptitle(f'OD: {OD_x:.2f}, Ac: {Ac2_r/r31:.2f}' + '$\Gamma$, ' + '$\Gamma$' + f': {r31*1e-6} MHz' + f' | Time: {tend*1e6:.2f}us | CalTime: {int(cal_time // 60)} m {int(cal_time % 60)} s | {date}', fontsize=18)
        
        alpha = 1
        
        # p21_norm = np.abs( p21_record[...,j] )**2 / np.max( np.abs( p21_record[...] )**2 )
        
        # wc_f_norm = np.abs( wc_f_record[...,j] )**2 / np.max( np.abs( wc_f_record[...] )**2 )
        # wc_b_norm = np.abs( wc_b_record[...,j] )**2 / np.max( np.abs( wc_b_record[...] )**2 )
        
        # wp_r_norm_in = np.abs( wp_r_record[:, 0, -1] )**2 / np.max(np.abs( wp_r_record[:, 0, -1] )**2 )
        # wp_r_norm_out = np.abs( wp_r_record[:, -1, -1] )**2 / np.max(np.abs( wp_r_record[:, -1, -1] )**2, )

        # [67, 34, 100]
        # wp_r_norm = np.abs( wp_r_record[...,j] )**2 / np.max( np.abs( wp_r_record[...,j] )**2 )
        # wp_r_norm_in = np.abs( wp_r_record[33, 0, :] )**2 / np.max(np.abs( wp_r_record[33, 0, :] )**2 )
        # wp_r_norm_out = np.abs( wp_r_record[33, -1, :] )**2 / np.max(np.abs( wp_r_record[33, 0, :] )**2 )
        
        # delay_time = t_lb[ int( np.where( wp_r_norm_out == np.max( wp_r_norm_out ) )[0][0] )] - tmid_p*1e3
        
        '''
        Wp_r - 2D
        '''
        wp_r_2D = fig.add_subplot( 2, 2, 1)#, aspect = 1)
        
        wp_r_2D.set_xlabel( "x (mm)" )
        wp_r_2D.set_ylabel( 'y (mm)' )
        # wp_r_2D.set_title( r"$\omega_{p}$" + f" [ {t_lb[j]:.4f} ms, nwB = {str(nwB)} ]" )
        
        plt.pcolor( X, Y, np.abs(p21_record[...,j]), shading = 'auto', cmap = 'afmhot')
        plt.colorbar()

        
        
        '''
        P21 - 3D
        '''
        # ax = fig.add_subplot(221, projection='3d')
        
        # ax.set_zlim(0, 1) 
        # ax.view_init(elev=60, azim=-45)
        # ax.set_box_aspect((1, 1, 1))
        # plt.colorbar(ax.plot_surface(X, Y, np.abs( p21_record[...,j] )**2 / np.max( np.abs( p21_record[...,j] )**2 ), cmap = style), location = "left", shrink=0.5)
        # plt.title( "p21" )
        # plt.xlabel( "x(mm)" )
        # plt.ylabel( "y(mm)" )
        
        
        # '''
        # P21 - Phase (wp_r_record)
        # '''
        # rep21 = np.real( wc_r_record )
        # imp21 = np.imag( wc_r_record )
        # phase = np.arctan( imp21 / rep21 )
        
        # phase_2D = fig.add_subplot( 2, 2, 3, aspect = 1)
        
        # phase_2D.set_xlabel( "x(mm)" )
        # phase_2D.set_ylabel( "y(mm)" )
        # phase_2D.set_title( r"$\omega_{c}$" + " [ Phase ]" )
        
        # plt.pcolor( X, Y, phase[...,j], shading = 'auto', cmap = 'turbo')
        # cbar = plt.colorbar()
        # cbar.set_ticks( np.linspace( np.min( phase[...,j] ), np.max( phase[...,j] ), 5 ) )
        # cbar.set_ticklabels( ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'] )
        
        
        # '''
        # wp_r_3D
        # '''
        # wp_r_3D = fig.add_subplot( 2, 2, 2, projection='3d')
        
        # wp_r_3D.set_zlim(0, 1) 
        # wp_r_3D.view_init(elev=50, azim=-45)
        # wp_r_3D.set_box_aspect((1, 1, 1))
        # wp_r_3D.set_xlabel( "x (mm)", labelpad = 8 )
        # wp_r_3D.set_ylabel( "y (mm)", labelpad = 8 )
        # wp_r_3D.set_title( r"$\omega_{c}$" + f"(Right)[ {t_lb[j]:.4f} us ]" )
        
        # cbar = plt.colorbar(wp_r_3D.plot_surface(X, Y, wp_r_norm , cmap = 'afmhot', alpha = alpha), location = "right", shrink=0.5)
        # cbar.set_ticks( np.linspace( np.min( wp_r_norm  ), np.max( wp_r_norm  ), 5 ) )
        # cbar.set_ticklabels( np.linspace( 0, 1, 5 ) )
        switch = fig.add_subplot(2, 2, 2)
        switch.set_xlabel( 't (us)' )
        switch.set_ylabel( 'on/off' )
        switch.set_title('Switch ' + f"T = [ {t_lb[j]:.4f} us ]" )
        plt.grid()
        plt.plot(t*10**6, switch_p, linestyle='-', color='green')
        plt.plot(t*10**6, switch_on, linestyle='-')
        plt.plot(t*10**6, switch_off, linestyle='--')
        # plt.plot(t*10**6, wc_l_record[:,25,j], linestyle='-', color='green')
        plt.scatter(t_lb[j], switch_on[j*video_interval])
        plt.legend(['probe', 'wc_r2', 'wc_r1'])

        # '''
        # slow_light
        # '''
        # slow_light = fig.add_subplot(224)
        
        # slow_light.set_xlabel( "t_lb(us)" )
        # slow_light.set_ylabel( '' )
        # slow_light.set_xticks(np.linspace(0, tend, 5), np.linspace(0, tend*1e6, 5))
        # # slow_light.set_title( r"$\omega_{p}$" + f" [delay: {delay_time*1e3:.2f} us, vg: {xend/delay_time:.2f} m/s]" )
        # slow_light.set_title( r"$\omega_{p}$")
        # plt.plot( np.linspace( -0.2, 0.2, len( wp_r_norm_in ) ), wp_r_norm_in, label = 'in')
        # plt.plot(np.linspace( -0.2, 0.2, len( wp_r_norm_in ) ), wp_r_norm_out,':', label = 'out')
        # plt.legend()
        solution_nor = fig.add_subplot(2, 2, 4)
        solution_nor.set_xlabel( "y (mm)" )
        solution_nor.set_ylabel( 'Normaling Intensity' )
        solution_nor.set_title('Section at x = 0 (mm)')
        # plt.plot(y,np.abs(wp_r_record[:,25,j])**2,'r')
        plt.ylim((-0.05, 1.05))
        plt.plot(y,np.abs(wp_r_record[:,25,j])**2/np.max(np.abs(wp_r_record[:,25,j])**2),'r',y,np.abs(input_wr[:,j])**2/np.max(np.abs(input_wr[:,j])**2),'b--')
        # plt.plot(y,np.abs(wp_r_record[:,25,j])**2,'r',y,np.abs(input_wr[:,j])**2/np.max(np.abs(input_wr[:,j])**2),'b--')
        plt.legend(['Numerical sol.', 'Analytical sol.'])
        
        # solution = fig.add_subplot(2, 2, 3)
        # solution.set_xlabel( "y (mm)" )
        # solution.set_ylabel( 'Intensity' )
        # solution.set_title('Section at x = 0 (mm)')
        # plt.ylim((-0.02, 0.32))
        # plt.plot(y,np.abs(wp_r_record[:,25,j])**2,'r')
        # plt.legend(['Numerical sol.'])
        
        solution = fig.add_subplot(2, 2, 3)
        solution.set_xlabel( "y (mm)" )
        solution.set_ylabel( 'Intensity' )
        solution.set_title('Section at x = -2.5/2.5 (mm)')   
        plt.ylim((-0.02, 1.02))     
        plt.plot(y,np.abs(wp_r_record[:,0,j])**2, color='sienna')
        plt.plot(y,np.abs(wp_r_record[:,-1,j])**2, color='sandybrown', linestyle='--', marker='o')
        plt.legend(['in', 'out'])

        fig.tight_layout()
        
        # 保存圖片文件
        plt.rcParams['font.size'] = 16
        plt.savefig(files[j])
        plt.close()
        
        '''
        Progress Bar
        '''
        # 計算當前進度
        progress = int( j / len( t_lb ) * progress_bar_length )
        
        # 輸出進度條
        print('\r'+'[', '|' * progress, '|', ' ' * (progress_bar_length - progress - 1), ']', f' {round( j / p21_record.shape[-1] * 100 + 1 , 1 )} %     ', sep='', end='')
        

    # 生成 mp4
    with imageio.get_writer(f'{file_path}{filename}.mp4', mode='I', fps=5) as writer:
        for f in files[1 : -1]:
            image = imageio.imread(f)
            writer.append_data(image)
    
    return None

# plt.plot(np.linspace(0, tend, len(switch_f)), switch_f)
# plt.plot([0, tend], [1, 1])
MP4_3D()

# %%
# 慢光結果
# %matplotlib widget

'''

y: all
x:-1 , 1
t: all

'''
wp_r_norm_in = np.abs( wp_r_record[:, 0, -1] )**2 / np.max(np.abs( wp_r_record[:, 0, -1] )**2 )
wp_r_norm_out = np.abs( wp_r_record[:, -1, -1] )**2 / np.max(np.abs( wp_r_record[:, -1, -1] )**2, )
# plt.xticks(np.linspace(0, tend, 5), np.linspace(0, tend*1e6, 5))
# plt.plot( np.linspace( 0, tend, len( wp_r_norm_in ) ), wp_r_norm_in, label = f'({(-1 + 0 * dx):.0f}, {(-1 + 33 * dy):.0f})')
# plt.plot(np.linspace( 0, tend, len( wp_r_norm_in ) ), wp_r_norm_out, label = f'({(-1 + 34 * dx):.0f}, {(-1 + 33 * dy):.0f})')

plt.plot( np.linspace( -0.2, 0.2, len( wp_r_norm_in ) ), wp_r_norm_in, label = f'({(-1 + 0 * dx):.0f}, {(-1 + 33 * dy):.0f})')
plt.plot(np.linspace( -0.2, 0.2, len( wp_r_norm_in ) ), wp_r_norm_out,':', label = f'({(-1 + 34 * dx):.0f}, {(-1 + 33 * dy):.0f})')
plt.legend()
plt.show()
#%%
# switch 時序 (interval = dt)

t = arange( 0, tend + dt / 2, dt )
c = np.array([])
p = np.array([])

for n in range( len(t) ):
    ### input probe
    # switch_p = 0.5*(np.tanh((n-(t_s1))/(0.25*tou))-np.tanh((n-(t_s2))/(0.25*tou)))
    switch_p = exp( - (( n -  25000)/2500)**2)
    p = np.append(p, switch_p)
    switch_c = -0.5*(np.tanh((n-(t_s3))/(0.25*tou))-np.tanh((n-(t_s4))/(0.25*tou)))+1
    # switch_c = 1
    c = np.append(c, switch_c)
    
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel('t (us)')
# ax.set_ylabel('value')
ax.set_title('Switch')
ax.grid()
ax.plot(t*10**6, c, linestyle='-')
ax.plot(t*10**6, p, linestyle='--')#, marker='^', markersize=1)
ax.legend(['couple', 'probe'])

# %%
# switch 時序 (interval = 750*dt)

t_plot = arange( dt, tend + dt / 2, dt*video_interval )
c = np.array([])
p = np.array([])


for n in range( frames ):
    ### input probe
    n = n*video_interval    
    # switch_p = 0.5*(np.tanh((n-(t_s1))/(0.25*tou))-np.tanh((n-(t_s2))/(0.25*tou)))
    switch_p = exp( - (( n -  25000)/12500)**2)
    p = np.append(p, switch_p)
    switch_c = -0.5*(np.tanh((n-(t_s3))/(0.25*tou))-np.tanh((n-(t_s4))/(0.25*tou)))+1
    c = np.append(c, switch_c)
    
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel('t (us)')
# ax.set_ylabel('value')
ax.set_title('Switch')
ax.grid()
ax.plot(t_plot*10**6, c, linestyle='-')
ax.plot(t_plot*10**6, p, linestyle='--')#, marker='^', markersize=1)
ax.legend(['couple', 'probe'])

#%%
# 入口/出口 probe 時序 (t=-1)
# plt.plot(y,np.abs(wp_r_record[:,0,-1])**2/np.max(np.abs(wp_r_record[:,0,-1])**2))
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel('t (us)')
ax.set_ylabel('intensity')
ax.set_title('Probe light')
ax.grid()

# t_plot = arange( dt, tend + dt / 2, dt*video_interval )
# test_t = np.arange(1, np.size( p21_record, 2 )+1)*dt_plot*10**6

# for j in range(wp_r_record.shape[2]):
#     # plt.plot(y,np.abs(wp_r_record[:,0,j])**2, linestyle='--')
#     plt.plot(y,np.abs(wp_r_record[:,-1,j])**2)
ax.plot( t_record*10**6, wp_r_record[40][0], color='r', marker='o', markersize=3)
ax.plot( t_record*10**6, wp_r_record[40][-1], color='blue')#, marker='o')
ax.legend(['input', 'output'])
# ax.plot(t_plot*10**6, c, linestyle='-')
# ax.plot(t_plot*10**6, p, linestyle='--')#, marker='^', markersize=1)
# ax.legend(['input', 'output','sw_couple', 'sw_probe'])

#%% 
# 繪圖 Omega_c

fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel('t (us)')
ax.set_ylabel('intensity')
ax.set_title('Probe light')
ax.grid()
ax.plot( t_record*10**6, wc_l_record[1][0]/r31, linewidth='3')
ax.plot( t_record*10**6, wc_r1_record[1][0]/r31, color='green')
ax.plot( t_record*10**6, wc_r2_record[1][0]/r31, linestyle='--')
ax.legend(['wc_l', 'wc_r1', 'wc_r2'])

# ax.plot( wp_r_record[..., 0, 0], linestyle='--')
# %%
test1 = np.abs(test)
# %%
test1 = [20, 15, 10, 5, 2.5, 1.25]
test2 = [0.19, 0.21, 0.27, 0.68, 2.4, 8.9]
test3 = [0.033, 0.06, 0.13, 0.53, 2.1, 8.5]
test4 = [0.19/0.033, 0.21/0.06, 0.27/0.13, 0.68/0.53, 2.4/2.1, 8.9/8.5]
# plt.plot(test1, test2)
plt.plot(test1, test4)
# %%
