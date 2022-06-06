# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:19:19 2022

@author: Luoji
"""

import pandas as pd
import random
import numpy as np
import time
import pickle as pk
import multiprocessing as mp
from scipy.sparse import lil_matrix
from numpy import random as nran
#%%
def Year_structure(age):#根据年龄定义每个个体的易感性
    """ age: 20m*1,  susc: 20m*1  """
    susc = np.ones(len(age))
    susc[age['age']<=14] = 0.34
    susc[age['age']>64] = 1.47   
    return np.array(susc)

def split(full_list,shuffle,ratio):
    n_total = len(full_list)
    offset = round(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
        return sublist_1,sublist_2

def patient_zero(N,numI,r):#随机生成初始感染者
    numA = int(numI*(1-r)/r)
    num = numI+numA
    Infecters = random.sample(range(N),num) 
    """ InfectStatus: 20m*1  """
    InfectStatus = np.zeros(N,int)        
    IndexI,IndexA=split(Infecters,True,r)    
    for i in IndexI:
        InfectStatus[i] = 1 
    for i in IndexA:
        InfectStatus[i] = 2                  
    return InfectStatus

def RMij(Mij,Age_struc):#生成3×3的接触矩阵
    
    Mij0_14=Mij['0-4']+Mij['5-9']+Mij['10-14']
    P0_14= Age_struc['P'][8:11]
    P0_14=P0_14/sum(P0_14)
    m11=np.dot(Mij0_14[0:3],P0_14)
    P15_64= Age_struc['P'][11:21]
    P15_64=P15_64/sum(P15_64)
    m21=np.dot(Mij0_14[3:13],P15_64)
    m31=Mij0_14[13]
    
    Mij15_64=Mij['15-19']+Mij['20-24']+Mij['25-29']+Mij['30-34']+Mij['35-39']+Mij['40-44']+Mij['45-49']+Mij['50-54']+Mij['55-59']+Mij['60-64']
    m12=np.dot(Mij15_64[0:3],P0_14)
    m22=np.dot(Mij15_64[3:13],P15_64)
    m32=Mij15_64[13]
    
    Mij65=Mij['65+']
    m13=np.dot(Mij65[0:3],P0_14)
    m23=np.dot(Mij65[3:13],P15_64)
    m33=Mij65[13]
    Mij3=np.array([[m11,m12,m13],[m21,m22,m23],[m31,m32,m33]])
    return Mij3

def Info2SIAR(Info,a):#根据初始信息（Info)生成0时刻的SIARa; 记录了在0时刻,每个子区域内, a(b,c类似)年龄段个体的S,I,A,R的数量和总数
    """ M: number of sub-areas """
    """ Persona: 10m*5 """
    Persona = Info[Info['Susc']==a]
    x=pd.DataFrame(Persona)
    x.loc[:,'Count']=1
    """ place: 5000*1 """
    place = list(range(1,M+1,1))
    """ S: 5000*1, I:5000*1, A:5000*1, R:5000*1 """
    S = np.zeros(M)
    I = np.zeros(M)
    A = np.zeros(M)
    R = np.zeros(M)
    """ SIARa: 5000*5  """
    SIARa = pd.DataFrame({'Area0':place,'S':S,'I':I,'A':A,'R':R})
    """ data1: 5000*6 """
    data1 = Persona.groupby(['Area0','Status']).agg({'Count':sum})
    data1 = data1.reset_index()

    temp = data1[(data1['Status']==0)]
    """ x: 5000*6 """
    x = pd.merge(temp, SIARa,how='right',on=['Area0'],sort='False')    
    x = x.fillna(0)
    SIARa['S'] = x['Count']
    
    temp = data1[(data1['Status']==1)]
    """ x: 5000*6 """
    x = pd.merge(temp, SIARa,how='right',on=['Area0'],sort='False')    
    x = x.fillna(0)
    SIARa['I'] = x['Count']
    
    temp = data1[(data1['Status']==2)]
    """ x: 5000*6 """
    x = pd.merge(temp, SIARa,how='right',on=['Area0'],sort='False')    
    x = x.fillna(0)
    SIARa['A'] = x['Count']
    
    SIARa['NUM'] = SIARa['S']+SIARa['I']+SIARa['A']+SIARa['R']    
    return SIARa

#kind=0,1,2表示第几类人群
def SIAR(SIARkind,kind,susckind,SIARa,SIARb,SIARc,Mij3,beta,gamma,alpha,r,dt):#模拟dt时间段内的感染过程
    m11 = Mij3[kind,0]
    m12 = Mij3[kind,1]
    m13 = Mij3[kind,2]
    #m21 = Mij3[1,0]
    #m22 = Mij3[1,1]
    #m23 = Mij3[1,2]
    #m31 = Mij3[2,0]
    #m32 = Mij3[2,1]
    #m33 = Mij3[2,2]
    """ EdS: 5000*1, EdS2I: 5000*1, EdS2A: 5000*1, EdI2R: 5000*1, EdA2R: 5000*1"""
    EdS = -dt * susckind * beta * SIARkind['S'] * \
          ((m11*SIARa['I']/SIARa['NUM']) + alpha*(m11*SIARa['A']/SIARa['NUM'])\
          +(m12*SIARb['I']/SIARb['NUM']) + alpha*(m12*SIARb['A']/SIARb['NUM'])\
          +(m13*SIARc['I']/SIARc['NUM']) + alpha*(m13*SIARc['A']/SIARc['NUM']))
    EdS2I = -r*EdS
    EdS2A = -(1-r)*EdS
    EdI2R = dt*gamma * SIARkind['I']
    EdA2R = dt*gamma * SIARkind['A']
    
    EdS2I[EdS2I<0]=0
    EdS2A[EdS2A<0]=0
    EdI2R[EdI2R<0]=0
    EdA2R[EdA2R<0]=0
    EdS2I=EdS2I.fillna(0)
    EdS2A=EdS2A.fillna(0)
    EdI2R=EdI2R.fillna(0)
    EdA2R=EdA2R.fillna(0)
    """ dS2I: 5000*1, dS2A: 5000*1, dI2R: 5000*1, dA2R: 5000*1"""
    dS2I = np.random.poisson(EdS2I)
    dS2A = np.random.poisson(EdS2A)
    dI2R = np.random.poisson(EdI2R)
    dA2R = np.random.poisson(EdA2R)
    
    dI2R = np.minimum(dI2R,SIARkind['I'])
    dA2R = np.minimum(dA2R,SIARkind['A'])
    
    """ dS: 5000*1, dI: 5000*1, dA: 5000*1, dR: 5000*1 """
    dS = -(dS2I+dS2A)
    probplace=(SIARkind['S']+dS)<0
    probS=SIARkind['S'][probplace]
    probS2I=np.rint(r*probS)
    probS2A=probS-probS2I
    dS[probplace]=-probS
    dS2I[probplace]=probS2I
    dS2A[probplace]=probS2A
    
    dI = dS2I-dI2R
    dA = dS2A-dA2R
    dR = dI2R+dA2R
    """ SIARkindnew: 5000*5"""
    #SIARkindnew = pd.DataFrame({'S':SIARkind['S']+np.rint(dS),'I':SIARkind['I']+np.rint(dI),'A':SIARkind['A']+np.rint(dA),'R':SIARkind['R']+np.rint(dR),'NUM':SIARkind['NUM']}) 
    SIARkindnew = pd.DataFrame({'S':SIARkind['S']+dS,'I':SIARkind['I']+dI,'A':SIARkind['A']+dA,'R':SIARkind['R']+dR,'NUM':SIARkind['NUM']})       
      
    return SIARkindnew,dS2I

def Stop_IAR(SIARkind,gamma,dt):#模拟dt时间段内的感染过程
    
    EdI2R = dt*gamma * SIARkind['I']
    EdA2R = dt*gamma * SIARkind['A']
    
    EdI2R[EdI2R<0]=0
    EdA2R[EdA2R<0]=0

    EdI2R=EdI2R.fillna(0)
    EdA2R=EdA2R.fillna(0)

    dI2R = np.random.poisson(EdI2R)
    dA2R = np.random.poisson(EdA2R)
    
    #dI2R = np.rint(dI2R)
    #dA2R = np.rint(dA2R)
    
    dI2R = np.minimum(dI2R,SIARkind['I'])
    dA2R = np.minimum(dA2R,SIARkind['A'])
    
    dI = -dI2R
    dA = -dA2R
    dR = dI2R+dA2R
    """ SIARkindnew: 5000*5"""
    #SIARkindnew = pd.DataFrame({'S':SIARkind['S']+np.rint(dS),'I':SIARkind['I']+np.rint(dI),'A':SIARkind['A']+np.rint(dA),'R':SIARkind['R']+np.rint(dR),'NUM':SIARkind['NUM']}) 
    SIARkindnew = pd.DataFrame({'S':SIARkind['S'],'I':SIARkind['I']+dI,'A':SIARkind['A']+dA,'R':SIARkind['R']+dR,'NUM':SIARkind['NUM']})       
    
    
    return SIARkindnew

def SIAR_P(a_Norm,b_Norm,c_Norm,SIARkind,kind,susckind,SIARa,SIARb,SIARc,Mij3,beta,gamma,alpha,r,dt):#模拟dt时间段内的感染过程
    m11 = Mij3[kind,0]
    m12 = Mij3[kind,1]
    m13 = Mij3[kind,2]
    """ EdS: 5000*1, EdS2I: 5000*1, EdS2A: 5000*1, EdI2R: 5000*1, EdA2R: 5000*1"""
    '''EdS = -dt * susckind * beta * SIARkind['S'] * \
          ((m11*SIARa['I']/a_Norm['NUM']) + alpha*(m11*SIARa['A']/a_Norm['NUM'])\
          +(m12*SIARb['I']/b_Norm['NUM']) + alpha*(m12*SIARb['A']/b_Norm['NUM'])\
          +(m13*SIARc['I']/c_Norm['NUM']) + alpha*(m13*SIARc['A']/c_Norm['NUM']))'''
    EdS = -dt * susckind * beta * SIARkind['S'] * \
          ((m11*SIARa['I']/a_Norm) + alpha*(m11*SIARa['A']/a_Norm)\
          +(m12*SIARb['I']/b_Norm) + alpha*(m12*SIARb['A']/b_Norm)\
          +(m13*SIARc['I']/c_Norm) + alpha*(m13*SIARc['A']/c_Norm))
    ###m11*SIARa['NUM']/a_Norm['NUM']*SIARa['I']/SIARa['NUM']
    #由于总人数相对于平常下降，而造成接触数降低
    EdS2I = -r*EdS
    EdS2A = -(1-r)*EdS
    EdI2R = dt*gamma * SIARkind['I']
    EdA2R = dt*gamma * SIARkind['A']
    
    EdS2I[EdS2I<0]=0
    EdS2A[EdS2A<0]=0
    EdI2R[EdI2R<0]=0
    EdA2R[EdA2R<0]=0
    EdS2I=EdS2I.fillna(0)
    EdS2A=EdS2A.fillna(0)
    EdI2R=EdI2R.fillna(0)
    EdA2R=EdA2R.fillna(0)
    """ dS2I: 5000*1, dS2A: 5000*1, dI2R: 5000*1, dA2R: 5000*1"""
    dS2I = np.random.poisson(EdS2I)
    dS2A = np.random.poisson(EdS2A)
    dI2R = np.random.poisson(EdI2R)
    dA2R = np.random.poisson(EdA2R)
    
    dI2R = np.minimum(dI2R,SIARkind['I'])
    dA2R = np.minimum(dA2R,SIARkind['A'])
    
    """ dS: 5000*1, dI:5000*1, dA: 5000*1, dR: 5000*1 """
    dS = -(dS2I+dS2A)
    
    probplace=(SIARkind['S']+dS)<0
    probS=SIARkind['S'][probplace]
    probS2I=np.rint(r*probS)
    probS2A=probS-probS2I
    dS[probplace]=-probS
    dS2I[probplace]=probS2I
    dS2A[probplace]=probS2A
    
    dI = dS2I-dI2R
    dA = dS2A-dA2R
    dR = dI2R+dA2R
    """ SIARkindnew: 5000*5"""
    #SIARkindnew = pd.DataFrame({'S':SIARkind['S']+np.rint(dS),'I':SIARkind['I']+np.rint(dI),'A':SIARkind['A']+np.rint(dA),'R':SIARkind['R']+np.rint(dR),'NUM':SIARkind['NUM']}) 
    SIARkindnew = pd.DataFrame({'S':SIARkind['S']+dS,'I':SIARkind['I']+dI,'A':SIARkind['A']+dA,'R':SIARkind['R']+dR,'NUM':SIARkind['NUM']})       
      
    return SIARkindnew,dS2I

def SIAR_P_onlyb(b_Norm,SIARkind,kind,susckind,Mij3,beta,gamma,alpha,r,dt):#模拟dt时间段内的感染过程
    #m11 = Mij3[kind,0]
    m12 = Mij3[kind,1]
    #m13 = Mij3[kind,2]
    """ EdS: 5000*1, EdS2I: 5000*1, EdS2A: 5000*1, EdI2R: 5000*1, EdA2R: 5000*1"""    
    EdS = -dt * susckind * beta * SIARkind['S'] * \
          ((m12*SIARkind['I']/b_Norm) + alpha*(m12*SIARkind['A']/b_Norm))
    ###m11*SIARa['NUM']/a_Norm['NUM']*SIARa['I']/SIARa['NUM']
    #由于总人数相对于平常下降，而造成接触数降低
    EdS2I = -r*EdS
    EdS2A = -(1-r)*EdS
    EdI2R = dt*gamma * SIARkind['I']
    EdA2R = dt*gamma * SIARkind['A']
    
    EdS2I[EdS2I<0]=0
    EdS2A[EdS2A<0]=0
    EdI2R[EdI2R<0]=0
    EdA2R[EdA2R<0]=0
    EdS2I=EdS2I.fillna(0)
    EdS2A=EdS2A.fillna(0)
    EdI2R=EdI2R.fillna(0)
    EdA2R=EdA2R.fillna(0)
    """ dS2I: 5000*1, dS2A: 5000*1, dI2R: 5000*1, dA2R: 5000*1"""
    dS2I = np.random.poisson(EdS2I)
    dS2A = np.random.poisson(EdS2A)
    dI2R = np.random.poisson(EdI2R)
    dA2R = np.random.poisson(EdA2R)
    
    dI2R = np.minimum(dI2R,SIARkind['I'])
    dA2R = np.minimum(dA2R,SIARkind['A'])
    
    """ dS: 5000*1, dI:5000*1, dA: 5000*1, dR: 5000*1 """
    dS = -(dS2I+dS2A)
    
    probplace=(SIARkind['S']+dS)<0
    probS=SIARkind['S'][probplace]
    probS2I=np.rint(r*probS)
    probS2A=probS-probS2I
    dS[probplace]=-probS
    dS2I[probplace]=probS2I
    dS2A[probplace]=probS2A
    
    dI = dS2I-dI2R
    dA = dS2A-dA2R
    dR = dI2R+dA2R
    """ SIARkindnew: 5000*5"""
    #SIARkindnew = pd.DataFrame({'S':SIARkind['S']+np.rint(dS),'I':SIARkind['I']+np.rint(dI),'A':SIARkind['A']+np.rint(dA),'R':SIARkind['R']+np.rint(dR),'NUM':SIARkind['NUM']}) 
    SIARkindnew = pd.DataFrame({'S':SIARkind['S']+dS,'I':SIARkind['I']+dI,'A':SIARkind['A']+dA,'R':SIARkind['R']+dR,'NUM':SIARkind['NUM']})       
      
    return SIARkindnew,dS2I

def MultiNom(X,D):
    place=np.where(X>0)[0]
    if len(place)==0:
        X1=X
    else:
        D=D.transpose()
        D = lil_matrix(D)
        Drow=D.rows
        Ddata=D.data
        #start_time=time.time()
        move=np.zeros((5080))
        def _func(k):
            nk=X[k]
            y=Ddata[k]
            if (len(y)>=1):
                z = nran.multinomial(n=nk, pvals=y)
                move[Drow[k]]=move[Drow[k]]+z
            
            # elif (len(y)==1):
            #     move[k]=move[k]+nk
                
            else:
                move[k]=move[k]+nk
            return move
        list(map(_func, place))[0]            
        X1=move
    return X1


def Move5(SIARkind, D) -> pd.DataFrame: #模拟个体在子区域间的移动

    """ Snew: 5800*1, Inew: 5800*1, Anew: 5800*1, Rnew: 5800*1, NUM: 5800*1"""
    Snew=MultiNom(SIARkind['S'],D)
    Inew=MultiNom(SIARkind['I'],D)
    Anew=MultiNom(SIARkind['A'],D)
    Rnew=MultiNom(SIARkind['R'],D)
    NUM=Snew+Inew+Anew+Rnew
    SIARkindnew= pd.DataFrame({'S':Snew,'I':Inew,'A':Anew,'R':Rnew,'NUM':NUM})

    return SIARkindnew

def Confirm_case(NewCases,step,dt,days):#根据每日新增感染人数生成每日新增确诊病例数
    a = 1.85
    Td = 6
    NewConfirmed_hat = np.zeros(days)
    for t in range(len(NewCases)):
        num = int(NewCases[t])
        if num > 0:
            TD=np.random.gamma(shape=a,scale=Td/a,size=num)
            for td in TD:
                tx = int(t*dt+td)
                if tx < days:
                    NewConfirmed_hat[tx] = NewConfirmed_hat[tx]+1
    return NewConfirmed_hat


def Compute_stop(method,SIARa_lockdown,SIARb_lockdown,SIARc_lockdown):
    Ia_stop=sum(SIARa_lockdown['I'])
    Ib_stop=sum(SIARb_lockdown['I'])
    Ic_stop=sum(SIARc_lockdown['I'])
    
    Aa_stop=sum(SIARa_lockdown['A'])
    Ab_stop=sum(SIARb_lockdown['A'])
    Ac_stop=sum(SIARc_lockdown['A'])
    
    Ra_stop=sum(SIARa_lockdown['R'])
    Rb_stop=sum(SIARb_lockdown['R'])
    Rc_stop=sum(SIARc_lockdown['R'])
    if method==1:
        Sa_stop=sum(SIARa_lockdown['S'])
        Sb_stop=sum(SIARb_lockdown['S'])
        Sc_stop=sum(SIARc_lockdown['S'])
        return Sa_stop,Sb_stop,Sc_stop,Ia_stop,Ib_stop,Ic_stop,Aa_stop,Ab_stop,Ac_stop,Ra_stop,Rb_stop,Rc_stop
    else:
        return Ia_stop,Ib_stop,Ic_stop,Aa_stop,Ab_stop,Ac_stop,Ra_stop,Rb_stop,Rc_stop
def lockdown_day_age_I(SIARa,SIARb,SIARc,pl):
    #I=SIARa['I']+SIARb['I']+SIARc['I']
    I=SIARa['I']+SIARb['I']+SIARc['I']+SIARa['A']+SIARb['A']+SIARc['A']
    I_ran=nran.binomial(I,pl)
    I20=I
    #Istandard=np.percentile(I,(100-pl))
    Istandard=1
    index_lockdown=I_ran>=Istandard
            
    SIARb_lockdown=SIARb.copy()
    SIARb_lockdown[~index_lockdown]=0
    SIARb[index_lockdown]=0
            
    return SIARb,SIARb_lockdown,I20,index_lockdown

def lockdown_period_onlyb(pla,index_lockdown,b_stop,a_stop,c_stop,Mij3,beta,gamma,alpha,r,dt,NI,keys,k,mobility_without_zero,NUMb,SIARa,SIARb,SIARc,SIARb_lockdown,Sa_stop,Sb_stop,Sc_stop,resulta,resultb,resultc,B_policy,A_policy,C_policy):
    indexk=keys[(k-71)%168]
    D=mobility_without_zero[indexk]
    b_Norm=NUMb+SIARb['NUM']
    NUMb=MultiNom(NUMb,D)
    #本来打算前往被隔离子区域的，假设暂时在家隔离SIARb_lockdown_unliveable
    SIARb_lockdown_unliveable=SIARb[index_lockdown]
    SIARb[index_lockdown]=0
    #没有被隔离的子区域，有pla比例的人被暂时隔离SAIRb_stop
    SIARb_stop,SIARb_move=split_stop_move(SIARb,pla)        
    #被隔离个体，只发生状态变化，不移动
    SIARa=Stop_IAR(SIARa,gamma,dt)
    SIARb_lockdown=Stop_IAR(SIARb_lockdown,gamma,dt)
    SIARc=Stop_IAR(SIARc,gamma,dt)
    #被暂时隔离个体，只发生状态变化，正常移动
    SIARb_lockdown_unliveable=Stop_IAR(SIARb_lockdown_unliveable,gamma,dt)
    SIARbnew_stop=Stop_IAR(SIARb_stop,gamma,dt)        
    Ia_stop,Ib_stop,Ic_stop,Aa_stop,Ab_stop,Ac_stop,Ra_stop,Rb_stop,Rc_stop=Compute_stop(2,SIARa,SIARb_lockdown,SIARc)        
           
    #自由活动的个体，感染+移动
    SIARbnew_move,NIb = SIAR_P_onlyb(b_Norm,SIARb_move,1,1,Mij3,beta,gamma,alpha,r,dt)        
    NI.append(sum(NIb))
            
    SIARbnew_move[index_lockdown]=SIARb_lockdown_unliveable
    SIARbnew=SIARbnew_move+SIARbnew_stop
    
    SIARb = Move5(SIARbnew,D)
    
    resulta.append((Sa_stop, Ia_stop, Aa_stop, Ra_stop))
    resultb.append((Sb_stop+sum(SIARb['S']), Ib_stop+sum(SIARb['I']), Ab_stop+sum(SIARb['A']), Rb_stop+sum(SIARb['R'])))
    resultc.append((Sc_stop, Ic_stop, Ac_stop, Rc_stop))
            
    B_policy.append(dt*(b_stop+sum(SIARb_stop['NUM'])+sum(SIARb_lockdown_unliveable['NUM'])))
    A_policy.append(dt*a_stop)
    C_policy.append(dt*c_stop)             
    
    return NUMb,SIARa,SIARb,SIARc,SIARb_lockdown,resulta,resultb,resultc,NI,B_policy,A_policy,C_policy

def split_stop_move(SIARkind,pl):
    #SIARkind_stop=(SIARkind*pl+0.5).astype(int)
    SIARkind_stop=np.random.binomial(SIARkind,pl)
    SIARkind_stop=pd.DataFrame(SIARkind_stop,columns=['S','I','A','R','NUM'])
    SIARkind_stop['NUM']=SIARkind_stop['S']+SIARkind_stop['I']+SIARkind_stop['A']+SIARkind_stop['R']
    SIARkind_move=SIARkind-SIARkind_stop
    return SIARkind_stop,SIARkind_move

def Result(pl,pla,SIARa,SIARb,SIARc,beta,gamma,alpha,r,dt,days,step,Mij3_1):#根据给定参数模拟传染病传播过程,得到SIAR随时间变化情况与每日新增确诊病例数
    B_policy=[0]
    A_policy=[0]
    C_policy=[0]
    resulta = [(sum(SIARa['S']), sum(SIARa['I']), sum(SIARa['A']), sum(SIARa['R']))]
    resultb = [(sum(SIARb['S']), sum(SIARb['I']), sum(SIARb['A']), sum(SIARb['R']))]
    resultc = [(sum(SIARc['S']), sum(SIARc['I']), sum(SIARc['A']), sum(SIARc['R']))]

    NI = [2]
    Mij3 = Mij3_1
    for k in range(step-1):
        x = int(k*dt)

        if (x<20):
            SIARanew,NIa = SIAR(SIARa,0,0.34,SIARa,SIARb,SIARc,Mij3,beta,gamma,alpha,r,dt)
            SIARbnew,NIb = SIAR(SIARb,1,1,SIARa,SIARb,SIARc,Mij3,beta,gamma,alpha,r,dt)
            SIARcnew,NIc = SIAR(SIARc,2,1.47,SIARa,SIARb,SIARc,Mij3,beta,gamma,alpha,r,dt)
            NI.append(sum(NIa)+sum(NIb)+sum(NIc))
            indexk=keys[(k-71)%168]
            D=mobility_without_zero[indexk]            
            """ SIARa: 5000*5, SIARb: 5000*5, SIARc: 5000*5 """
            SIARa = Move5(SIARanew,D)
            SIARb = Move5(SIARbnew,D)
            SIARc = Move5(SIARcnew,D)
            B_policy.append(0)
            A_policy.append(0)
            C_policy.append(0)
            resulta.append((sum(SIARa['S']), sum(SIARa['I']), sum(SIARa['A']), sum(SIARa['R'])))
            resultb.append((sum(SIARb['S']), sum(SIARb['I']), sum(SIARb['A']), sum(SIARb['R'])))
            resultc.append((sum(SIARc['S']), sum(SIARc['I']), sum(SIARc['A']), sum(SIARc['R'])))

        if k==(20*24):
            indexk=keys[(k-71)%168]
            D=mobility_without_zero[indexk]
            b_Norm=SIARb['NUM']
            #根据子区域感染情况，隔离14天的b类个体SIARb_lockdown
            SIARb,SIARb_lockdown,I20,index_lockdown=lockdown_day_age_I(SIARa,SIARb,SIARc,pl)
            #被暂时隔离的b类个体SIARb_stop
            SIARb_stop,SIARb_move=split_stop_move(SIARb,pla)
            #a和c类个体，被永久隔离
            #被隔离个体，只发生状态变化，不移动
            SIARa=Stop_IAR(SIARa,gamma,dt)
            SIARb_lockdown=Stop_IAR(SIARb_lockdown,gamma,dt)
            SIARc=Stop_IAR(SIARc,gamma,dt)
            SIARbnew_stop=Stop_IAR(SIARb_stop,gamma,dt)
            #自由活动的个体，感染+移动    
            SIARbnew_move,NIb = SIAR_P_onlyb(b_Norm,SIARb_move,1,1,Mij3,beta,gamma,alpha,r,dt)                     
            NI.append(sum(NIb))
            SIARbnew=SIARbnew_move+SIARbnew_stop
            SIARb = Move5(SIARbnew,D)
            
            #相关指标           
            NUMb=MultiNom(SIARb_lockdown['NUM'],D)            
            
            Sa_stop,Sb_stop,Sc_stop,Ia_stop,Ib_stop,Ic_stop,Aa_stop,Ab_stop,Ac_stop,Ra_stop,Rb_stop,Rc_stop=Compute_stop(1,SIARa,SIARb_lockdown,SIARc)
            b_stop=Sb_stop+Ib_stop+Ab_stop+Rb_stop
            a_stop=Sa_stop+Ia_stop+Aa_stop+Ra_stop
            c_stop=Sc_stop+Ic_stop+Ac_stop+Rc_stop
            
            resulta.append((Sa_stop, Ia_stop, Aa_stop, Ra_stop))
            resultb.append((Sb_stop+sum(SIARb['S']), Ib_stop+sum(SIARb['I']), Ab_stop+sum(SIARb['A']), Rb_stop+sum(SIARb['R'])))
            resultc.append((Sc_stop, Ic_stop, Ac_stop, Rc_stop))
            
            B_policy.append(dt*(b_stop+sum(SIARb_stop['NUM'])))
            A_policy.append(dt*a_stop)
            C_policy.append(dt*c_stop)
                        
        if (k>20*24)&(k<34*24):
            NUMb,SIARa,SIARb,SIARc,SIARb_lockdown,resulta,resultb,resultc,NI,B_policy,A_policy,C_policy=lockdown_period_onlyb(pla,index_lockdown,b_stop,a_stop,c_stop,Mij3,beta,gamma,alpha,r,dt,NI,keys,k,mobility_without_zero,NUMb,SIARa,SIARb,SIARc,SIARb_lockdown,Sa_stop,Sb_stop,Sc_stop,resulta,resultb,resultc,B_policy,A_policy,C_policy)
            
        if (k==34*24):
            indexk=keys[(k-71)%168]
            D=mobility_without_zero[indexk] 
            
            SIARb_lockdown,SIARb=unlock_SR(SIARb,SIARb_lockdown,index_lockdown)
            SIARb,SIARb_lockdown2,I34,index_lockdown2=lockdown_day_age_I(SIARa,SIARb,SIARc,pl)
                       
            SIARb_stop,SIARb_move=split_stop_move(SIARb,pla)
            SIARbnew_stop=Stop_IAR(SIARb_stop,gamma,dt)
            
            SIARb_lockdown=SIARb_lockdown+SIARb_lockdown2
            #被隔离个体，只发生状态变化，不移动
            SIARa=Stop_IAR(SIARa,gamma,dt)
            SIARb_lockdown=Stop_IAR(SIARb_lockdown,gamma,dt)
            SIARc=Stop_IAR(SIARc,gamma,dt)
                        
            b_Norm=SIARb_lockdown['NUM']+SIARb['NUM']
            
            #自由活动的个体，感染+移动
            SIARbnew_move,NIb = SIAR_P_onlyb(b_Norm,SIARb_move,1,1,Mij3,beta,gamma,alpha,r,dt)
            NI.append(sum(NIb))
            SIARbnew=SIARbnew_move+SIARbnew_stop
            SIARb = Move5(SIARbnew,D)
            #相关指标
            NUMb=MultiNom(SIARb_lockdown['NUM'],D)
            Sa_stop,Sb_stop,Sc_stop,Ia_stop,Ib_stop,Ic_stop,Aa_stop,Ab_stop,Ac_stop,Ra_stop,Rb_stop,Rc_stop=Compute_stop(1,SIARa,SIARb_lockdown,SIARc)
            b_stop=Sb_stop+Ib_stop+Ab_stop+Rb_stop
            
            resulta.append((Sa_stop, Ia_stop, Aa_stop, Ra_stop))
            resultb.append((Sb_stop+sum(SIARb['S']), Ib_stop+sum(SIARb['I']), Ab_stop+sum(SIARb['A']), Rb_stop+sum(SIARb['R'])))
            resultc.append((Sc_stop, Ic_stop, Ac_stop, Rc_stop))
            
            B_policy.append(dt*(b_stop+sum(SIARb_stop['NUM'])))
            A_policy.append(dt*a_stop)
            C_policy.append(dt*c_stop)
            
        if (k>34*24)&(k<48*24):
            NUMb,SIARa,SIARb,SIARc,SIARb_lockdown,resulta,resultb,resultc,NI,B_policy,A_policy,C_policy=lockdown_period_onlyb(pla,index_lockdown2,b_stop,a_stop,c_stop,Mij3,beta,gamma,alpha,r,dt,NI,keys,k,mobility_without_zero,NUMb,SIARa,SIARb,SIARc,SIARb_lockdown,Sa_stop,Sb_stop,Sc_stop,resulta,resultb,resultc,B_policy,A_policy,C_policy)
            
        if (k==48*24):            
            
            indexk=keys[(k-71)%168]
            D=mobility_without_zero[indexk] 
            index_lockdown=index_lockdown|index_lockdown2
            
            SIARb_lockdown,SIARb=unlock_SR(SIARb,SIARb_lockdown,index_lockdown)
            
            SIARb,SIARb_lockdown3,I48,index_lockdown3=lockdown_day_age_I(SIARa,SIARb,SIARc,pl)
            
            SIARb_stop,SIARb_move=split_stop_move(SIARb,pla)
            SIARbnew_stop=Stop_IAR(SIARb_stop,gamma,dt)
            
            SIARb_lockdown=SIARb_lockdown+SIARb_lockdown3
            #被隔离个体，只发生状态变化，不移动
            SIARa=Stop_IAR(SIARa,gamma,dt)
            SIARb_lockdown=Stop_IAR(SIARb_lockdown,gamma,dt)
            SIARc=Stop_IAR(SIARc,gamma,dt)
            
            b_Norm=SIARb_lockdown['NUM']+SIARb['NUM']
            
            #自由活动的个体，感染+移动
            SIARbnew_move,NIb = SIAR_P_onlyb(b_Norm,SIARb_move,1,1,Mij3,beta,gamma,alpha,r,dt)                     
            NI.append(sum(NIb))
            SIARbnew=SIARbnew_move+SIARbnew_stop
            SIARb = Move5(SIARbnew,D)
            #相关指标
            NUMb=MultiNom(SIARb_lockdown['NUM'],D)
            Sa_stop,Sb_stop,Sc_stop,Ia_stop,Ib_stop,Ic_stop,Aa_stop,Ab_stop,Ac_stop,Ra_stop,Rb_stop,Rc_stop=Compute_stop(1,SIARa,SIARb_lockdown,SIARc)
            b_stop=Sb_stop+Ib_stop+Ab_stop+Rb_stop
            
            resulta.append((Sa_stop, Ia_stop, Aa_stop, Ra_stop))
            resultb.append((Sb_stop+sum(SIARb['S']), Ib_stop+sum(SIARb['I']), Ab_stop+sum(SIARb['A']), Rb_stop+sum(SIARb['R'])))
            resultc.append((Sc_stop, Ic_stop, Ac_stop, Rc_stop))
            
            B_policy.append(dt*(b_stop+sum(SIARb_stop['NUM'])))
            A_policy.append(dt*a_stop)
            C_policy.append(dt*c_stop)
        if (k>48*24):
            NUMb,SIARa,SIARb,SIARc,SIARb_lockdown,resulta,resultb,resultc,NI,B_policy,A_policy,C_policy=lockdown_period_onlyb(pla,index_lockdown3,b_stop,a_stop,c_stop,Mij3,beta,gamma,alpha,r,dt,NI,keys,k,mobility_without_zero,NUMb,SIARa,SIARb,SIARc,SIARb_lockdown,Sa_stop,Sb_stop,Sc_stop,resulta,resultb,resultc,B_policy,A_policy,C_policy)
            
    resulta = np.array(resulta)
    resultb = np.array(resultb)
    resultc = np.array(resultc)

    NewConfirmed_hat = Confirm_case(NI,step,dt,days)
        
    return resulta,resultb,resultc,NewConfirmed_hat,B_policy,A_policy,C_policy,I20,I34,I48

def unlock_SR(SIARa,SIARa_lockdown,index_lockdown):
    SIARa['S'][index_lockdown]=SIARa['S'][index_lockdown]+SIARa_lockdown['S']
    SIARa['R'][index_lockdown]=SIARa['R'][index_lockdown]+SIARa_lockdown['R']
    SIARa_lockdown['S']=0
    SIARa_lockdown['R']=0
    SIARa_lockdown['NUM']=SIARa_lockdown['I']+SIARa_lockdown['A']
    SIARa['NUM']=SIARa['S']+SIARa['I']+SIARa['A']+SIARa['R']
    return SIARa_lockdown,SIARa
#%%Initailize
""" area: 20m*2 """
area = pd.read_csv('area_SH_using_mobility1.csv')#初始时刻个体在每个子区域的分布
""" age: 20m*1 """
age = pd.read_csv('ageSH.csv')
N = len(age)
""" susc: 20m*1 """
susc = Year_structure(age)
numI = 2 #初始感染者数量
M = 5080 #子区域数量
dt = 1/24
days = 60
step = int(days/dt)
NC = pd.read_csv('NewCases.csv')
NewConfirmed = NC['NewCase'][0:days]
Age_struc = pd.read_csv('Age_struc2016.csv')
Mij14_1 = pd.read_csv('Mij1.csv')#接触矩阵
Mij3_1 = RMij(Mij14_1,Age_struc)


#Rate=Rate[:11]
f=open('0525mobility_frac_matrix_max_nozero.pkl','rb')
mobility_without_zero=pk.load(f)
keys = list(mobility_without_zero.keys())

filepath='2multi_Best_para_rmse_5per.csv'
best_para=pd.read_csv(open(filepath,'rb'))

def Initial_Status(N,numI,r,area,age,susc):#生成0时刻,每个子区域内, a(b,c)年龄段个体的S,I,A,R的数量和总数
    """ Status: 20m * 1 """
    """ Info: 20m * 5 """ 
    Status = patient_zero(N,numI,r)
    Info = pd.DataFrame({'Agent':area['agent'],'Status':Status,'Age':age['age'],'Susc':susc,'Area0':area['period0']}) # Info -> 20m * 5
    SIARa = Info2SIAR(Info,0.34)  
    SIARb = Info2SIAR(Info,1)
    SIARc = Info2SIAR(Info,1.47)
    return SIARa,SIARb,SIARc

#%%主程序

def main(best_para,dt,days,step,Mij3_1,N,numI,area,age,susc,pool):
    
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Xunhuan = 30
    alpha=0.55
    for i in range(len(best_para)):
        pl=best_para['PL'].iloc[i]
        pla=best_para['PLA'].iloc[i]
        r=best_para['Rate'].iloc[i]
        """ SIARa: 5000*5, SIARb: 5000*5,SIARc: 5000*5 """
        SIARa0,SIARb0,SIARc0 = Initial_Status(N,numI,r,area,age,susc)#生成初始分布

        beta=best_para['Beta'].iloc[i]
        gamma=best_para['Gamma'].iloc[i]
        g=best_para['index'].iloc[i]

        pool.apply_async(cal_para, args=(pl,pla,r, SIARa0, SIARb0, SIARc0, g, alpha, beta, gamma, Xunhuan, i))

    return 1

def cal_para(pl,pla,r, SIARa0, SIARb0, SIARc0, g, alpha, beta, gamma, Xunhuan, i):
    print('Para %s:...' % str(i))
    Ave_resulta = np.zeros([step,4])
    Ave_resultb = np.zeros([step,4])
    Ave_resultc = np.zeros([step,4])
    Ave_B_policy = np.zeros(step)
    Ave_A_policy = np.zeros(step)
    Ave_C_policy = np.zeros(step)
    Ave_NewConfirmed_hat = np.zeros(days)
    ResultABC={}
    ResultA={}
    ResultB={}
    ResultC={}
    NewConfirmed_HAT={}
    B_POLICY={}
    A_POLICY={}
    C_POLICY={}
    I20th={}
    I34th={}
    I48th={}
    #根据给定参数模拟传染病传播过程
    for xunhuan in range(Xunhuan):
        resulta,resultb,resultc,NewConfirmed_hat,B_policy,A_policy,C_policy,I20,I34,I48= Result(pl,pla,SIARa0,SIARb0,SIARc0,beta,gamma,alpha,r,dt,days,step,Mij3_1)
        Ave_resulta = Ave_resulta + resulta
        Ave_resultb = Ave_resultb + resultb
        Ave_resultc = Ave_resultc + resultc
        Ave_NewConfirmed_hat = Ave_NewConfirmed_hat + NewConfirmed_hat
        result=resulta+resultb+resultc
        Ave_B_policy=Ave_B_policy+B_policy
        Ave_A_policy=Ave_A_policy+A_policy
        Ave_C_policy=Ave_C_policy+C_policy
            
        ResultABC[xunhuan]=result
        ResultA[xunhuan]=resulta
        ResultB[xunhuan]=resultb
        ResultC[xunhuan]=resultc
        NewConfirmed_HAT[xunhuan]=NewConfirmed_hat
        B_POLICY[xunhuan]=B_policy
        A_POLICY[xunhuan]=A_policy
        C_POLICY[xunhuan]=C_policy
        I20th[xunhuan]=I20
        I34th[xunhuan]=I34
        I48th[xunhuan]=I48
       # subname='New20d_lockdown'+str(int(pl*100))
    subname='multi_combine_PL'+str(int(pl*100))+'_PLA_'+str(int(pla*100))
    with open("".join([str(g), 'NewConfirmed_hat_',subname, '.pkl']), 'wb') as handle:
        pk.dump(NewConfirmed_HAT, handle)
        
    with open("".join([str(g), 'SIAR_',subname, '.pkl']), 'wb') as handle:
        pk.dump(ResultABC, handle)
    with open("".join([str(g), 'SIARa_',subname, '.pkl']), 'wb') as handle:
        pk.dump(ResultA, handle)
    with open("".join([str(g), 'SIARb_',subname, '.pkl']), 'wb') as handle:
        pk.dump(ResultB, handle)
    with open("".join([str(g), 'SIARc_',subname, '.pkl']), 'wb') as handle:
        pk.dump(ResultC, handle)
    with open(str(g)+'Bpolicy_'+subname+'.pkl', 'wb') as handle:
        pk.dump(B_POLICY, handle)
    with open(str(g)+'Apolicy_'+subname+'.pkl', 'wb') as handle:
        pk.dump(A_POLICY, handle)
    with open(str(g)+'Cpolicy_'+subname+'.pkl', 'wb') as handle:
        pk.dump(C_POLICY, handle)
    with open(str(g)+'I20_'+subname+'.pkl', 'wb') as handle:
        pk.dump(I20th, handle)
    with open(str(g)+'I34_'+subname+'.pkl', 'wb') as handle:
        pk.dump(I34th, handle)
    with open(str(g)+'I48_'+subname+'.pkl', 'wb') as handle:
        pk.dump(I48th, handle)
    Ave_resulta = Ave_resulta/Xunhuan
    Ave_resultb = Ave_resultb/Xunhuan
    Ave_resultc = Ave_resultc/Xunhuan
    Ave_result = Ave_resulta + Ave_resultb + Ave_resultc
    Ave_NewConfirmed_hat = Ave_NewConfirmed_hat/Xunhuan
    Ave_B_policy = Ave_B_policy/Xunhuan
    Ave_A_policy = Ave_A_policy/Xunhuan
    Ave_C_policy = Ave_C_policy/Xunhuan
    
    Para = pd.DataFrame({'Beta':[beta],'Gamma':[gamma],'Alpha':[alpha],'Rate':[r]})
    
    Para.to_csv("".join([str(g), 'Para_hat_',subname,'.tsv']),encoding='utf_8_sig')
    Ave_resulta=pd.DataFrame(Ave_resulta)
    Ave_resulta.to_csv("".join([str(g), 'SIARa_',subname, '.tsv']),encoding='utf_8_sig')
    Ave_resultb=pd.DataFrame(Ave_resultb)
    Ave_resultb.to_csv("".join([str(g), 'SIARb_',subname, '.tsv']),encoding='utf_8_sig')
    Ave_resultc=pd.DataFrame(Ave_resultc)
    Ave_resultc.to_csv("".join([str(g), 'SIARc_',subname, '.tsv']),encoding='utf_8_sig')
    Ave_result=pd.DataFrame(Ave_result)
    Ave_result.to_csv("".join([str(g), 'SIAR_',subname, '.tsv']),encoding='utf_8_sig')
    Ave_NewConfirmed_hat=pd.DataFrame(Ave_NewConfirmed_hat)
    Ave_NewConfirmed_hat.to_csv("".join([str(g), 'NewConfirmed_hat_',subname, '.tsv']),encoding='utf_8_sig') 
    Ave_B_policy=pd.DataFrame(Ave_B_policy)
    Ave_B_policy.to_csv(str(g)+'Bpolicy_'+subname+'.tsv',encoding='utf_8_sig')
    Ave_A_policy=pd.DataFrame(Ave_A_policy)
    Ave_A_policy.to_csv(str(g)+'Apolicy_'+subname+'.tsv',encoding='utf_8_sig') 
    Ave_C_policy=pd.DataFrame(Ave_C_policy)
    Ave_C_policy.to_csv(str(g)+'Cpolicy_'+subname+'.tsv',encoding='utf_8_sig') 
    
    print('Para %s done. '%str(i), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

PL=[0,0.05,0.15]
PLA=[0,0.02,0.05,0.1,0.2,0.4,0.6,0.8]
best_para=best_para[best_para['index']==7636]
best_para['PLA']=PLA[0]
best_para1=best_para.copy()
for i in range(1,len(PLA)):
    best_para1['PLA']=PLA[i]
    best_para=best_para.append(best_para1)
best_para['PL']=PL[0]
best_para1=best_para.copy()
for i in range(1,len(PL)):
    best_para1['PL']=PL[i]
    best_para=best_para.append(best_para1)
if __name__ == '__main__':
    
    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    main(best_para,dt,days,step,Mij3_1,N,numI,area,age,susc,pool)
    pool.close()
    pool.join()
    end = time.time()
    print(end-start)