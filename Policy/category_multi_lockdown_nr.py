# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 19:08:27 2021

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

def SIAR_PL(nPL,SIARkind,kind,susckind,SIARa,SIARb,SIARc,Mij3,beta,gamma,alpha,r,dt):#模拟dt时间段内的感染过程
    m11 = Mij3[kind,0]
    m12 = Mij3[kind,1]
    m13 = Mij3[kind,2]

    """ EdS: 5000*1, EdS2I: 5000*1, EdS2A: 5000*1, EdI2R: 5000*1, EdA2R: 5000*1"""
    EdS = -dt * susckind * beta * SIARkind['S'] * \
          ((m11*nPL*SIARa['I']/SIARa['NUM']) + alpha*(m11*nPL*SIARa['A']/SIARa['NUM'])\
          +(m12*nPL*SIARb['I']/SIARb['NUM']) + alpha*(m12*nPL*SIARb['A']/SIARb['NUM'])\
          +(m13*nPL*SIARc['I']/SIARc['NUM']) + alpha*(m13*nPL*SIARc['A']/SIARc['NUM']))
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

    """ Snew: 5000*1, Inew: 5000*1, Anew: 5000*1, Rnew: 5000*1, NUM: 5000*1"""
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

def split_stop_move(SIARkind,pl):
    #SIARkind_stop=(SIARkind*pl+0.5).astype(int)
    SIARkind_stop=np.random.binomial(SIARkind,pl)
    SIARkind_stop=pd.DataFrame(SIARkind_stop,columns=['S','I','A','R','NUM'])
    SIARkind_stop['NUM']=SIARkind_stop['S']+SIARkind_stop['I']+SIARkind_stop['A']+SIARkind_stop['R']
    SIARkind_move=SIARkind-SIARkind_stop
    return SIARkind_stop,SIARkind_move

def Result_no_lockdown(category,region_portion,SIARa,SIARb,SIARc,beta,gamma,alpha,r,dt,days,step,Mij3_1):#根据给定参数模拟传染病传播过程,得到SIAR随时间变化情况与每日新增确诊病例数
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

            #例迁移矩阵(M*M)
            #d11,d21,d31
            #d12,d22,d32
            #d13,d23,d33
            """ SIARa: 170000*5, SIARb: 170000*5, SIARc: 170000*5 """
            SIARa = Move5(SIARanew,D)
            SIARb = Move5(SIARbnew,D)
            SIARc = Move5(SIARcnew,D)
            B_policy.append(0)
            A_policy.append(0)
            C_policy.append(0)
        else:
            indexk=keys[(k-71)%168]
            PL=region_portion[indexk][:,category]
            SIARa_stop,SIARa_move=split_stop_move(SIARa,PL)
            SIARb_stop,SIARb_move=split_stop_move(SIARb,PL)
            SIARc_stop,SIARc_move=split_stop_move(SIARc,PL)
            
            SIARanew_stop=Stop_IAR(SIARa_stop,gamma,dt)
            SIARbnew_stop=Stop_IAR(SIARb_stop,gamma,dt)
            SIARcnew_stop=Stop_IAR(SIARc_stop,gamma,dt)
            
            nPL=np.array(1-PL).flatten()
            SIARanew_move,NIa = SIAR_PL(nPL,SIARa_move,0,0.34,SIARa_move,SIARb_move,SIARc_move,Mij3,beta,gamma,alpha,r,dt)
            SIARbnew_move,NIb = SIAR_PL(nPL,SIARb_move,1,1,SIARa_move,SIARb_move,SIARc_move,Mij3,beta,gamma,alpha,r,dt)
            SIARcnew_move,NIc = SIAR_PL(nPL,SIARc_move,2,1.47,SIARa_move,SIARb_move,SIARc_move,Mij3,beta,gamma,alpha,r,dt)
            NI.append(sum(NIa)+sum(NIb)+sum(NIc))
            indexk=keys[(k-71)%168]
            D=mobility_without_zero[indexk]
            
            SIARanew=SIARanew_move+SIARanew_stop
            SIARbnew=SIARbnew_move+SIARbnew_stop
            SIARcnew=SIARcnew_move+SIARcnew_stop
            
            SIARa = Move5(SIARanew,D)
            SIARb = Move5(SIARbnew,D)
            SIARc = Move5(SIARcnew,D)
            B_policy.append(dt*sum(SIARb_stop['NUM']))
            A_policy.append(dt*sum(SIARa_stop['NUM']))
            C_policy.append(dt*sum(SIARc_stop['NUM']))
            
        resulta.append((sum(SIARa['S']), sum(SIARa['I']), sum(SIARa['A']), sum(SIARa['R'])))
        resultb.append((sum(SIARb['S']), sum(SIARb['I']), sum(SIARb['A']), sum(SIARb['R'])))
        resultc.append((sum(SIARc['S']), sum(SIARc['I']), sum(SIARc['A']), sum(SIARc['R'])))

    resulta = np.array(resulta)
    resultb = np.array(resultb)
    resultc = np.array(resultc)

    NewConfirmed_hat = Confirm_case(NI,step,dt,days)
        
    return resulta,resultb,resultc,NewConfirmed_hat,B_policy,A_policy,C_policy
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

f=open('1003region_type_portion_nozero.pkl','rb')
region_portion=pk.load(f)

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

def main(region_portion,best_para,dt,days,step,Mij3_1,N,numI,area,age,susc,pool):
  
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Xunhuan = 30
    alpha=0.55
    for i in range(len(best_para)):
        r=best_para['Rate'].iloc[i]
        """ SIARa: 5000*5, SIARb: 5000*5,SIARc: 5000*5 """
        SIARa0,SIARb0,SIARc0 = Initial_Status(N,numI,r,area,age,susc)#生成初始分布
        category=best_para['PL'].iloc[i]
        beta=best_para['Beta'].iloc[i]
        gamma=best_para['Gamma'].iloc[i]
        g=best_para['index'].iloc[i]
        pool.apply_async(cal_para, args=(category,region_portion,r, SIARa0, SIARb0, SIARc0, g, alpha, beta, gamma, Xunhuan, i))

    return 1

def cal_para(category,region_portion,r, SIARa0, SIARb0, SIARc0, g, alpha, beta, gamma, Xunhuan, i):
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
    #根据给定参数模拟传染病传播过程
    for xunhuan in range(Xunhuan):
        resulta,resultb,resultc,NewConfirmed_hat,B_policy,A_policy,C_policy= Result_no_lockdown(category,region_portion,SIARa0,SIARb0,SIARc0,beta,gamma,alpha,r,dt,days,step,Mij3_1)
        
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
       # subname='New20d_lockdown'+str(int(pl*100))
    subname='Nbi_multi_category'+str(int(category))
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


Category=[2,4,5,6,7]
best_para['PL']=Category[0]
best_para1=best_para.copy()
for i in range(1,len(Category)):
    best_para1['PL']=Category[i]
    best_para=best_para.append(best_para1)
    
if __name__ == '__main__':
    
    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    main(region_portion,best_para,dt,days,step,Mij3_1,N,numI,area,age,susc,pool)
    pool.close()
    pool.join()
    end = time.time()
    print(end-start)