# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:35:21 2022

@author: Luoji
"""

import pandas as pd
import numpy as np
import random

#%%随机生成人群年龄分布和地区分布                                
def Population(N):
    
    data=pd.read_csv(open('C:\\Users\\LJY\\Desktop\\论文\\传染病模型文献\\传染病模型程序\\传染病模型参数估计\\Age_struc2016.csv','rb'))
    AGE=[]
    alpha1=data['P'][0]
    alpha2=alpha1+data['P'][1]
    alpha3=alpha2+data['P'][2]
    alpha4=alpha3+data['P'][3]
    alpha5=alpha4+data['P'][4]
    alpha6=alpha5+data['P'][5]
    alpha7=alpha6+data['P'][6]
    for i in range(N):
        p=random.random()
        if p<=alpha1:
            agei=random.randint(1,10)
        if (p>alpha1)&(p<=alpha2):
            agei=random.randint(10,20)
        if (p>alpha2)&(p<=alpha3):
            agei=random.randint(20,30)
        if (p>alpha3)&(p<=alpha4):
            agei=random.randint(30,40)
        if (p>alpha4)&(p<=alpha5):
            agei=random.randint(40,50)
        if (p>alpha5)&(p<=alpha6):
            agei=random.randint(50,60)
        if (p>alpha6)&(p<=alpha7):
            agei=random.randint(60,70)
        if (p>alpha7):
            agei=random.randint(70,100)
        AGE.append(agei)
    AGE=pd.DataFrame({'age':AGE})
    AGE.to_csv('ageSH.csv',encoding='utf_8_sig')
Population(24281400)                 

#根据移动数据生成初始时刻个体在每个子区域的分布
data2 = pd.read_csv("Initial_distribution_in_5079_region.csv")
#data2 = pd.read_csv("Initial_distribution_in_355_region.csv")
data2=data2.drop(index=(data2.loc[(data2['agent_id5']==0)].index))
count=data2['count']
weights = [a/sum(count) for a in count]
agent_id=data2['agent_id5']
indexes = np.random.choice(agent_id, 24281400, p=weights)
agent=list(range(24281400))
area_SH=pd.DataFrame({'agent':agent,'period0':indexes})
area_SH.to_csv('area_SH_using_mobility1.csv',encoding='utf_8_sig')

