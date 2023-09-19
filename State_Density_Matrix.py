import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from collections import Counter

class State_Density_Matrix():
    def __init__(self):        # n个qubit N个密度矩阵
        #抛利算符和单位算符
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,0-1j],[0+1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
        self.E = np.array([[1,0],[0,1]])
        #tomography里除了单位算符的15个算符，有泡利算符和单位算符直积np.kron()而成
        self.EX=np.kron(self.E,self.X)
        self.EY=np.kron(self.E,self.Y)
        self.EZ=np.kron(self.E,self.Z)
        self.XE=np.kron(self.X,self.E)
        self.XX=np.kron(self.X,self.X)
        self.XY=np.kron(self.X,self.Y)
        self.XZ=np.kron(self.X,self.Z)
        self.YE=np.kron(self.Y,self.E)
        self.YX=np.kron(self.Y,self.X)
        self.YY=np.kron(self.Y,self.Y)
        self.YZ=np.kron(self.Y,self.Z)
        self.ZE=np.kron(self.Z,self.E)
        self.ZX=np.kron(self.Z,self.X)
        self.ZY=np.kron(self.Z,self.Y)
        self.ZZ=np.kron(self.Z,self.Z)
        self.SIGMA=None
        self.PPT=None
        self.LOCAL=None
        self.PRO=None
    
    def realize_state(self,Save=None):
        self.SIGMA=np.real(self.SIGMA)
        if Save!=None:
            if not os.path.exists('Random_Qubits'):
                os.makedirs('Random_Qubits')
            if not os.path.exists('Random_Qubits/density_matrix'):
                os.makedirs('Random_Qubits/density_matrix')
            
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'_realized_'+Save,densitymatrix=self.SIGMA)
        return self.SIGMA
    
    def random_purestates(self,n,N,Save=None):
        SIGMA=[]
        para=np.random.uniform(-1,1,(2**n,1))+1J*np.random.uniform(-1,1,(2**n,1))
        C=np.real(np.dot(para.T.conjugate(),para))
        tai=np.dot(para,para.T.conjugate())/C
        for i in range(N):
            para=np.random.uniform(-1,1,(2**n,1))+1J*np.random.uniform(-1,1,(2**n,1))
            sigma=np.dot(para,para.T.conjugate())
            SIGMA.append(sigma)
        self.SIGMA=np.array(SIGMA)

        if Save!=None:
            if not os.path.exists('Random_Qubits'):
                os.makedirs('Random_Qubits')
            if not os.path.exists('Random_Qubits/density_matrix'):
                os.makedirs('Random_Qubits/density_matrix')
            #np.savetxt('density_matrix/data/'+'twoqubitsdensitymatrix'+str(N)+'.txt',SIGMA) #会把SIGMA从列表变成数组
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'__'+Save,densitymatrix=self.SIGMA)
        return self.SIGMA

    def werner(self,N,Save=None):
        SIGMA=[]
        ENT=[]
        self.N=N
        P=np.random.uniform(0,1,self.N)
        vec=np.sqrt(1/2)*np.array([[1,0,0,0]])+np.sqrt(1/2)*np.array([[0,0,0,1]])
        for p in P:
            #rou_B=0.5*np.array([[1,0],[0,0]])+0.5*np.array([[0,0],[0,1]])
            SIGMA.append(p*np.dot(vec.T,vec.conjugate())+0.25*(1-p)*np.identity(4))
            if p<=1/3:
                ENT.append(1)  # 1可分 0纠缠
            else:
                ENT.append(0)
        self.SIGMA=np.array(SIGMA)
        ENT=np.array(ENT)

        if Save!=None:
            if not os.path.exists('Random_Qubits'):
                os.makedirs('Random_Qubits')
            if not os.path.exists('Random_Qubits/density_matrix'):
                os.makedirs('Random_Qubits/density_matrix')
            #np.savetxt('density_matrix/data/'+'twoqubitsdensitymatrix'+str(N)+'.txt',SIGMA) #会把SIGMA从列表变成数组
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'__'+Save,densitymatrix=self.SIGMA)
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'Auto_ent_lab_of_'+Save,ENT=ENT)
        return self.SIGMA,ENT
    
    def generalized_werner_2(self,N,Save=None):#Save写入保存的名称 字符串格式
        SIGMA=[]
        lab=[]
        Ev_min=[]
        self.N=N
        Sita=np.random.uniform(0,np.pi,self.N)
        Fai=np.random.uniform(0,2*np.pi,self.N)
        P=np.random.uniform(0,1,self.N)
        for sita,fai,p in zip(Sita,Fai,P):
            fai=complex(str(fai)+'j')
            vec=np.cos(0.5*sita)*np.array([[1,0,0,0]])+np.exp(fai)*np.sin(0.5*sita)*np.array([[0,0,0,1]])
            SIGMA.append(p*np.dot(vec.T,vec.conjugate())+0.25*(1-p)*np.identity(4))
        self.SIGMA=np.array(SIGMA)
        
        if Save!=None:
            if not os.path.exists('Random_Qubits'):
                os.makedirs('Random_Qubits')
            if not os.path.exists('Random_Qubits/density_matrix'):
                os.makedirs('Random_Qubits/density_matrix')
            #np.savetxt('density_matrix/data/'+'twoqubitsdensitymatrix'+str(N)+'.txt',SIGMA) #会把SIGMA从列表变成数组
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'__'+Save,densitymatrix=self.SIGMA)  #savetxt只能保存1D和2D  3D以上会保存 要用savez 
        return self.SIGMA        
        
    def generalized_werner_1(self,N,Save=None):#Save写入保存的名称 字符串格式
        SIGMA=[]
        ENT=[]
        STEER=[]
        LOCAL=[]
        
        self.N=N
        Sita=np.random.uniform(0,2*np.pi,self.N)
        P=np.random.uniform(0,1,self.N)
        for sita,p in zip(Sita,P):
            vec=np.cos(sita)*np.array([[1,0,0,0]])+np.sin(sita)*np.array([[0,0,0,1]])
            rou_B=(np.cos(sita)**2)*np.array([[1,0],[0,0]])+(np.sin(sita)**2)*np.array([[0,0],[0,1]])
            SIGMA.append(p*np.dot(vec.T,vec.conjugate())+0.5*(1-p)*np.kron(np.identity(2),rou_B))
            if p<=1/3:
                ENT.append(1)#可分
            else:
                ENT.append(0)#纠缠
            if p>np.sqrt(0.5) and p<=(1/np.sqrt(1+np.sin(sita)**2)):
                STEER.append(0)#可导引
            else:
                STEER.append(1)#不可导引
            if p>(1/np.sqrt(1+np.sin(sita)**2)):
                LOCAL.append(0) #非定域的
            else:
                LOCAL.append(1)  #定域的
        self.SIGMA=np.array(SIGMA)
        ENT=np.array(ENT)
        STEER=np.array(STEER)
        LOCAL=np.array(LOCAL)

        if Save!=None:
            if not os.path.exists('Random_Qubits'):
                os.makedirs('Random_Qubits')
            if not os.path.exists('Random_Qubits/density_matrix'):
                os.makedirs('Random_Qubits/density_matrix')
            #np.savetxt('density_matrix/data/'+'twoqubitsdensitymatrix'+str(N)+'.txt',SIGMA) #会把SIGMA从列表变成数组
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'_'+Save,densitymatrix=self.SIGMA)  #savetxt只能保存1D和2D  3D以上会保存 要用savez 
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'Auto_ent_lab_of_'+Save,ENT=ENT)
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'Auto_ste_lab_of_'+Save,STEER=STEER)
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'Auto_loc_lab_of_'+Save,LOCAL=LOCAL)
        return self.SIGMA,ENT,STEER,LOCAL  
        
    def random_qubits(self,n,N,Save=None):#N为密度矩阵数量 same_prop对应True或False
        SIGMA=[]
        self.N=N
        for i in range(N):
            mat=np.random.uniform(-6,6,(2**n,2**n))+np.random.uniform(-6,6,(2**n,2**n))*1j
            mat_conj=(mat.T).conjugate()
            mat_em=np.dot(mat,mat_conj)
            sigma=mat_em/np.trace(mat_em)
            SIGMA.append(sigma)
        SIGMA=np.array(SIGMA)
        self.SIGMA=SIGMA
            
        if Save!=None:
            if not os.path.exists('Random_Qubits'):
                os.makedirs('Random_Qubits')
            if not os.path.exists('Random_Qubits/density_matrix'):
                os.makedirs('Random_Qubits/density_matrix')
            #np.savetxt('density_matrix/data/'+'twoqubitsdensitymatrix'+str(N)+'.txt',SIGMA) #会把SIGMA从列表变成数组
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'__'+str(n)+'_'+Save,densitymatrix=self.SIGMA)  #savetxt只能保存1D和2D  3D以上会保存 要用savez 
        return self.SIGMA      #SIGMA变成npz里字符串‘SIGMA’对应的对象
    
    def PPT_criterion(self,Save=None):# 这里必须是2X2密度矩阵        
        ppt=[]
        min_ev=[]
        for tai in self.SIGMA:
            tai_bt=copy.copy(tai)
            hang=0
            while hang<=2:     #B部分部分转置
                lie=1
                while lie<=3:
                    c=tai_bt[hang,lie]
                    tai_bt[hang,lie]=tai_bt[hang+1,lie-1]
                    tai_bt[hang+1,lie-1]=c
                    lie+=2
                hang+=2
            #求最小本征值
            eig_tai_bt=np.linalg.eig(tai_bt)#这个数组0是本征值 1是本征态
            ev_tai_bt=np.real(eig_tai_bt[0])#避免极小虚部 所以只要实部
            # 1代表可分 0代表纠缠
            min_ev.append(np.min(ev_tai_bt))
            if np.min(ev_tai_bt)>=0:
                ppt.append(1)
            else:
                ppt.append(0)
        ppt=np.array(ppt)
        min_ev=np.array(min_ev)
        PPT=[ppt,min_ev]   #标签和最小本征值的列表在放到一个列表
        self.PPT=PPT
        if Save!=None:
            np.savez('Random_Qubits/density_matrix/'+'entangle_labels_of_'+str(self.N)+'__'+Save,PPT=PPT)
        return self.PPT
    
    def chsh_criterion(self,Save=None):# 这里必须是2X2密度矩阵  
        chsh=(self.ZX-self.ZZ-self.ZX-self.ZZ+self.XX-self.XZ+self.XX+self.XZ)/np.sqrt(2)
        #print(chsh)
        out_value=[]
        locality=[]
        for tai in self.SIGMA:
            out=np.real(np.trace(np.dot(chsh,tai)))
            out_value.append(out)
            if out>=-2:
                locality.append(1) 
            else:
                locality.append(0) # 0代表是非定域的 1代表是定域的
        out_value=np.array(out_value)
        locality=np.array(locality)
        LOCAL=[locality,out_value]
        self.LOCAL=LOCAL
        if Save!=None:
            np.savez('Random_Qubits/density_matrix/'+'locality_labels_of_'+str(self.N)+'__'+Save,LOCAL=self.LOCAL)
        return self.LOCAL
    
    def prop_of_ent(self,Plot=True,Save=None):#得出纠缠态占比#sigma需要2X2密度矩阵列表或数组
        #PPT=random_qubits.PPT_criterion(sigma,False)
        pro_ent=Counter(self.PPT[0])[0]   #Counter(self.PPT[0])[0] self.PPT[0]里元素0的数量 也就是纠缠数量
        pro_sep=Counter(self.PPT[0])[1]
        self.PRO=[pro_ent,pro_sep]
        #min_ev=np.linspace(-1,1,1001)#用于划分区间的
        mev_axis=(np.arange(1000)*2-1000)*0.001+0.001#真正用来画图的x轴
        self.ent_axis=np.zeros(500)  #y轴--ent
        self.spa_axis=np.zeros(500)  #y轴--spa
        for j in self.PPT[1]:#部分转置最小本征值列表
            if j<0:
                for i in range(500):
                    if (j>=-1+i*0.002) and (j<-1+(i+1)*0.002):
                        self.ent_axis[i]+=1
            else:
                for i in range(500):
                    if (j>=0+i*0.002) and (j<0+(i+1)*0.002):
                        self.spa_axis[i]+=1
        self.ent_axis=self.ent_axis
        self.spa_axis=self.spa_axis
        if Plot==True:
            fig=plt.figure(figsize=(6,6),dpi=100)
            ax=fig.add_subplot(1,1,1,facecolor='white')
            ax.plot(mev_axis[:500],self.ent_axis,color='red',linewidth=0.7)
            ax.plot(mev_axis[500:],self.spa_axis,color='black',linewidth=0.7)
            ax.set_xlabel('minimum  eigenvalue of partial transposed matrix ')
            ax.set_ylabel('proportion')
        
        if Save!=None:
            if not os.path.exists('Random_Qubits/density_matrix/plot'):
                os.makedirs('Random_Qubits/density_matrix/plot')
            plt.savefig('Random_Qubits/density_matrix/plot/'+str(self.N)+'__'+Save+'_proportion',figsize=[7,7],facecolor='white',dpi=100)
        return self.PRO
    
    def ent_spa_sameprop(self,Save=None):
        #根据随机矩阵的情况 先暂时默认纠缠态多余可分态
        if self.PRO[0]>self.PRO[1]:
            num_select_ent=0
            i=1
            while num_select_ent<=self.PRO[1]:
                num_select_ent+=self.ent_axis[-i]
                i+=1
            eig_value_border=-(i-1)*0.002
            selected_index=np.where(np.array(self.PPT[1])>=eig_value_border)
            self.SIGMA=self.SIGMA[selected_index]
            self.N=len(self.SIGMA)
        
        if self.PRO[0]<self.PRO[1]:#可分态多时
            num_select_spa=0
            i=0
            while num_select_spa<=self.PRO[0]:
                num_select_spa+=self.spa_axis[i]
                i+=1
            eig_value_border=i*0.002
            selected_index=np.where(np.array(self.PPT[1])<=eig_value_border)
            self.SIGMA=self.SIGMA[selected_index]
            self.N=len(self.SIGMA)

        if Save!=None:
            if not os.path.exists('Random_Qubits'):
                os.makedirs('Random_Qubits')
            if not os.path.exists('Random_Qubits/density_matrix'):
                os.makedirs('Random_Qubits/density_matrix')
            #np.savetxt('density_matrix/data/'+'twoqubitsdensitymatrix'+str(N)+'.txt',SIGMA) #会把SIGMA从列表变成数组
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'__'+Save+'_with_sameprop',densitymatrix=self.SIGMA)
    
    def realization(self,Save=None):
        n=len(self.SIGMA[0])
        R_SIGMA=np.zeros((self.N,2*n,2*n))
        for num in range(self.N):
            R_SIGMA[num,:n,:n]=np.real(self.SIGMA[num])
            R_SIGMA[num,:n,n:]=-np.imag(self.SIGMA[num])
            R_SIGMA[num,n:,:n]=np.imag(self.SIGMA[num])
            R_SIGMA[num,n:,n:]=np.real(self.SIGMA[num])
        
        if Save!=None:
            if not os.path.exists('Random_Qubits'):
                os.makedirs('Random_Qubits')
            if not os.path.exists('Random_Qubits/density_matrix'):
                os.makedirs('Random_Qubits/density_matrix')
            #np.savetxt('density_matrix/data/'+'twoqubitsdensitymatrix'+str(N)+'.txt',SIGMA) #会把SIGMA从列表变成数组
            np.savez('Random_Qubits/density_matrix/'+str(self.N)+'__'+'realized_'+Save,densitymatrix=R_SIGMA)
            
        return R_SIGMA
