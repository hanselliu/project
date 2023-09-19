import numpy as np
import math
#from keras.utils import to_categorical
from Neural_network import Neural_network_complex as Net
from Random_Qubits import State_Density_Matrix as SDM
import sys
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
class Convolution:
    def __init__(self,w,b,stride_w=1,stride_h=1,pad=0,dtype=float):
        self.dtype=dtype
        self.w=w
        self.b=b
        self.stride_w=stride_w
        self.stride_h=stride_h
        self.pad=pad
        
        self.img=None
        self.col=None
        self.col_w=None
        self.imgshape=None
    
    def forward(self,img):
        N, C, H, W = img.shape
        FN, C, FH, FW = self.w.shape
        self.img=img
        self.imgshape=img.shape
        OH=(H+2*self.pad-FH)//self.stride_h+1
        OW=(W+2*self.pad-FW)//self.stride_w+1
        img=np.pad(img,[(0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)],'constant')
        col=np.zeros((N,C,FH,FW,OH,OW),self.dtype)
        col_w=self.w.reshape(FN,-1).T#变成FN列 C*FH*FW行
        for y in range(FH):
            y_max=y+self.stride_h*OH
            for x in range(FW):
                x_max=x+self.stride_w*OW
                col[:,:,y,x,:,:]=img[:,:,y:y_max:self.stride_h,x:x_max:self.stride_w]  #每个x y对应共OH*OW*N*C个元素
        col=col.transpose(0,4,5,1,2,3).reshape(N*OH*OW,-1)#C*FH*FW列       
        
        out=np.dot(col,col_w)+self.b#N*OH*OW行 FN列
        out=out.reshape(N,OH,OW,FN).transpose(0,3,1,2)#转为数量 通道 行 列 N FN OH OWd
        
        self.col=col
        self.col_w=col_w
        return out
    
    def backward(self,delta):#delta为N FN OH OW  
        FN, C, FH, FW = self.w.shape
        N,FN,OH,OW=delta.shape
        #print('inner delta',delta)
        delta=delta.transpose(0,2,3,1).reshape(-1,FN)#变成N*OH*OW FN
        #print('self.col',self.col)
        db=np.sum(delta,axis=0)/N  #也就是在列上求sum 输出FN长度的一维数组
        dw=np.dot(self.col.T,delta)/N
        dw=dw.transpose(1,0).reshape(FN, C, FH, FW)
        #print('inner dw',dw)
        dcol=np.dot(delta,self.col_w.T)  #N*OH*OW C*FH*FW
        #dx = col2im(dcol, self.xshape, self.pool_h, self.pool_w, self.stride, self.pad)
        N, C, H, W = self.imgshape
        #OH=(H+2*self.pad-FH)//self.stride+1
        #OW=(W+2*self.pad-FW)//self.stride+1
        dcol=dcol.reshape(N,OH,OW,C,FH,FW).transpose(0,3,4,5,1,2)
        delta_next=np.zeros((N,C,H+2*self.pad+self.stride_h-1,W+2*self.pad+self.stride_w-1),self.dtype)
        
        for y in range(FH):
            y_max=y+self.stride_h*OH
            for x in range(FW):
                x_max=x+self.stride_w*OW
                delta_next[:,:,y:y_max:self.stride_h, x:x_max:self.stride_w]+=dcol[:,:,y,x,:,:]
        return delta_next[:,:,self.pad:H+self.pad,self.pad:W+self.pad],dw,db

class Hermite_Measure_TII:
    def __init__(self,n,meas,Base):  #Base-[(meas,n,FN,C,FH,FW)=(meas,n,1,1,2,2)]
        self.n=n
        self.meas=meas
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,0-1j],[0+1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
        self.E = np.array([[1,0],[0,1]])
        self.Base_w=Base  #Base-[(meas,n,FN,C,FH,FW)=(meas,n,1,1,2,2)]
        #print('self.Base_w',self.Base_w,self.Base_w.shape)
        #print(self.Base_w[0,0],self.Base_w[0,0].shape)
        
    def forward(self,sigma):#sigma要求 （N,H,W）
        self.N=sigma.shape[0]
        self.ConV=[]#各个卷积层所在
        out=np.zeros((self.meas,self.N))
        
        sigma_in=sigma.reshape((1,self.N,sigma.shape[1],sigma.shape[2])).transpose(1,0,2,3)#(N,1,H,W)
        for mea in range(self.meas):
            mea_Conv=[]#当前测量的卷积层列表
            mea_Conv_in=[sigma_in]
            for parti in range(self.n):
           #     print('mea=',mea,'parti=',parti,'_Conv_in_',mea_Conv_in[-1],mea_Conv_in[-1].shape)
                mea_Conv.append(Convolution(self.Base_w[mea,parti],b=np.array([0]),stride_w=2,stride_h=2,dtype=complex))
                conv_out=mea_Conv[-1].forward(mea_Conv_in[-1])#（N 1 H W）*（1 1 2 2）=(N 1 H/2 W/2)
                #print('conv_out',conv_out,conv_out.shape)  #所有parti轮完一定是(N 1 1 1)
                mea_Conv_in.append(conv_out)   #所有parti轮完一定是(N 1 1 1)
            self.ConV.append(mea_Conv)
            out[mea,:]=np.real(conv_out[:,0,0,0].reshape(self.N))
        #print('out')
        #print(len(self.ConV),len(self.ConV[0]))
        return out
    
    def backward(self,delta):#delta形状应为(meas,N) 和out相同
        dBase_w=np.zeros(self.Base_w.shape,complex)  #w是优化参数一个长度n数组：w=(meas, n, 1, 1, 2, 2)
        
        for mea in range(self.meas):
            delta_mea=delta[mea,:].reshape(self.N,1,1,1)  #(N 1 H/2 W/2)
            #print('delta_mea',delta_mea)
            for parti in range(1,self.n+1):
                #print('mea=',mea+1,'parti=',self.n+1-parti)
                #print('delta_mea_part',delta_mea,delta_mea.shape)
                delta_mea,dBase,db=self.ConV[mea][-parti].backward(delta_mea)  #dbase(1,1,2,2)
                #print('dbase',dBase,dBase.shape)
                #dBase=dBase*self.N  #因为ConV.backward求梯度时已经/N了 但此N非self.N，
                dBase_w[mea,-parti]=dBase
                #print('delta_mea',delta_mea,delta_mea.shape)
                #print('delta_mea_part',delta_mea,delta_mea.shape)
        return dBase_w

class Hermite_Measure_TIII_1:
    def __init__(self,n,meas,Base):  #Base-[(meas,n,FN,C,FH,FW)=(meas,n,1,1,2,2)]
        self.n=n
        self.meas=meas
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,0-1j],[0+1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
        self.E = np.array([[1,0],[0,1]])
        self.Base_w=Base  #Base-[(meas,n,FN,C,FH,FW)=(meas,n,1,1,2,2)]
        #print('self.Base_w',self.Base_w,self.Base_w.shape)
        #print(self.Base_w[0,0],self.Base_w[0,0].shape)
    
    def forward(self,sigma):#sigma要求 （N,H,W）
        self.N=sigma.shape[0]
        self.ConV=[]#各个卷积层所在
        out=np.zeros((self.meas*(2**self.n),self.N))
        
        sigma_in=sigma.reshape((1,self.N,sigma.shape[1],sigma.shape[2])).transpose(1,0,2,3)#(N,1,H,W)
        for mea in range(self.meas):
            mea_Conv=[]#当前测量的卷积层列表
            mea_Conv_in=[sigma_in]
            for parti in range(self.n):
                ##print('mea=',mea,'parti=',parti)
                #与单位矩阵I组合 准备进入卷积计算
                base_I=np.zeros((2,1,2,2),complex)
                base_I[0,0,:,:]=self.Base_w[mea,parti,0,0]
                base_I[1,0,:,:]=self.E
                
                mea_Conv.append(Convolution(base_I,b=np.array([0]),stride_w=2,stride_h=2,dtype=complex))
                ##print('mea_Conv_in[-1]',mea_Conv_in[-1].shape)
                conv_out=mea_Conv[-1].forward(mea_Conv_in[-1])#（N 1 H W）*（2 1 2 2）=(N 2 H/2 W/2)
                ##print('conv_out1',conv_out,conv_out.shape)
                #把通道并入数量 达成交叉
                conv_out=conv_out.reshape(mea_Conv_in[-1].shape[0]*2,1,mea_Conv_in[-1].shape[2]//2,mea_Conv_in[-1].shape[3]//2)
                ##print('conv_out1*',conv_out,conv_out.shape)
                
                mea_Conv_in.append(conv_out)   #所有parti轮完一定是(N*2**parti 1 1 1)
            self.ConV.append(mea_Conv)
            out[mea*(2**self.n):(mea+1)*(2**self.n),:]=np.real(conv_out.reshape(self.N,-1).T)
        
        ##print('out',out,out.shape)
        return out
    
    def backward(self,delta):#delta形状应为(meas*(2**n),N) 和out相同
        dBase_w=np.zeros(self.Base_w.shape,complex)  #w是优化参数一个长度n数组：w=(meas, n, 1, 1, 2, 2)
        
        for mea in range(self.meas):
            delta_mea=delta[mea*(2**self.n):(mea+1)*(2**self.n),:].T.reshape(self.N*(2**self.n),1,1,1)#变成conv_out1*的形状 单通道的
            #print('delta_mea',delta_mea)
            for parti in range(1,self.n+1):
                delta_mea=delta_mea.reshape(-1,2,delta_mea.shape[2],delta_mea.shape[3]) #回复到conv_out1的2通道形状(self.N*2**(self.n-1),2,1,1)
                #print('mea=',mea+1,'parti=',self.n+1-parti)
                #print('delta_mea_part',delta_mea,delta_mea.shape)
                delta_mea,dBase,db=self.ConV[mea][-parti].backward(delta_mea)  #dBase(2,1,2,2) delta(self.N*2**(self.n-1),1,2,2)
                #print('dbase',dBase,dBase.shape)
                #dBase=dBase*self.N  #因为ConV.backward求梯度时已经/N了 但此N非self.N，
                dBase_w[mea,-parti,0,0]=dBase[0,0]
                #print('delta_mea',delta_mea,delta_mea.shape)
                #print('delta_mea_part',delta_mea,delta_mea.shape)
        return dBase_w
    
class C_B_Hermite_Measure:
    def __init__(self,n,meas,BASE,Out_num,add_I='no'):  
        self.n=n
        self.meas=meas
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,0-1j],[0+1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
        self.E = np.array([[1,0],[0,1]])
        self.Base_w=BASE  #BASE=[[base1_1,base1_2...],[base2_1,base2_2...],...]
        self.add_I=add_I
        ##print('self.Base_w',self.Base_w)   #每一个base i_j-(FN,1,2,2)
        self.Out_num=Out_num
        ##print('self.Out_num',self.Out_num)
        
    def forward(self,sigma):#sigma要求 （N,H,W）
        self.N=sigma.shape[0]
        self.ConV=[]#各个卷积层所在
        out=np.zeros((self.Out_num[-1],self.N))
        
        sigma_in=sigma.reshape((1,self.N,sigma.shape[1],sigma.shape[2])).transpose(1,0,2,3)#(N,1,H,W)
        for mea in range(self.meas):
            mea_Conv=[]#当前测量的卷积层列表
            mea_Conv_in=[sigma_in]
            for parti in range(self.n):
                ##print('mea=',mea,'parti=',parti)
                #与单位矩阵I组合 准备进入卷积计算
                fn,fc,fw,fh=self.Base_w[mea][parti].shape
                if self.add_I!='no':
                    base=np.zeros((fn+1,fc,fw,fh),complex)
                    base[:-1,:,:,:]=self.Base_w[mea][parti]
                    base[-1,0,:,:]=self.E
                else:
                    base=self.Base_w[mea][parti]
                #print('base',base,base.shape)
                mea_Conv.append(Convolution(base,b=np.array([0]),stride_w=fw,stride_h=fh,dtype=complex))
                ##print('mea_Conv_in[-1]',mea_Conv_in[-1],mea_Conv_in[-1].shape)
                conv_out=mea_Conv[-1].forward(mea_Conv_in[-1])#（N 1 H W）*（FN 1 2 2）=(N FN H/2 W/2)
                ##print('conv_out1',conv_out,conv_out.shape)
                #把通道并入数量 达成交叉
                conv_out=conv_out.reshape(conv_out.shape[0]*conv_out.shape[1],1,conv_out.shape[2],conv_out.shape[3])
                ##print('conv_out1*',conv_out,conv_out.shape)
                
                mea_Conv_in.append(conv_out)   #所有parti轮完一定是(N*FN1*FN2... 1 1 1)
            self.ConV.append(mea_Conv)
            out[self.Out_num[mea]:self.Out_num[mea+1],:]=np.real(conv_out.reshape(self.N,-1).T)
        
        ##print('out',out,out.shape)
        return out
    
    def backward(self,delta):#delta形状应为(meas*(2**n),N) 和out相同
        dBASE=[]
        
        for mea in range(self.meas):
            dBase_mea=[]
            delta_mea=delta[self.Out_num[mea]:self.Out_num[mea+1],:].T.reshape(self.N*(self.Out_num[mea+1]-self.Out_num[mea]),1,1,1)#变成conv_out1*的形状 单通道的
            ##print('delta_mea',delta_mea)
            for parti in range(1,self.n+1):
                fn,fc,fw,fh=self.Base_w[mea][-parti].shape
                if self.add_I!='no':
                    fn+=1
                delta_mea=delta_mea.reshape(-1,fn,delta_mea.shape[2],delta_mea.shape[3]) #回复到conv_out1的2通道形状(self.N*2**(self.n-1),2,1,1)
                ##print('mea=',mea+1,'parti=',self.n+1-parti)
                ##print('delta_mea_part',delta_mea,delta_mea.shape)
                delta_mea,dBase,db=self.ConV[mea][-parti].backward(delta_mea)  #dBase(2,1,2,2) delta(self.N*2**(self.n-1),1,2,2)
                ##print('dbase',dBase,dBase.shape)
                if self.add_I=='no':
                    dBase_mea.append(dBase)
                else:
                    dBase_mea.append(dBase[:-1])
                #print('delta_mea',delta_mea,delta_mea.shape)
                #print('delta_mea_part',delta_mea,delta_mea.shape)
            dBase_mea.reverse()
            dBASE.append(dBase_mea)
            ##print('dBASE',dBASE)
        return dBASE