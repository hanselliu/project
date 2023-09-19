class NetWork_test(object):
    def __init__(self,Parti_N,Meas,H,W,B,dtype=complex):
        #np.random.seed(666)
        self.dtype=dtype
        self.Parti_N=Parti_N  #粒子数
        self.Meas=Meas #测量数
        
        self.Hermite=H
        #print(self.Hermite)
        self.weights=W
        self.biases=B
        #print(self.biases[1])
    
    def get_parameter(self):
        return self.Hermite,self.weights,self.biases
    
    def forward(self,x,y):
        self.N=x.shape[0]
        
        self.Hermite_layer=Hermite_Measure_TII(self.Parti_N,self.Meas,self.Hermite)
        #self.sigmoid_lay0=Net.Sigmoid()
        
        self.full_connect_layer1=Net.Affine(self.weights[0],self.biases[0])
        self.sigmoid_lay1=Net.Relu()
        self.full_connect_layer2=Net.Affine(self.weights[1],self.biases[1])
        self.sigmoid_lay2=Net.Relu()
        self.full_connect_layer3=Net.Affine(self.weights[2],self.biases[2])
        self.sigmoid_lay3=Net.Relu()
        self.full_connect_layer4=Net.Affine(self.weights[3],self.biases[3])
        #self.sigmoid_lay4=Net.Relu()
        #self.full_connect_layer5=Net.Affine(self.weights[4],self.biases[4])
        self.final_layer=Net.SigmoidWithCrossEntropy()
        
        #print('x.shape:',x.shape)
        hermite=self.Hermite_layer.forward(x)  #hermite应该是(Meas,self.N)
        #hermite_a=self.sigmoid_lay0.forward(hermite)
        #print('hermite.shape:',hermite,hermite.shape)
        full_connect1_z=self.full_connect_layer1.forward(hermite)
        full_connect1_a=self.sigmoid_lay1.forward(full_connect1_z)
        #print('full_connect1_a.shape:',full_connect1_a.shape)
        full_connect2_z=self.full_connect_layer2.forward(full_connect1_a)
        full_connect2_a=self.sigmoid_lay2.forward(full_connect2_z)
        #print('full_connect2_a.shape:',full_connect2_a.shape)
        full_connect3_z=self.full_connect_layer3.forward(full_connect2_a)
        full_connect3_a=self.sigmoid_lay3.forward(full_connect3_z)
        full_connect4_z=self.full_connect_layer4.forward(full_connect3_a)
        #full_connect4_a=self.sigmoid_lay4.forward(full_connect4_z)
        
        #full_connect5_z=self.full_connect_layer5.forward(full_connect4_a)
        out,loss=self.final_layer.forward(full_connect4_z,y)
        #print('full_connect3_z.shape:',full_connect3_z.shape)
        #print('out.shape:',out.shape)
        return out,loss
    
    def evaluate(self,images,labels):
        out,loss=self.forward(images,labels)
        result=0
        for num in range(images.shape[0]):
            #if np.argmax(out[:,num])==np.argmax(labels[:,num]):
            if np.abs(out[0,num]-labels[0,num])<0.5:
                result+=1
        return result/labels.shape[1]
    class NetWork_test(object):
    def __init__(self,Parti_N,Meas,H,W,B,dtype=complex):
        #np.random.seed(666)
        self.dtype=dtype
        self.Parti_N=Parti_N  #粒子数
        self.Meas=Meas #测量数
        
        self.Hermite=H
        #print(self.Hermite)
        self.weights=W
        self.biases=B
        #print(self.biases[1])
    
    def get_parameter(self):
        return self.Hermite,self.weights,self.biases
    
    def forward(self,x,y):
        self.N=x.shape[0]
        
        self.Hermite_layer=Hermite_Measure_TII(self.Parti_N,self.Meas,self.Hermite)
        #self.sigmoid_lay0=Net.Sigmoid()
        
        self.full_connect_layer1=Net.Affine(self.weights[0],self.biases[0])
        self.sigmoid_lay1=Net.Relu()
        self.full_connect_layer2=Net.Affine(self.weights[1],self.biases[1])
        self.sigmoid_lay2=Net.Relu()
        self.full_connect_layer3=Net.Affine(self.weights[2],self.biases[2])
        self.sigmoid_lay3=Net.Relu()
        self.full_connect_layer4=Net.Affine(self.weights[3],self.biases[3])
        #self.sigmoid_lay4=Net.Relu()
        #self.full_connect_layer5=Net.Affine(self.weights[4],self.biases[4])
        self.final_layer=Net.SigmoidWithCrossEntropy()
        
        #print('x.shape:',x.shape)
        hermite=self.Hermite_layer.forward(x)  #hermite应该是(Meas,self.N)
        #hermite_a=self.sigmoid_lay0.forward(hermite)
        #print('hermite.shape:',hermite,hermite.shape)
        full_connect1_z=self.full_connect_layer1.forward(hermite)
        full_connect1_a=self.sigmoid_lay1.forward(full_connect1_z)
        #print('full_connect1_a.shape:',full_connect1_a.shape)
        full_connect2_z=self.full_connect_layer2.forward(full_connect1_a)
        full_connect2_a=self.sigmoid_lay2.forward(full_connect2_z)
        #print('full_connect2_a.shape:',full_connect2_a.shape)
        full_connect3_z=self.full_connect_layer3.forward(full_connect2_a)
        full_connect3_a=self.sigmoid_lay3.forward(full_connect3_z)
        full_connect4_z=self.full_connect_layer4.forward(full_connect3_a)
        #full_connect4_a=self.sigmoid_lay4.forward(full_connect4_z)
        
        #full_connect5_z=self.full_connect_layer5.forward(full_connect4_a)
        out,loss=self.final_layer.forward(full_connect4_z,y)
        #print('full_connect3_z.shape:',full_connect3_z.shape)
        #print('out.shape:',out.shape)
        return out,loss
    
    def evaluate(self,images,labels):
        out,loss=self.forward(images,labels)
        result=0
        for num in range(images.shape[0]):
            #if np.argmax(out[:,num])==np.argmax(labels[:,num]):
            if np.abs(out[0,num]-labels[0,num])<0.5:
                result+=1
        return result/labels.shape[1]
    class NetWork_Test2(object):
    def __init__(self,Parti_N,Meas,fn,H,W,B,add_I='no',dtype=complex):
        ###np.random.seed(666)
        self.dtype=dtype
        self.Parti_N=Parti_N  #粒子数
        self.Meas=Meas #测量数
        self.add_I=add_I
        
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,0-1j],[0+1j,0]])
        self.Z = np.array([[1,0],[0,-1]])
        self.E = np.array([[1,0],[0,1]])
        #关键处：卷积核的初始化 目前：（2，1，2，2）
        self.base=np.zeros((4,fn,1,2,2),complex) #(4 fn fc fw fh)
        for i in range(self.base.shape[1]):
            self.base[0,i,0]=self.X.T
            self.base[1,i,0]=self.Y.T
            self.base[2,i,0]=self.Z.T
            self.base[3,i,0]=self.E.T
        #print('self.base',self.base)
        
        self.Hermite=[]  #=[[base1_1,base1_2...],[base2_1,base2_2...],...]
        rd=np.zeros((Meas,Parti_N,4,self.base.shape[1],1,1,1))
        for par in range(Parti_N):
            rd[:,par,:,:,:,:,:]=np.random.uniform(-np.sqrt(6/((2**(Parti_N-par))**2+(2**(Parti_N-par-1))**2)),np.sqrt(6/((2**(Parti_N-par))**2+(2**(Parti_N-par-1))**2)),rd[:,par,:,:,:,:,:].shape)
        ##print('rd',rd)
        for mea in range(Meas):
            H_p_mea=[]
            for par in range(Parti_N):
                #print('mea=',mea,'par=',par)
                #print((2**(Parti_N-par))**2+(2**(Parti_N-par-1))**2)
                #rd=np.random.uniform(-np.sqrt(6/((2**(Parti_N-par))**2+(2**(Parti_N-par-1))**2)),np.sqrt(6/((2**(Parti_N-par))**2+(2**(Parti_N-par-1))**2)),(4,1,1,1,1))
                #print('rd',rd,rd.shape)
                h_p_m_p=np.sum(self.base*rd[mea,par],axis=0)
                H_p_mea.append(h_p_m_p)
            self.Hermite.append(H_p_mea)
        #print('self.Hermite',self.Hermite)
        
        self.Out_num=[0]  #计算各个路径各自最终输出所占的节点数
        out_loc=0
        for b in self.Hermite:
            out_num=1
            for bb in b:
                if add_I=='no':
                    out_num=out_num*(bb.shape[0])
                else:
                    out_num=out_num*(bb.shape[0]+1)
                    
            out_loc=out_loc+out_num
            self.Out_num.append(out_loc)
        #print('self.Out_num',self.Out_num)
        
        self.weights=[np.random.uniform(-np.sqrt(6/(self.Out_num[-1]+1024)),np.sqrt(6/(self.Out_num[-1]+1024)),(1024,self.Out_num[-1]))]
        self.weights.append(np.random.uniform(-np.sqrt(6/(1024*2)),np.sqrt(6/(1024*2)),(1024,1024)))
        self.weights.append(np.random.uniform(-np.sqrt(6/(1024*2)),np.sqrt(6/(1024*2)),(1024,1024)))
        #self.weights.append(np.random.uniform(-np.sqrt(6/(200+200)),np.sqrt(6/(200+200)),(200,200)))
        self.weights.append(np.random.uniform(-np.sqrt(6/(1024+1)),np.sqrt(6/(1024+1)),(1,1024)))
        
        self.biases=[np.random.uniform(-np.sqrt(6/(self.Out_num[-1]+1024)),np.sqrt(6/(self.Out_num[-1]+1024)),(1024,1))]#注意是列向量
        self.biases.append(np.random.uniform(-np.sqrt(6/(1024*2)),np.sqrt(6/(1024*2)),(1024,1)))
        self.biases.append(np.random.uniform(-np.sqrt(6/(1024*2)),np.sqrt(6/(1024*2)),(1024,1)))
        #self.biases.append(np.random.uniform(-np.sqrt(6/(200+100)),np.sqrt(6/(200+100)),(200,1)))
        self.biases.append(np.random.uniform(-np.sqrt(6/(1024+1)),np.sqrt(6/(1024+1)),(1,1)))
        #print(self.biases[1])
        
        self.Hermite=H
        self.weights=W
        self.biases=B
    
    def get_parameter(self):
        return self.Hermite,self.weights,self.biases
    
    def forward(self,x,y):
        self.N=x.shape[0]
        
        self.Hermite_layer=C_B_Hermite_Measure(self.Parti_N,self.Meas,self.Hermite,self.Out_num,self.add_I)#注意具体类型
        #self.sigmoid_lay0=Net.Sigmoid()
        
        self.full_connect_layer1=Net.Affine(self.weights[0],self.biases[0])
        self.sigmoid_lay1=Net.Relu()
        self.full_connect_layer2=Net.Affine(self.weights[1],self.biases[1])
        self.sigmoid_lay2=Net.Relu()
        self.full_connect_layer3=Net.Affine(self.weights[2],self.biases[2])
        self.sigmoid_lay3=Net.Relu()
        self.full_connect_layer4=Net.Affine(self.weights[3],self.biases[3])
        #self.sigmoid_lay4=Net.Relu()
        #self.full_connect_layer5=Net.Affine(self.weights[4],self.biases[4])
        self.final_layer=Net.SigmoidWithCrossEntropy()
        
        #print('x.shape:',x.shape)
        hermite=self.Hermite_layer.forward(x)  #hermite应该是(Meas,self.N)
        #hermite_a=self.sigmoid_lay0.forward(hermite)
        ##print('hermite.shape:',hermite,hermite.shape)
        full_connect1_z=self.full_connect_layer1.forward(hermite)
        full_connect1_a=self.sigmoid_lay1.forward(full_connect1_z)
        #print('full_connect1_a.shape:',full_connect1_a.shape)
        full_connect2_z=self.full_connect_layer2.forward(full_connect1_a)
        full_connect2_a=self.sigmoid_lay2.forward(full_connect2_z)
        #print('full_connect2_a.shape:',full_connect2_a.shape)
        full_connect3_z=self.full_connect_layer3.forward(full_connect2_a)
        full_connect3_a=self.sigmoid_lay3.forward(full_connect3_z)
        full_connect4_z=self.full_connect_layer4.forward(full_connect3_a)
        #full_connect4_a=self.sigmoid_lay4.forward(full_connect4_z)
        
        #full_connect5_z=self.full_connect_layer5.forward(full_connect4_a)
        out,loss=self.final_layer.forward(full_connect4_z,y)
        #print('full_connect3_z.shape:',full_connect3_z.shape)
        #print('out.shape:',out.shape)
        return out,loss
    
    def evaluate(self,images,labels):
        out,loss=self.forward(images,labels)
        result=0
        for num in range(images.shape[0]):
            #if np.argmax(out[:,num])==np.argmax(labels[:,num]):
            if np.abs(out[0,num]-labels[0,num])<0.5:
                result+=1
        return result/labels.shape[1],loss
    G2werer_train=np.load('Random_Qubits/density_matrix/60000__generalized_werner2.npz')['densitymatrix']
G2label_train=np.load('Random_Qubits/density_matrix/labels_of_60000__generalized_werner2.npz')['PPT']
G2werer_test=np.load('Random_Qubits/density_matrix/10000__test_generalized_werner2.npz')['densitymatrix']
G2label_test=np.load('Random_Qubits/density_matrix/labels_of_10000__test_generalized_werner2.npz')['PPT']
G2label_train=G2label_train[0]
G2label_test=G2label_test[0]
print(G2werer_train.shape)
print(G2label_train.shape)
print(G2werer_test.shape)
print(G2label_test.shape)

def label2D_for_twoclassify(label):
    y=np.zeros((1,label.shape[0]))
    y[0]=label
    return y

G2Werer_train=G2werer_train
G2Label_train=label2D_for_twoclassify(G2label_train)
G2Werer_test=G2werer_test
G2Label_test=label2D_for_twoclassify(G2label_test)

print(G2Werer_train.shape)
print(G2Label_train.shape)
print(G2Werer_test.shape)
print(G2Label_test.shape)
def G2werner_test():
    P=np.linspace(0,1,101)
    Sita=np.linspace(0,np.pi,101)
    Fai=np.linspace(0,2*np.pi,101)
    Rou=np.zeros((len(P),len(Sita),len(Fai),4,4),complex)
    Ent_label=np.zeros((len(P),len(Sita),len(Fai)))
    
    for i in range(len(P)):
        p=P[i]
        for j in range(len(Sita)):
            sita=Sita[j]
            for k in range(len(Fai)):
                fai=complex(str(Fai[k])+'j')
                vec=np.cos(0.5*sita)*np.array([[1,0,0,0]])+np.exp(fai)*np.sin(0.5*sita)*np.array([[0,0,0,1]])
                Rou[i,j,k]=p*np.dot(vec.T,vec.conjugate())+0.25*(1-p)*np.identity(4)
                eigen=0.25*(1-p)-p*np.cos(0.5*sita)*np.sin(0.5*sita)
                if eigen<0:
                    Ent_label[i,j,k]=0#纠缠
                else:
                    Ent_label[i,j,k]=1
    return Rou,Ent_label
Test_G2werner,Test_G2werner_label=G2werner_test()
# 3个测量量对G2 werner [隐藏层 1024 Relu ]
H1=np.load('卷积神经网络对werner分类汇总/1024relu/G2_werner/网络参数/3个测量量/H_of_G2werner.npz')['H']
LOSS1=np.load('卷积神经网络对werner分类汇总/1024relu/G2_werner/网络参数/3个测量量/Loss_of_G2werner.npz')['Loss']
W1=[]
B1=[]
for i in range(2):
    W1.append(np.load('卷积神经网络对werner分类汇总/1024relu/G2_werner/网络参数/3个测量量/W'+str(i)+'_of_G2werner.npz')['W'])
    B1.append(np.load('卷积神经网络对werner分类汇总/1024relu/G2_werner/网络参数/3个测量量/B'+str(i)+'_of_G2werner.npz')['B'])
print(H1.shape)

net1=NetWork_test(Parti_N=2,Meas=3,H=H1,W=W1,B=B1,dtype=complex)

Error_Rate1=np.zeros((101,101))
for i in range(Test_G2werner.shape[0]):
    for j in range(Test_G2werner.shape[1]):
        Error_Rate1[i,j]=1-net1.evaluate(Test_G2werner[i,j,:,:,:],Test_G2werner_label[i,j,:].reshape(1,101))
