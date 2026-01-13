import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv,inv

#Carregar os dados
data = np.loadtxt("Solubilidade.csv",delimiter=',')

X = data[:,:-1]
y = data[:,-1:]


fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],y[:,0],color='pink',edgecolor='k')
ax.set_xlabel("Quantidade de Carbono")
ax.set_ylabel("Peso Molecular")
ax.set_zlabel("Nível de Solubilidade")

X = np.hstack((
    np.ones((N,1)),X
))


#Desempenhos:
media_sse = []


#Definição da quantidade de rodadas
rodadas = 500
controle_plot = True
for r in range(rodadas):
    # Embaralhar o conjunto de dados
    idx = np.random.permutation(N)
    Xr = X[idx,:]
    yr = y[idx,:]
    
    # Particionamento do conjunto de dados (80/20)
    X_treino = Xr[:int(.8*N),:]
    y_treino = yr[:int(.8*N),:]
    
    X_teste = Xr[int(.8*N):,:]
    y_teste = yr[int(.8*N):,:]
    
    
    # Treino (Modelo baseado na média, Modelo baseado no MQO/OLS)
    # Média:
    beta_media = np.array([
        [np.mean(y_treino)],
        [0],
        [0]
    ])
    
    #MQO:
    beta_hat = inv(X_treino.T@X_treino)@X_treino.T@y_treino
    
    if controle_plot:
        fig = plt.figure(2)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X_treino[:,1],X_treino[:,2],y_treino[:,0],color='pink',edgecolor='k')
        # ax.scatter(X_teste[:,1],X_teste[:,2],y_teste[:,0],color='cyan',edgecolor='k')
        ax.set_xlabel("Quantidade de Carbono")
        ax.set_ylabel("Peso Molecular")
        ax.set_zlabel("Nível de Solubilidade")
        
        x1 = np.linspace(0,35)
        x2 = np.linspace(13,700)
        X3d,Y3d = np.meshgrid(x1,x2)
        Z = beta_media[0,0] + beta_media[1,0]*X3d + beta_media[2,0]*Y3d
        ax.plot_surface(X3d,Y3d,Z,cmap='gray',alpha=.1, edgecolors='k',
                        rstride=20,cstride=20)
        Z = beta_hat[0,0] + beta_hat[1,0]*X3d + beta_hat[2,0]*Y3d
        ax.plot_surface(X3d,Y3d,Z,cmap='turbo',alpha=.1, edgecolors='k',
                        rstride=20,cstride=20)
        controle_plot = False
    
    # Teste (SSE, MSE, R^2)
    #Modelo Média:
    y_hat_teste = X_teste@beta_media
    y_hat_treino = X_treino@beta_media
    
    print("--------  Média --------")
    SSE = np.sum((y_teste - y_hat_teste)**2)
    media_sse.append(SSE)
    print(f"{SSE:.5f}")
    MSE = np.mean((y_teste - y_hat_teste)**2)
    print(f"{MSE:.5f}")
    SST = np.sum((y_treino - np.mean(y_treino))**2)
    SSE = np.sum((y_treino - y_hat_treino)**2)
    R2 = 1 - SSE/SST
    print(f"{R2:.5f}")
    
    print("-----------  MQO --------")
    y_hat_teste = X_teste@beta_hat
    y_hat_treino = X_treino@beta_hat
    SSE = np.sum((y_teste - y_hat_teste)**2)
    print(f"{SSE:.5f}")
    MSE = np.mean((y_teste - y_hat_teste)**2)
    print(f"{MSE:.5f}")
    SST = np.sum((y_treino - np.mean(y_treino))**2)
    SSE = np.sum((y_treino - y_hat_treino)**2)
    R2 = 1 - SSE/SST
    print(f"{R2:.5f}")
    bp=1
    
    

plt.show()
bp=1