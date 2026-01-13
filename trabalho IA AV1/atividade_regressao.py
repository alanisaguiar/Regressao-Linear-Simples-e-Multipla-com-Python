import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#Carregar os dados
data = np.loadtxt("aerogerador.dat", delimiter="\t")

X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
N = y.shape[0]

#Grafico
fig, ax = plt.subplots() #consertou n sei pq (perguntar)
ax.scatter(X, y, color="pink", alpha=0.6)
ax.set_xlabel("Velocidade do vento")
ax.set_ylabel("Potência gerada")
ax.set_title("Dispersão: Velocidade vs Potência")

X = np.hstack((
    np.ones((N,1)),X
))

#Desempenhos:
media_rss = []                          #RSS do modelo baseado na média
rss_mqo   = []                          #RSS do MQO tradicional
lambdas   = [0.25, 0.5, 0.75, 1.0]      #Ridge sem duplicar o MQO
rss_ridge = {lam: [] for lam in lambdas}

R = 500  #núm de rodadas Monte Carlo

#Monte Carlo (80/20)
for r in range(R):
    #embaralhar
    idx = np.random.permutation(N)
    Xr  = X[idx, :]
    yr  = y[idx, :]

    # Particionamento do conjunto de dados (80/20)
    cut = int(0.8 * N)
    X_treino, y_treino = Xr[:cut, :], yr[:cut, :]
    X_teste,  y_teste  = Xr[cut:, :], yr[cut:, :]

    # ----- Modelo da MÉDIA -----
    mu = float(np.mean(y_treino))
    beta_media = np.array([[mu], [0.0]])     # para bater com [1, x]
    y_hat = X_teste @ beta_media
    RSS = np.sum((y_teste - y_hat) ** 2)
    media_rss.append(RSS)

    # ----- MQO (OLS) -----
    XtX = X_treino.T @ X_treino
    Xty = X_treino.T @ y_treino
    beta_ols = inv(XtX) @ Xty
    y_hat = X_teste @ beta_ols
    RSS = np.sum((y_teste - y_hat) ** 2)
    rss_mqo.append(RSS)

    # ----- Ridge (Tikhonov) -----
    #n regulariza o intercepto:  
    D = np.eye(X_treino.shape[1])
    D[0, 0] = 0.0
    for lam in lambdas:
        beta_r = inv(XtX + lam * D) @ Xty
        y_hat = X_teste @ beta_r
        RSS = np.sum((y_teste - y_hat) ** 2)
        rss_ridge[lam].append(RSS)


#Estatísticas finais
def resumo(nome, valores):
    valores = np.asarray(valores, dtype=float)
    return {
        "Modelo": nome,
        "Média": float(np.mean(valores)),
        "Desvio-Padrão": float(np.std(valores, ddof=1)),
        "Maior": float(np.max(valores)),
        "Menor": float(np.min(valores)),
    }

tabela = []
tabela.append(resumo("Média da variável dependente", media_rss))
tabela.append(resumo("MQO tradicional", rss_mqo))
for lam in lambdas:
    tabela.append(resumo(f"MQO regularizado (λ={lam})", rss_ridge[lam]))

#exibe resultados bonitinho
print("\n--- Resultados (RSS no conjunto de TESTE, R=500) ---")
for linha in tabela:
    print("{:<32s}  Média: {:>12.6f} | Desv.Pad.: {:>12.6f} | Maior: {:>12.6f} | Menor: {:>12.6f}".format(
        linha["Modelo"], linha["Média"], linha["Desvio-Padrão"], linha["Maior"], linha["Menor"]
    ))

plt.show()
