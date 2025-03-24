import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler

arquiteturas = [(15, 7), (30, 15), (60, 30)] 
iteracoes = {
    'teste2.npy': 500,  
    'teste3.npy': 700,
    'teste4.npy': 3000,
    'teste5.npy': 70000,
}
execucoes = 10  

def carregar_dados(arquivo_teste):
    arquivo = np.load(arquivo_teste, allow_pickle=True)
    x, y_original = arquivo[0], arquivo[1]
    scale = MaxAbsScaler().fit(y_original)
    y = np.ravel(scale.transform(y_original))
    return x, y, y_original

def treinar_rede(x, y, arq, max_iter):
    regr = MLPRegressor(
        hidden_layer_sizes=arq,
        max_iter=max_iter,
        activation='relu', 
        solver='adam',
        learning_rate='adaptive',
        n_iter_no_change=max_iter,
        random_state=np.random.randint(0, 10000)
    )
    regr.fit(x, y)
    return regr

def plot_resultados(x, y, y_est, loss_curve, media_erro, arquivo_teste, arq):
    plt.figure(figsize=[14, 7])
    plt.suptitle(f'{arquivo_teste} - Arquitetura {arq}', fontsize=14)

    plt.subplot(1, 3, 1)
    plt.title('Função Original')
    plt.plot(x, y, color='green')

    plt.subplot(1, 3, 2)
    plt.title(f'Curva de erro (Média {media_erro:.5f})')
    plt.plot(loss_curve, color='red')

    plt.subplot(1, 3, 3)
    plt.title('Original vs Aproximada')
    plt.plot(x, y, linewidth=1, color='green', label='Original')
    plt.plot(x, y_est, linewidth=2, color='blue', label='Aproximada')
    plt.legend()

    plt.show()

def main():
    for arquivo_teste, max_iter in iteracoes.items():
        x, y, y_original = carregar_dados(arquivo_teste)
        print(f'Arquivo: {arquivo_teste}')

        for arq in arquiteturas:
            erros_finais = []
            melhores_resultados = None
            menor_erro = float('inf')

            for _ in range(execucoes):
                regr = treinar_rede(x, y, arq, max_iter)
                erros_finais.append(regr.loss_)
                if regr.loss_ < menor_erro:
                    menor_erro = regr.loss_
                    melhores_resultados = (regr.predict(x), regr.loss_curve_)

            media_erro = np.mean(erros_finais)
            desvio_erro = np.std(erros_finais)
            print(f'  Arquitetura: {arq} | Média Erro: {media_erro:.5f} | Desvio Padrão: {desvio_erro:.5f}')

            y_est, loss_curve = melhores_resultados
            plot_resultados(x, y, y_est, loss_curve, media_erro, arquivo_teste, arq)

if __name__ == "__main__":
    main()
