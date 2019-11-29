import numpy as np
import os
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# definição da função PSD para o sinal no domínio da frequência
def PSD(x):
    return np.sqrt(np.abs(x))


#leitura dos dados
path = 'emg_data_psd_2019-2'
arquivos = os.listdir(path)

cont = 0
for arquivo in arquivos:
    x = np.load(path + '/' + arquivo)
    x = np.transpose(x, (0, 2, 1))
    # print(x.shape)

    #segmentação
    salto = 2
    segmento = 14
    n_win = int((x.shape[-1] - segmento) / salto) + 1
    ids = np.arange(n_win) * salto
    seg = np.array([x[:,:,k:(k + segmento)] for k in ids]).transpose(1, 2, 0, 3)
    # print(seg.shape)


    #stft
    _, _, w = stft(x, nperseg=14, noverlap=12, fs=200)
    w = np.swapaxes(w, 2, 3)
    # print(w.shape)


    #caracteristicas dominio da frequencia
    # FMD
    fmd = np.sum(PSD(w), axis=-1) / 2
    # print(fmd.shape)
    # MMDF
    mmdf = np.sum(np.abs(w), axis=-1) / 2
    # print(mmdf.shape)


    #vetor de caracteristicas
    features = list()
    for feature in (fmd, mmdf):
        feature = feature.transpose(0, 2, 1)
        feature = feature.reshape(4 * 701, 4)
        # print('Feature: {}'.format(feature), feature.shape)
        features.append(feature)

    X = np.concatenate(features, axis=-1)
    # print(X.shape)


    #vetor de label
    y = np.array([[str(i)] * int(X.shape[0] / 4) for i in range(4)])
    y = y.reshape(y.shape[0] * y.shape[1])
    # print(y.shape)


    #treinamento do modelo
    # dividindo as porções de dados em treino e teste (70 e 30% respectivamente)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    # print(clf)


    #classificação
    res = clf.predict(X_test)
    tot_hit = sum([1 for i in range(len(res)) if res[i] == y_test[i]])
    print('Acurácia: {:.2f}%'.format(tot_hit / X_test.shape[0] * 100), arquivos[cont])
    cont -= -1








    # input()