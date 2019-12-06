import numpy as np
import os
from scipy.signal import stft
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

QUAIS_FILTROS = ["sem filtro", "notch", "notch10x", "bandpass", "todos", "todos10x"]



# definicao de funcoes para filtros
def butter_bandpass(data, lowcut, highcut, fs=200, order=4):
    nyq = fs * 0.5
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return signal.filtfilt(b, a, data)


def butter_lowpass(data, lowcut, fs=200, order=4):
    nyq = fs * 0.5
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='lowpass')
    return signal.filtfilt(b, a, data)


def butter_highpass(data, highcut, fs=200, order=4):
    nyq = fs * 0.5
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='highpass')
    return signal.filtfilt(b, a, data)


def butter_notch(data, cutoff, var=1, fs=200, order=4):
    nyq = fs * 0.5
    low = (cutoff - var) / nyq
    high = (cutoff + var) / nyq
    b, a = signal.iirfilter(order, [low, high], btype='bandstop', ftype="butter")
    return signal.filtfilt(b, a, data)

# definição da função PSD para o sinal no domínio da frequência
def PSD(x):
    return np.sqrt(np.abs(x))


#leitura dos dados
path = 'emg_data_psd_2019-2/Danilo'
arquivos = os.listdir(path)

for filtro in QUAIS_FILTROS:
    print('\n', filtro)
    cont = 0
    for arquivo in arquivos:
        x = np.load(path + '/' + arquivo)
        x = np.transpose(x, (0, 2, 1))
        # print(x.shape)

        #filtros
        if filtro == "notch10x":
            for _ in range(10):
                x = butter_notch(x, 60)
        if filtro == "notch":
            for _ in range(10):
                x = butter_notch(x, 60)
        elif filtro == "bandpass":
            x = butter_notch(x, 60)
            x = butter_bandpass(x, 5, 50)
        elif filtro == "todos":
            x = butter_notch(x, 60)
            x = butter_highpass(x, 5)
            x = butter_lowpass(x, 50)
        elif filtro == "todos10x":
            for _ in range(10):
                x = butter_notch(x, 60)
                x = butter_highpass(x, 5)
                x = butter_lowpass(x, 50)

        #segmentação
        salto = 3
        segmento = 20
        n_win = int((x.shape[-1] - segmento) / salto) + 1
        ids = np.arange(n_win) * salto
        seg = np.array([x[:,:,k:(k + segmento)] for k in ids]).transpose(1, 2, 0, 3)
        # print(seg.shape)


        #stft
        _, _, w = stft(x, nperseg=50, noverlap=49, fs=200)
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
            feature = feature.reshape(5 * 2001, 4)
            # print('Feature: {}'.format(feature), feature.shape)
            features.append(feature)

        X = np.concatenate(features, axis=-1)
        # print(X.shape)


        #vetor de label
        y = np.array([[str(i)] * int(X.shape[0] / 5) for i in range(5)])
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
        print(arquivos[cont], 'Acurácia: {:.2f}%'.format(tot_hit / X_test.shape[0] * 100))
        cont -= -1









