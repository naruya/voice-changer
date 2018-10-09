import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from config import wave_len
from config import window_size
from config import data_dim
from config import latent_dim
from config import batch_size

class DataLoaderforReconstOpt(object):
    def __init__(self, files):
        self.files = files
    
    # 一つの音声を分割して、全て返す。
    def load(self, index, shuffle=True, NO_SILENT=True, DEMO=False):
        
        files = self.files
        
        data = []
        amp = []
        
        if DEMO: # 無音も集める
            amp_all = []
            
        wave, _ = sf.read(files[index])
        # ランダムな場所から切り取りを始めることで、簡単なデータのかさ増し
        wave = wave[np.random.randint(wave_len):]
        
        for i in range(0, int(len(wave)/wave_len)):
            _data = wave[i*wave_len:(i+1)*wave_len]
            _amp = np.max(np.abs(_data))
            
            if DEMO:
                amp_all.append(_amp)
            
            if NO_SILENT and _amp < 0.005: # 無音を捨てる
                continue
            
            data.append(_data/_amp)
            amp.append(_amp)

        if shuffle:
            temp = list(zip(data, amp))
            np.random.shuffle(temp)
            data, amp = zip(*temp)
        
        data = np.array(data)
        amp = np.array(amp)
        
        if DEMO:
            
            print("【wave】")
            for i in range(4):
                print("amp:", amp[i])
                plt.figure(figsize=(20,1))
                plt.plot(data[i])
                plt.show()
                print("-"*80)
            
            amp_all = np.array(amp_all)
            print("【amp】 # max(amp)=={}, mean(amp)=={}".format(np.max(amp), np.mean(amp)))
            plt.figure(figsize=(12,1))
            plt.hist(amp_all, bins=500)
            plt.axvline(0.005, color="r")
            plt.show()
            
            print("data.shape ==", data.shape) # (49724, 160, 2) 等
            print("amp.shape ==", amp.shape) # (49724) 等
        return data, amp


class DataLoaderforLatentOpt(object):
    def __init__(self, files, index):
        
        data = []
        amp = []
        wave, _ = sf.read(files[index])
        
        wave = wave[np.random.randint(wave_len):]
        
        for i in range(0, int(len(wave)/wave_len)): # 切り分ける
            _data = wave[i*wave_len:(i+1)*wave_len]
            _amp = np.max(np.abs(_data))
            # このdata_loaderでは無音区間を捨てない
#             if NO_SILENT and _amp < 0.005: # 無音を捨てる
#                 continue
            data.append(_data/_amp)
            amp.append(_amp)
        
        self.data = np.array(data)
        self.amp = np.array(amp)
        
#         print("data.shape ==", self.data.shape) # (64812, 160, 2) 等
#         print("amp.shape ==", self.amp.shape) # (64812) 等
    
    # 一つの音声ファイルを分割して、batch_sizeのaとBを返す。
    def load(self, DEMO=False):
        
        # 有音のインデックスを集めて、aとB[0]を決定する。
        k_candidate = np.argwhere(self.amp>=0.005).T.flatten()
#         k_candidate = len(self.data) # 無音も使う場合

        k = np.random.choice(
            k_candidate-1,size=(batch_size, 1),replace=False)
        
        data_a = self.data[k] # (batch_size, 1, wave_len, data_dim)
        amp_a = self.amp[k] # (batch_size, 1)
        data_B = self.data[k+1] # (batch_size, 1, wave_len, data_dim)
        amp_B = self.amp[k+1] # (batch_size, 1)
        
        k_other = np.random.choice(
            k_candidate,size=(batch_size, n_choice-1),replace=False)
        
        _data_B = self.data[k_other] # (batch_size, 7, wave_len, data_dim)
        _amp_B = self.amp[k_other] # (batch_size, 7)
        
        data_B = np.append(data_B, _data_B, axis=1) # (batch_size, 8, wave_len, data_dim)
        amp_B = np.append(amp_B, _amp_B, axis=1) # (batch_size, 8)

        if DEMO:
           
            # (1+選択肢数)個すべての波形を横に並べて表示する。
            indices = np.random.choice(batch_size, 4)
            for i in indices:
                _x = np.array([data_a[i,0], *data_B[i]])
                
                x = np.concatenate(_x, 0)
                plt.figure(figsize=(12,2))
                plt.subplot(2,1,1)
                for j in range(n_choice):
                    plt.axvline((j+1)*wave_len, alpha=0.5)
                plt.plot(x)
                
                print("amp: ", end="")
                _x[0] *= amp_a[i,0]
                print("{0:.4f}".format(amp_a[i,0]), end=", ")
                for j in range(n_choice):
                    _x[j+1] *= amp_B[i,j]
                    print("{0:.4f}".format(amp_B[i,j]), end=", ")
                x = np.concatenate(_x, 0)
                plt.subplot(2,1,2)
                for j in range(n_choice):
                    plt.axvline((j+1)*wave_len, alpha=0.5)
                plt.plot(x)
                plt.show()
                
                print("-"*80)
                
            print("data_a.shape ==", data_a.shape) # (128, 1, 160, 2)
            print("amp_a.shape ==", amp_a.shape) # (128, 1)
            print("data_B.shape ==", data_B.shape) # (128, 4, 160, 2)
            print("amp_B.shape ==", amp_B.shape)# (128, 4)

        return data_a, data_B, amp_a, amp_B


class DataLoaderforVocoder(object):
    def __init__(self, files):
        self.files = files
    
    # 少しづつ重複させながら切り取る。
    def load(self, index, shuffle=True, DEMO=False, NO_SILENT=True):
        files = self.files
        
        data = []
        amp = []
        wave, _ = sf.read(files[index])
        t = 0
        while t < len(wave):
            _data = wave[t:t+wave_len]
            
            t += wave_len - 20*2 - 40 # 20==切り捨てる量, 40==かぶせる量
            
            if not len(_data) == wave_len:
                continue
            _amp = np.max(np.abs(_data))
            
            data.append(_data/_amp)
            amp.append(_amp)
            
        data = np.array(data)
        amp = np.array(amp)
        
        return data, amp