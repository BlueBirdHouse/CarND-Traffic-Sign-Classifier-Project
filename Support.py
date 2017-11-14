'''
本文件定义支持功能函数
'''
#%%包导入区
import random
import time
import matlab.engine
import matplotlib.pyplot as plt

#%%函数定义区
def RandomPrint(X,y):
    '''随机打印文件'''
    index = random.randint(0, len(X))
    image = X[index]

    plt.figure(figsize=(1,1))
    plt.imshow(image)
    plt.show()
    print(y[index])

def FigureDataToMatlab(DataBase,NumFig_Start,NumFig_End,MAT_Var_Name):
    '''
    随机输出NumFig个文件到Matlab
    '''
    Figs = DataBase[NumFig_Start:NumFig_End]
    eng = matlab.engine.connect_matlab()
    print('正在生成Python标准数据。')
    Figs = Figs.tolist()
    print('正在生成Matlab变量。')
    Figs = matlab.uint8(Figs)
    print('正在将数据传输给MATLAB')
    eng.workspace[MAT_Var_Name] = Figs
    eng.quit()

def ContinuousPrinting(DataBase):
    '''
    连续打印DataBase当中的内容不会停止
    '''
    for AFig in DataBase:
        plt.imshow(AFig)
        plt.show()
        time.sleep(1)
    
