import pickle
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import os

#利用循环的方法，从Matlab那里一点一点的取回文件来
Data_Length = 34799
X_train_Matlab = np.zeros([Data_Length,32,32,3],dtype = np.uint8)
y_train_Matlab = np.zeros([Data_Length],dtype = np.uint8)

eng = matlab.engine.connect_matlab()
Masked_DataBase = eng.workspace['Masked_DataBase']
Masked_DataBase_y = eng.workspace['Masked_DataBase_y']

for Counter in range(34799):
    AFig = np.array(Masked_DataBase[Counter],dtype=np.uint8)
    Alabel = np.array(Masked_DataBase_y[Counter],dtype=np.uint8)
    
    X_train_Matlab[Counter] = AFig
    y_train_Matlab[Counter] = Alabel
    
    print(Counter)

eng.quit()

#%% 保存训练信息到文件
pickle_file = './train_Data'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                    {
                        'X_train_Matlab': X_train_Matlab,
                        'y_train_Matlab': y_train_Matlab
                    },
            pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise


