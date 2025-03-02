#!/home/adriano/Escritorio/TFG/venv/bin/python3
from math import sqrt
import numpy as np
import tgen.ts_plots as plt
#import activity_data as act
import tgen.activity_data as act
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import joblib


def transformary(y):
     _y=np.ones(len(y))*-1

     for i in range(0,len(y)):
          for j in range(0,5):
               if y[i,j]<=1.1 and y[i,j]>=0.9:
                    _y[i]=j
     return _y

def main():
     data_name="WISDM"
     data_folder="/home/adriano/Escritorio/TFG/data/WISDM/"
     #voy a  obtener el maximo de todo el data set.
     X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False)
     print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
     created=True
     X_N=np.zeros((8163, 3, 129))
     
     for i in range(0,8163):
          X_N[i,0,:]=X_train[i,:,0]
          X_N[i,1,:]=X_train[i,:,1]
          X_N[i,2,:]=X_train[i,:,2]
     X_N=np.array(X_N)
     X_Nrp=np.zeros((8163, 3, 128))
     for i in range(0,8163):
          X_Nrp[i,0,:]=X_train[i,:-1,0]
          X_Nrp[i,1,:]=X_train[i,:-1,1]
          X_Nrp[i,2,:]=X_train[i,:-1,2]
     X_Nrp=np.array(X_Nrp)
     #print("X_train", X_N[-1,0,:],X_train[-1,:,0])
     
     stype="GAF"   
     model=SVC()
     
     y =transformary(y_train)
     if created==False:
          if "RP"==stype:
               model.fit(X_Nrp[:,0,:],y)
               joblib.dump(model,"modelo_svcrpx.joblib")
          else:
               model.fit(X_N[:,0,:],y)
               joblib.dump(model,"modelo_svcx.joblib")
     else:
          if "RP"==stype:
               model=joblib.load("modelo_svcrpx.joblib")
               
          else:
               model=joblib.load("modelo_svcx.joblib")
               
     data_folder="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
     data_folder1="/home/adriano/Escritorio/TFG/data/WISDM/tseries/GAF/sampling_loto/3-fold/fold-0/train"
     data_folder2="/home/adriano/Escritorio/TFG/data/WISDM/tseries/MTF/sampling_loto/3-fold/fold-0/train"
    
     indices_T=np.load(f"{data_folder}/y_training_data.npy")
     X_all_gaf=np.load(f"{data_folder1}/X_all_gaf.npy")
     X_all_mtf=np.load(f"{data_folder2}/X_all_mtf.npy")
     X_all_rec=np.load(f"{data_folder}/X_all_rec.npy")
     X_test=[]
     if "RP"==stype:
          X_test=X_all_rec[:,0,:]
     if "GAF"==stype:
          X_test=X_all_gaf[:,0,:]
     if "MTF"==stype:
          X_test=X_all_mtf[:,0,:]
     indices_T=transformary(indices_T)
     prediccion=model.predict(X_test)
    
     
     plt.plot_cm(indices_T,prediccion)
        #train_model(X_N,sj_train,5)
        #print("MODELO CREADO")
     #model=tf.keras.models.load_model("modelo.h5")
     #model.evaluate()
     print(classification_report(indices_T,prediccion))
     #model.evaluate(Reconstrucciones,sj_train)
if __name__ == '__main__':
    main()


    