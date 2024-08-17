#!/home/adriano/Escritorio/TFG/venv/bin/python3
import numpy as np
import tgen.activity_data as act
import tgen.recurrence_plots as rec
import tgen.GAF as gaf
import tgen.MTF as mtf
import tgen.ts_plots as plots
import cv2
import argparse

"""./Playground_tsplot.py --image-type RP --dim 1"""
def main():
     p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
     p.add_argument('--image-type',type=str, default="RP", help='Also you can reconstruct one specific image type GAF,RP,MTF')
     p.add_argument('--dim',type=int, default="0", help='0,1,2')
     args = p.parse_args()
     data_name="WISDM"
     data_folder="/home/adriano/Escritorio/TFG/data/WISDM/"
     #voy a  obtener el maximo de todo el data set.
     X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False)
     print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
     #print(sj_train[:,0])
     #print(y_train[:,0])
     #print(X_train)
     #he obtenido el máximo y el minimo del dataset minimo -78.47761 maximo 66.615074
     a=3301
     w = X_train[a]
     sj = sj_train[a][0]
     w_y = y_train[a]
     w_y_no_cat = np.argmax(w_y)
     rp=[]
     print(w.shape)
     dictionary=dict()
     for k in range(0,8163):
         l=X_train[k]
         #A=np.max(l[:,0])
         maximos=[np.max(l[:,0]),np.max(l[:,1]),np.max(l[:,2])]
         minimos=[np.min(l[:,0]),np.min(l[:,1]),np.min(l[:,2])]
         dictionary[k]=[maximos,minimos]

     dim=args.dim
     if args.image_type=="MTF":
        img = mtf.SavevarMTF_XYZ(w, 0, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129) 
        imagen = cv2.imread("./0x0mtf.png")  
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        print("Image shape",imagen.shape)
        rp=mtf.Reconstruct_MTF(imagen,dictionary,a)
     if  args.image_type=="RP": 
        img = rec.SavevarRP_XYZ(w, 0, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129) 
        imagen = cv2.imread("./0x0.png")  
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        print("Image shape",imagen.shape)
        rp,aux=rec.Reconstruct_RP(imagen,dictionary,a)
     if  args.image_type=="GAF": 
        img = gaf.SavevarGAF_XYZ(w, 0, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129) 
        imagen = cv2.imread("./0x0gasf.png")  
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        print("Image shape",imagen.shape)
        rp=gaf.Reconstruct_GAF(imagen,dictionary,a)
     plots.plot_time_series(w,rp,dim)#dimensión deber ser un valor entre 0-2
if __name__ == '__main__':
    main()