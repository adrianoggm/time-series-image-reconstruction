#!/home/adriano/Escritorio/TFG/venv/bin/python3
import numpy as np
import tgen.activity_data as act
import tgen.recurrence_plots as rec
import tgen.GAF as gaf
import tgen.MTF as mtf
import tgen.ts_plots as plots
import cv2
import argparse

"""./Playground_tsplot.py --image-type RP --dim 1 --val 4170"""

"""
Valpres interesantes para porbar
    GAF
    MEDIA Q 0.0044012627779020045 0.004399589350389923 4170
    MEDIA P 0.9998733035276017 0.9998733283554287 5406
    Min Q 7.442646919590436e-09 7.442646919590436e-09 4225
    Min P 0.9912638600577192 0.9912638600577192 1490
    Max Q 0.25154450250116284 0.25154450250116284 3661
    Max P 0.9999922113685015 0.9999922113685015 2156


    MTF
    MEDIA Q 110.39107219960607 110.38338496650113 4221
    MEDIA P 0.547349329114101 0.5474256616048051 3873
    Min Q 0.00039825152344224795 0.00039825152344224795 4218
    Min P 0.03311714899700962 0.03311714899700962 1558
    Max Q 1780.8914552240483 1780.8914552240483 3662
    Max P 0.9450789819158428 0.9450789819158428 377

    RP
    MEDIA Q 15.936688647211787 15.94236676824787 2084
    MEDIA P 0.7731528141371725 0.7731454984152836 4085
    Min Q 0.00015950539571661464 0.00015950539571661464 4424
    Min P 0.05182189333660742 0.05182189333660742 1104
    Max Q 416.0807310831489 416.0807310831489 3660
    Max P 0.9588916226063297 0.9588916226063297 2156
"""
def main():
     p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
     p.add_argument('--image-type',type=str, default="RP", help='Also you can reconstruct one specific image type GAF,RP,MTF')
     p.add_argument('--dim',type=int, default="0", help='0,1,2')
     p.add_argument('--val',type=int, default="0", help='0-5442 valores') 
     args = p.parse_args()
     
     data_folder="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
     X_train=np.load(f"{data_folder}/training_data.npy")#print(sj_train[:,0])
     #print(y_train[:,0])
     #print(X_train)
     #he obtenido el máximo y el minimo del dataset minimo -78.47761 maximo 66.615074
     a=args.val
     w = X_train[a]  
     
     rp=[]
     print(w.shape)
     dictionary=dict()
     for k in range(0,len(X_train)):
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
     plots.plot_time_series_g(w,rp)
if __name__ == '__main__':
    main()