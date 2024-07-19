
#https://github.com/xueleichen/PSNR-SSIM-UCIQE-UIQM-Python/blob/main/nevaluate.py
'''
Metrics for unferwater image quality evaluation.

Author: Xuelei Chen 
Email: chenxuelei@hotmail.com

Usage:
python evaluate.py RESULT_PATH
'''
#calcular UQIM
#calcular UCIQE
import numpy as np
#from skimage.measure import compare_psnr, compare_ssim
import math
import sys
from skimage import io, color, filters
import os
import math
#import wandb
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM 

def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    # c1 = 0.2745 # (menos peso, pois o brilho pode afetar a precisão da medição da cromaticidade)
    # c2 = 0.3742 # (maior peso, pois a saturação é importante para imagens com brilho)
    # c3 = 0.3743 # (maior peso, pois o contraste de luminância é crucial para imagens com brilho)
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    #2nd term
    top = int(np.round(0.01 * l.shape[0] * l.shape[1]))
    #top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top])-np.mean(sl[:top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753
    # p1 = 0.1938 # (menos peso, pois o brilho pode afetar a precisão da medição da cor)
    # p2 = 0.5155 # (maior peso, pois a nitidez é crucial para imagens com brilho)
    # p3 = 0.2907 # (peso moderado para o contraste)

    #1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = int(al1 * len(rgl))
    T2 = int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    #2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel=np.round(Rsobel).astype(np.uint8)
    Gsobel=np.round(Gsobel).astype(np.uint8)
    Bsobel=np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    #3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm,uciqe

def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch, blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb, ylb:yrb]
            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            if bottom == 0:
                m = 0
            else:
                m = top / bottom
            
            if m != 0:
                s += m * np.log(m)

    return plipmult(w, s)


def main():
    avaliacao = [ 
        ("/home/gusanagy/Documents/Glown-Diffusion/data/UDWdata/UIEB/val","/home/gusanagy/Documents/Glown-Diffusion/data/UDWdata/UIEB/val", "Claudio", "UIEB")
    ]

    for candidato in avaliacao:
        result_path ,gt, author ,dataset = candidato
        print(f"author: {author} dataset: {dataset}")
        
        result_dirs = os.listdir(result_path)
        result_gt = os.listdir(gt)

        sumuiqm, sumuciqe = 0.,0.
        psnr_sum, ssim_sum = 0., 0.
        N=0

        for imgdir, gt_file in tqdm(zip(result_dirs, result_gt), total=len(result_dirs)):
            if '.png' in imgdir or '.jpg' in imgdir and '.png' in gt_file or '.jpg' in gt_file:
                try:
                    corrected = io.imread(os.path.join(result_path, imgdir))
                    #gt_image = io.imread(os.path.join(result_path,gt))
                    gt_image = io.imread(os.path.join(gt, gt_file)) #ALTEREI
                except Exception as e:
                    print(f"Erro ao carregar a imagem: {e}")
                    continue

                try:
                    uiqm, uciqe = nmetrics(corrected)
                    psnr_value = PSNR(gt_image, corrected) #, data_range=255)
                    ssim_value = SSIM(gt_image, corrected, multichannel=True, win_size=3) #, data_range=255)
                except Exception as e:
                    print(f"Erro ao calcular métricas: {e}")
                    continue

                sumuiqm += uiqm
                sumuciqe += uciqe
                psnr_sum += psnr_value
                ssim_sum += ssim_value
                N += 1

        muiqm = sumuiqm / N
        muciqe = sumuciqe / N
        psnr_average = psnr_sum / N
        ssim_average = ssim_sum / N

        print(f'Average: uiqm={muiqm} uciqe={muciqe} psnr = {psnr_average} ssim = {ssim_average}')

if __name__ == '__main__':
    main()