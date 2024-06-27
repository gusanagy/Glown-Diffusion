import wandb
import argparse
from src import train, test, mask_generation, inference

#import packages

#initialize classes
if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    modelConfig = {
  
        "DDP": False,
        "state": "train", # or eval
        "epoch": 1001,
        "batch_size":8 ,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", #MODIFIQUEI
        "device_list": [0, 1],#[0, 1]
        #"device_list": [3,2,1,0],
        
        "ddim":True,
        "unconditional_guidance_scale":1,
        "ddim_step":100
    }

    ##Adicionar ao arg parse o transfer learning manual para o mask diffusion
    parser.add_argument('--dataset', type=str, default="all")
    parser.add_argument('--model', type=str, default="standart")#mask is the second option
    parser.add_argument('--dataset_path', type=str, default="./data/UDWdata/")
    parser.add_argument('--state', type=str, default="train")  #or eval
    parser.add_argument('--pretrained_path', type=str, default=None)  #or eval
    parser.add_argument('--output_path', type=str, default="./output/")  #or eval

    config = parser.parse_args()
    
    wandb.init(
             project="CLEDiffusion",
             config=vars(config),
             name="GlowDiff SUIM data",
             tags=["Train","No mask", "GlowDiff"],
             group="Branch glown_diffusion_train",
             job_type="train",

         ) 
    
    for key, value in modelConfig.items():
        setattr(config, key, value)
    
    print(config)

    train(config)#importar a funcao ou classe de papeline de treinamento== treino/teste e carregar as configs e rodar

    wandb.finish()

    
#start trainig papeline
    # Start load config
    # start Training
    # Test with training

#end training papeline

#inference 

