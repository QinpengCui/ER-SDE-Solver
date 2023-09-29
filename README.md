# ER-SDE-Solver

The official code for the paper [Elucidating the solution space of extended reverse-time SDE for diffusion models](https://arxiv.org/abs/2309.06169).

ER-SDE-Solver is a family of fast dedicated high-order solvers for extended reverse-time diffusion SDE (ER SDE) with the convergence order guarantee. Experiments have shown that ER-SDE-Solver can generate high-quality images in **around 20** function evaluations, achieving comparable levels to ODE-based solvers(such as [DPM-Solver](https://github.com/LuChengTHU/dpm-solver)).

<div style="display: flex; justify-content: space-between;"> 
  <img src="\assets\er_sde.bmp" style="width: 100%;"> 
</div>


## Usage
Before using our method, you need to confirm the *prediction type* of the pre-trained model and design the *noise schedule* (and *alphas schedule*) according to your needs. Then, refer to the following code example to use our method. 
#### For VE-type
```
from er_sde_solver import ER_SDE_Solver
sampler = ER_SDE_Solver(sde_type='ve', model_prediction_type='x_start')
x = sampler.ve_3_order_taylor(
    net,          # neural network
    x,            # initial Gaussian noise
    sigmas,       # noise schedule
    times,        # step size schedule
)
```
#### For VP-type
```
from er_sde_solver import ER_SDE_Solver
sampler = ER_SDE_Solver(sde_type='vp', model_prediction_type='x_start')
x = sampler.vp_3_order_taylor(
    net,          # neural network
    x,            # initial Gaussian noise
    alphas,       # alpha_t_bar schedule in DDPM
    sigmas,       # noise schedule
    times,        # step size schedule
)
```
We provide two specific usage examples, which are combined with [EDM](https://github.com/NVlabs/edm) and [guided-diffusion](https://github.com/openai/guided-diffusion). Please refer to the folder `examples` for details.



## Examples
Samples by **stochastic sampler** (ER-SDE-Solver-3 (ours)) and deterministic sampler (DPM-Solver-3) with 10, 20, 30, 40, 50 number of function evaluations (NFE) with the **same random seed** , using the pretrained model [guided-diffusion](https://github.com/openai/guided-diffusion) on ImageNet 256 × 256. The class is fixed as dome and classifier guidance scale is 2.0.


**DPM-Solver-3(left)**  and  **ER-SDE-Solver-3(right)** 

<div style="display: flex; justify-content: space-between;font-weight: normal;"> 
  NFE=10
  <img src="\assets\DPM_ImageNet_256x256_10_steps.jpg" alt="Image 1" style="width: 40%;"> 
  <img src="\assets\Ours_ImageNet_256x256_10_steps.jpg"  alt="Image 2" style="width: 40%;">     
</div>
<br> 
<div style="display: flex; justify-content: space-between;"> 
  NFE=20 
  <img src="\assets\DPM_ImageNet_256x256_20_steps.jpg" alt="Image 1" style="width: 40%;"> 
  <img src="\assets\Ours_ImageNet_256x256_20_steps.jpg"  alt="Image 2" style="width: 40%;">   
</div>
<br>  
<div style="display: flex; justify-content: space-between;"> 
  NFE=30 
  <img src="\assets\DPM_ImageNet_256x256_30_steps.jpg" alt="Image 1" style="width: 40%;"> 
  <img src="\assets\Ours_ImageNet_256x256_30_steps.jpg"  alt="Image 2" style="width: 40%;">   
</div>
<br>  
<div style="display: flex; justify-content: space-between;"> 
  NFE=40
  <img src="\assets\DPM_ImageNet_256x256_40_steps.jpg" alt="Image 1" style="width: 40%;"> 
  <img src="\assets\Ours_ImageNet_256x256_40_steps.jpg"  alt="Image 2" style="width: 40%;">   
</div>
<br>  
<div style="display: flex; justify-content: space-between;"> 
  NFE=50
  <img src="\assets\DPM_ImageNet_256x256_50_steps.jpg" alt="Image 1" style="width: 40%;"> 
  <img src="\assets\Ours_ImageNet_256x256_50_steps.jpg"  alt="Image 2" style="width: 40%;">   
</div>



## Citation
If you find this method and/or code useful, please consider citing

```bibtex
@article{cui2023elucidating,
  title={Elucidating the solution space of extended reverse-time SDE for diffusion models},
  author={Cui, Qinpeng and Zhang, Xinyi and Lu, Zongqing and Liao, Qingmin},
  journal={arXiv preprint arXiv:2309.06169},
  year={2023}
}
```

