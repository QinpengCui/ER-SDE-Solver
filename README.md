# ER-SDE-Solver

The official code for the paper [Elucidating the solution space of extended reverse-time SDE for diffusion models](https://arxiv.org/abs/2309.06169).

ER-SDE-Solver is a family of fast dedicated high-order solvers for extended reverse-time diffusion SDE (ER SDE) with the convergence order guarantee. Experiments have shown that ER-SDE-Solver can generate high-quality images in **around 20** function evaluations, achieving comparable levels to ODE-based solvers(such as [DPM-Solver](https://github.com/LuChengTHU/dpm-solver)).

<div style="display: flex; justify-content: space-between;"> 
  <img src="\assets\er_sde.bmp" style="width: 100%;"> 
</div>


## Usage

Our code will be released soon after the review is completed.





## Examples
Samples by **stochastic sampler** (ER-SDE-Solver-3 (ours)) and deterministic sampler (DPM-Solver-3) with 10, 20, 30, 40, 50 number of function evaluations (NFE) with the **same random seed** , using the pretrained model [guided-diffusion](https://github.com/openai/guided-diffusion) on ImageNet 256 × 256. The class is fixed as dome and classifier guidance scale is 2.0.


**DPM-Solver-3(left)**  and  **ER-SDE-Solver-3(right)** 

<div style="display: flex; justify-content: space-between;"> 
  <b>NFE=10<b> 
  <img src="\assets\DPM_ImageNet_256x256_10_steps.jpg" alt="Image 1" style="width: 40%;"> 
  <img src="\assets\Ours_ImageNet_256x256_10_steps.jpg"  alt="Image 2" style="width: 40%;">     
</div>
<br> 
<div style="display: flex; justify-content: space-between;"> 
  <b>NFE=20<b> <img src="\assets\DPM_ImageNet_256x256_20_steps.jpg" alt="Image 1" style="width: 40%;"> 
               <img src="\assets\Ours_ImageNet_256x256_20_steps.jpg"  alt="Image 2" style="width: 40%;">   
</div>
<br>  
<div style="display: flex; justify-content: space-between;"> 
  <b>NFE=30<b> <img src="\assets\DPM_ImageNet_256x256_30_steps.jpg" alt="Image 1" style="width: 40%;"> 
               <img src="\assets\Ours_ImageNet_256x256_30_steps.jpg"  alt="Image 2" style="width: 40%;">   
</div>
<br>  
<div style="display: flex; justify-content: space-between;"> 
  <b>NFE=40<b> <img src="\assets\DPM_ImageNet_256x256_40_steps.jpg" alt="Image 1" style="width: 40%;"> 
               <img src="\assets\Ours_ImageNet_256x256_40_steps.jpg"  alt="Image 2" style="width: 40%;">   
</div>
<br>  
<div style="display: flex; justify-content: space-between;"> 
  <b>NFE=50<b> <img src="\assets\DPM_ImageNet_256x256_50_steps.jpg" alt="Image 1" style="width: 40%;"> 
               <img src="\assets\Ours_ImageNet_256x256_50_steps.jpg"  alt="Image 2" style="width: 40%;">   
</div>




