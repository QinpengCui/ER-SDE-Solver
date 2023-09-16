# ER-SDE-Solver

The official code for the paper [Elucidating the solution space of extended reverse-time SDE for diffusion models](https://arxiv.org/abs/2309.06169).

ER-SDE-Solver is a family of fast dedicated high-order solvers for extended reverse-time diffusion SDE (ER SDE) with the convergence order guarantee. Experiments have shown that ER-SDE-Solver can generate high-quality images in **around 20** function evaluations, achieving comparable levels to ODE-based solvers(such as [DPM-Solver](https://github.com/LuChengTHU/dpm-solver)).

<img src="\assets\er_sde.bmp" width="953px" height="299px" >

## Usage

Our code will be released soon after the review is completed.







## Examples

Samples by **stochastic sampler** (ER-SDE-Solver-3 (ours)) and deterministic sampler (DPM-Solver-3) with 10, 20, 30, 40, 50 number of function evaluations (NFE) with the **same random seed** , using the pretrained model [guided-diffusion](https://github.com/openai/guided-diffusion) on ImageNet 256 × 256. 

$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$​ **DPM-Solver-3**  $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$​ **ER-SDE-Solver-3**

**NFE=10** <img src="\assets\DPM_ImageNet_256x256_10_steps.jpg" width="256px" height="128px"><img src="\assets\Ours_ImageNet_256x256_10_steps.jpg" width="256px" height="128px">

**NFE=20** <img src="\assets\DPM_ImageNet_256x256_20_steps.jpg" width="256px" height="128px"><img src="\assets\Ours_ImageNet_256x256_20_steps.jpg" width="256px" height="128px">

**NFE=30** <img src="\assets\DPM_ImageNet_256x256_30_steps.jpg" width="256px" height="128px"><img src="\assets\Ours_ImageNet_256x256_30_steps.jpg" width="256px" height="128px">

**NFE=40** <img src="\assets\DPM_ImageNet_256x256_40_steps.jpg" width="256px" height="128px"><img src="\assets\Ours_ImageNet_256x256_40_steps.jpg" width="256px" height="128px">

**NFE=50** <img src="\assets\DPM_ImageNet_256x256_50_steps.jpg" width="256px" height="128px"><img src="\assets\Ours_ImageNet_256x256_50_steps.jpg" width="256px" height="128px">



