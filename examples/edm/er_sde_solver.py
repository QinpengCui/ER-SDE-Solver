import torch
import math

# Extended Reverse-time SDE solver for Diffusion Model
# ER-SDE-solver

# noise scale fuction
def customized_func(sigma, func_type=7, eta=0):
    """
    We provide several feasible special solutions.
    You can customize the specific solution you want as long as
    it satisfies the constraint equation in the paper.
    How to customize this noise scale fuction, see appendix A.8 in the paper.
    """
    if func_type == 1:  # ODE
        return sigma
    elif func_type == 2:  # Original SDE
        return sigma ** 2
    elif func_type == 3:  # SDE_1: sigma * (np.exp(-1 / sigma) + 10)
        return sigma * (torch.exp(sigma ** (eta - 1) / (eta - 1)) + 10)
    elif func_type == 4:  # SDE_2
        return sigma ** 1.5
    elif func_type == 5:  # SDE_3
        return sigma ** 2.5
    elif func_type == 6:  # SDE_4
        return sigma ** 0.9 * torch.log10(1 + 100 * sigma ** 1.5)
    elif func_type == 7:  # SDE_5
        return sigma * (torch.exp(sigma ** 0.3) + 10)

class ER_SDE_Solver:
    def __init__(
            self,
            sde_type = 've',
            model_prediction_type = 'x_start'
    ):  
        """
        Only ve and x_start are support in this version.
        EDM can be seen as a special VE-type, so we directly use it as VE.
        The remaining types will be added later.
        """
        assert model_prediction_type in ['noise', 'x_start', 'v']
        assert sde_type in ['ve', 'vp']
        self.sde_type = sde_type
        self.model_prediction_type = model_prediction_type

    def ve_xstart_1_order(
            self,
            model,
            x,
            sigmas,
            times,
            fn_sigma = customized_func,
            progress = False,
            **kwargs,
    ):
        """
        Euler Method, 1-order ER-SDE Solver.
        Support ve-type and model which predicts the data x_0.
        sigmas: index[0, 1, ..., N], sigmas[0] = sigma_max, sigma[N - 1] = sigma_min, sigma[N] = 0
        """
        assert self.model_prediction_type == 'x_start'
        assert self.sde_type == 've'

        num_steps = len(sigmas) - 1
        indices = range(num_steps)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            x0 = model(x, times[i], **kwargs)
            r_fn = fn_sigma(sigmas[i + 1]) / fn_sigma(sigmas[i])
            noise = torch.randn_like(x) * torch.sqrt(self.numerical_clip(sigmas[i + 1]**2 - sigmas[i]**2 * r_fn**2))
            x = r_fn * x + (1 - r_fn) * x0 + noise
        return x

    def ve_start_2_order_taylor(
            self,
            model,
            x,
            sigmas,
            times,
            fn_sigma = customized_func,
            progress = True,
            **kwargs,
    ):
        """
        Taylor Method, 2-order ER-SDE Solver.
        Support ve-type and model which predicts the data x_0.
        sigmas: index [0, 1, ...,   N], sigmas[0] = sigma_max, sigma[N-1] = sigma_min, sigmas[N] = 0
        """
        assert self.model_prediction_type == 'x_start'
        assert self.sde_type == 've'

        num_steps = len(sigmas) - 1
        indices = range(num_steps)

        nums_intergrate = 100.0
        nums_indices = torch.arange(nums_intergrate, dtype=torch.float64, device=x.device)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices) 
           
        old_x0 = None
        for i in indices:
            x0 = model(x, times[i], **kwargs)
            r_fn = fn_sigma(sigmas[i + 1]) / fn_sigma(sigmas[i])
            noise = torch.randn_like(x) * torch.sqrt(self.numerical_clip(sigmas[i + 1]**2 - sigmas[i]**2 * r_fn**2)) 
            if old_x0 == None or sigmas[i + 1]==0:
                x = r_fn * x + (1 - r_fn) * x0 + noise
            else:
                sigma_indices = sigmas[i + 1] + nums_indices/ nums_intergrate*(sigmas[i] - sigmas[i + 1])
                s_int = torch.sum(1.0 / fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(sigmas[i] - sigmas[i - 1])
                x = r_fn * x + (1 - r_fn) * x0 + (sigmas[i + 1] - sigmas[i] + s_int * fn_sigma(sigmas[i + 1])) * d_x0 + noise
            old_x0 = x0
        return x

    def ve_start_3_order_taylor(
            self,
            model,
            x,
            sigmas,
            times,
            fn_sigma = customized_func,
            progress = False,
            **kwargs,
    ):
        """
        Taylor Method, 3-order ER-SDE Solver.
        Support ve-type and model which predicts the data x_0.
        sigmas: index [0, 1, ...,   N], sigmas[0] = sigma_max, sigma[N-1] = sigma_min, sigmas[N] = 0
        """
        assert self.model_prediction_type == 'x_start'
        assert self.sde_type == 've'

        num_steps = len(sigmas) - 1
        indices = range(num_steps)

        nums_intergrate = 100.0
        nums_indices = torch.arange(nums_intergrate, dtype=torch.float64, device=x.device)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices) 
           
        old_x0 = None
        old_d_x0 = None
        for i in indices:
            x0 = model(x, times[i], **kwargs)
            r_fn = fn_sigma(sigmas[i + 1]) / fn_sigma(sigmas[i])
            noise = torch.randn_like(x) * torch.sqrt(self.numerical_clip(sigmas[i + 1]**2 - sigmas[i]**2 * r_fn**2)) 
            if old_x0 == None or sigmas[i + 1]==0:
                x = r_fn * x + (1 - r_fn) * x0 + noise
                old_x0 = x0
            elif (old_x0!= None) and (old_d_x0 == None):
                sigma_indices = sigmas[i + 1] + nums_indices/ nums_intergrate*(sigmas[i] - sigmas[i + 1])
                s_int = torch.sum(1.0 / fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(sigmas[i] - sigmas[i - 1])
                x = r_fn * x + (1 - r_fn) * x0 + (sigmas[i + 1] - sigmas[i] + s_int * fn_sigma(sigmas[i + 1])) * d_x0 + noise

                old_x0 = x0
                old_d_x0 = d_x0
            else:
                sigma_indices = sigmas[i + 1] + nums_indices/ nums_intergrate*(sigmas[i] - sigmas[i + 1])
                s_int = torch.sum(1.0 / fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                s_d_int = torch.sum((sigma_indices - sigmas[i])/ fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(sigmas[i] - sigmas[i - 1])
                dd_x0 = 2 * (d_x0 - old_d_x0)/(sigmas[i] - sigmas[i - 2])
                x = r_fn * x + (1 - r_fn) * x0 + (sigmas[i + 1] - sigmas[i] + s_int * fn_sigma(sigmas[i + 1])) * d_x0 \
                    + ((sigmas[i + 1] - sigmas[i])**2/2 + s_d_int * fn_sigma(sigmas[i + 1])) * dd_x0 + noise
                old_x0 = x0
                old_d_x0 = d_x0
        return x
      
    def numerical_clip(self, x, eps = 1e-6):
        """
        Correct some numerical errors.
        Preventing negative numbers due to computer calculation accuracy errors.
        """
        if torch.abs(x) < eps:
            return torch.tensor(0.0).to(torch.float64).to(x.device)
        else:
            return x








