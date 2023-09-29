import torch
import math
import numpy as np

# Extended Reverse-time SDE solver for Diffusion Model
# ER-SDE-solver


# noise scale fuction
def customized_func(x, func_type=7, eta=0):
    """
    We provide several feasible special solutions.
    You can customize the specific solution you want as long as
    it satisfies the constraint equation in the paper.
    How to customize this noise scale fuction, see appendix A.8 in the paper.
    """
    if func_type == 1:    # ODE
        return x
    elif func_type == 2:  # Original SDE
        return x ** 2
    elif func_type == 3:  # SDE_1: x * (np.exp(-1 / x) + 10)
        return x * (np.exp(x ** (eta - 1) / (eta - 1)) + 10)
    elif func_type == 4:  # SDE_2
        return x ** 1.5
    elif func_type == 5:  # SDE_3
        return x ** 2.5
    elif func_type == 6:  # SDE_4
        return x ** 0.9 * np.log10(1 + 100 * x ** 1.5)
    elif func_type == 7:  # SDE_5
        return x * (np.exp(x ** 0.3) + 10)

class ER_SDE_Solver:
    def __init__(
        self,
        sde_type = 've',
        model_prediction_type = 'x_start'
    ):  
        """
        Only ve and vp support in this version, the remaining types will be added later.
        Noise, x_start and score are support in this version.
        """
        assert model_prediction_type in ['noise', 'x_start', 'score']
        assert sde_type in ['ve', 'vp']
        self.sde_type = sde_type
        self.model_prediction_type = model_prediction_type

    def _predict_xstart_from_others(self, model_out, xt, sigma, alpha = 1.0):
        """
        Default sde_type is ve. If sde_type is vp, 
        alpha parameter needs to be passed in.
        """
        if self.model_prediction_type == 'x_start':
            return model_out
        elif self.model_prediction_type == 'noise':
            xstart = (xt - sigma * model_out) / alpha
            return xstart
        elif self.model_prediction_type == 'score':
            xstart = (xt + sigma**2 * model_out) / alpha 

    @torch.no_grad()
    def ve_1_order(
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
        len(sigmas) = num_steps + 1
        sigmas: index[0, 1, ..., N], sigmas[0] = sigma_max, sigma[N - 1] = sigma_min, sigma[N] = 0
        """
        assert self.sde_type == 've'

        num_steps = len(sigmas) - 1
        indices = range(num_steps)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            out = model(x, times[i], **kwargs)
            x0 = self._predict_xstart_from_others(out, x, sigmas[i])
            r_fn = fn_sigma(sigmas[i + 1]) / fn_sigma(sigmas[i])
            noise = torch.randn_like(x) * np.sqrt(self.numerical_clip(sigmas[i + 1]**2 - sigmas[i]**2 * r_fn**2))
            x = r_fn * x + (1 - r_fn) * x0 + noise
        return x

    @torch.no_grad()
    def vp_1_order(
        self,
        model,
        x,
        alphas,
        sigmas,
        times,
        fn_lambda = customized_func,
        progress = False,
        **kwargs,
    ):  
        """
        Euler Method, 1-order ER-SDE Solver.Support vp-type. 
        """
        assert self.sde_type == 'vp'

        lambdas = sigmas / alphas
        num_steps = len(lambdas) - 1
        indices = range(num_steps)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for i in indices:
            out = model(x, times[i], **kwargs)
            x0 = self._predict_xstart_from_others(out, x, sigmas[i], alphas[i])
            r_fn = fn_lambda(lambdas[i + 1]) / fn_lambda(lambdas[i])
            r_alphas = alphas[i + 1] / alphas[i]
            noise = torch.randn_like(x) * np.sqrt(self.numerical_clip(lambdas[i + 1]**2 - lambdas[i]**2 * r_fn**2)) * alphas[i + 1]     
            x = r_alphas * r_fn * x + alphas[i + 1] * (1 - r_fn) * x0 + noise
        return x

    @torch.no_grad()
    def ve_2_order_taylor(
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
        Taylor Method, 2-order ER-SDE Solver.
        len(sigmas) = num_steps + 1
        sigmas: index [0, 1, ...,   N], sigmas[0] = sigma_max, sigma[N-1] = sigma_min, sigmas[N] = 0
        """
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
            out = model(x, times[i], **kwargs)
            x0 = self._predict_xstart_from_others(out, x, sigmas[i])
            r_fn = fn_sigma(sigmas[i + 1]) / fn_sigma(sigmas[i])
            noise = torch.randn_like(x) * np.sqrt(self.numerical_clip(sigmas[i + 1]**2 - sigmas[i]**2 * r_fn**2)) 
            if old_x0 == None or sigmas[i + 1]==0:
                x = r_fn * x + (1 - r_fn) * x0 + noise
            else:
                sigma_indices = sigmas[i + 1] + nums_indices/ nums_intergrate*(sigmas[i] - sigmas[i + 1])
                s_int = np.sum(1.0 / fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(sigmas[i] - sigmas[i - 1])
                x = r_fn * x + (1 - r_fn) * x0 + (sigmas[i + 1] - sigmas[i] + s_int * fn_sigma(sigmas[i + 1])) * d_x0 + noise
            old_x0 = x0
        return x

    @torch.no_grad()
    def vp_2_order_taylor(
        self,
        model,
        x,
        alphas,
        sigmas,
        times,
        fn_lambda = customized_func,
        progress = False,
        **kwargs,
    ):
        """
        Taylor Method, 2-order ER-SDE Solver.Support vp-type.
        """
        assert self.sde_type == 'vp'

        lambdas = sigmas / alphas
        num_steps = len(lambdas) - 1
        indices = range(num_steps)

        nums_intergrate = 100.0
        nums_indices = np.arange(nums_intergrate, dtype=np.float64)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        old_x0 = None
        for i in indices:
            out = model(x, times[i], **kwargs)
            x0 = self._predict_xstart_from_others(out, x, sigmas[i], alphas[i])
            r_fn = fn_lambda(lambdas[i + 1]) / fn_lambda(lambdas[i])
            r_alphas = alphas[i + 1] / alphas[i]
            noise = torch.randn_like(x) * np.sqrt(self.numerical_clip(lambdas[i + 1]**2 - lambdas[i]**2 * r_fn**2)) * alphas[i + 1]
            if old_x0 == None:
                x = r_alphas * r_fn * x + alphas[i + 1] * (1 - r_fn) * x0 + noise
            else:
                lambda_indices = lambdas[i + 1] + nums_indices/ nums_intergrate*(lambdas[i] - lambdas[i + 1])
                s_int = np.sum(1.0 / fn_lambda(lambda_indices) * (lambdas[i] - lambdas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(lambdas[i] - lambdas[i - 1])
                x = r_alphas * r_fn * x + alphas[i + 1] * (1 - r_fn) * x0 + alphas[i + 1] * (lambdas[i + 1] - lambdas[i] + s_int * fn_lambda(lambdas[i + 1])) * d_x0 + noise
            old_x0 = x0
        return x

    @torch.no_grad()
    def ve_3_order_taylor(
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
        len(sigmas) = num_steps + 1
        sigmas: index [0, 1, ...,   N], sigmas[0] = sigma_max, sigma[N-1] = sigma_min, sigmas[N] = 0
        """
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
            out = model(x, times[i], **kwargs)
            x0 = self._predict_xstart_from_others(out, x, sigmas[i])
            r_fn = fn_sigma(sigmas[i + 1]) / fn_sigma(sigmas[i])
            noise = torch.randn_like(x) * torch.sqrt(self.numerical_clip(sigmas[i + 1]**2 - sigmas[i]**2 * r_fn**2)) 
            if old_x0 == None or sigmas[i + 1]==0:
                x = r_fn * x + (1 - r_fn) * x0 + noise
            elif (old_x0!= None) and (old_d_x0 == None):
                sigma_indices = sigmas[i + 1] + nums_indices/ nums_intergrate*(sigmas[i] - sigmas[i + 1])
                s_int = torch.sum(1.0 / fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(sigmas[i] - sigmas[i - 1])
                x = r_fn * x + (1 - r_fn) * x0 + (sigmas[i + 1] - sigmas[i] + s_int * fn_sigma(sigmas[i + 1])) * d_x0 + noise

                old_d_x0 = d_x0
            else:
                sigma_indices = sigmas[i + 1] + nums_indices/ nums_intergrate*(sigmas[i] - sigmas[i + 1])
                s_int = torch.sum(1.0 / fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                s_d_int = torch.sum((sigma_indices - sigmas[i])/ fn_sigma(sigma_indices) * (sigmas[i] - sigmas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(sigmas[i] - sigmas[i - 1])
                dd_x0 = 2 * (d_x0 - old_d_x0)/(sigmas[i] - sigmas[i - 2])
                x = r_fn * x + (1 - r_fn) * x0 + (sigmas[i + 1] - sigmas[i] + s_int * fn_sigma(sigmas[i + 1])) * d_x0 \
                    + ((sigmas[i + 1] - sigmas[i])**2/2 + s_d_int * fn_sigma(sigmas[i + 1])) * dd_x0 + noise

                old_d_x0 = d_x0
            old_x0 = x0
        return x

    @torch.no_grad()
    def vp_3_order_taylor(
        self,
        model,
        x,
        alphas,
        sigmas,
        times,
        fn_lambda = customized_func,
        progress = False,
        **kwargs,
    ):
        """
        Taylor Method, 3-order ER-SDE Solver.Support vp-type.
        """
        assert self.sde_type == 'vp'

        lambdas = sigmas / alphas
        num_steps = len(lambdas) - 1
        indices = range(num_steps)

        nums_intergrate = 100.0
        nums_indices = np.arange(nums_intergrate, dtype=np.float64)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        old_x0 = None
        old_d_x0 = None
        for i in indices:
            out = model(x, times[i], **kwargs)
            x0 = self._predict_xstart_from_others(out, x, sigmas[i], alphas[i])
            r_fn = fn_lambda(lambdas[i + 1]) / fn_lambda(lambdas[i])
            r_alphas = alphas[i + 1] / alphas[i]
            noise = torch.randn_like(x) * np.sqrt(self.numerical_clip(lambdas[i + 1]**2 - lambdas[i]**2 * r_fn**2)) * alphas[i + 1]
            if old_x0 == None:
                x = r_alphas * r_fn * x + alphas[i + 1] * (1 - r_fn) * x0 + noise
            elif (old_x0 != None) and (old_d_x0 == None):
                lambda_indices = lambdas[i + 1] + nums_indices/ nums_intergrate*(lambdas[i] - lambdas[i + 1])
                s_int = np.sum(1.0 / fn_lambda(lambda_indices) * (lambdas[i] - lambdas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(lambdas[i] - lambdas[i - 1])
                x = r_alphas * r_fn * x + alphas[i + 1] * (1 - r_fn) * x0 + alphas[i + 1] * (lambdas[i + 1] - lambdas[i] + s_int * fn_lambda(lambdas[i + 1])) * d_x0 + noise

                old_d_x0 = d_x0
            else:
                lambda_indices = lambdas[i + 1] + nums_indices/ nums_intergrate*(lambdas[i] - lambdas[i + 1])
                s_int = np.sum(1.0 / fn_lambda(lambda_indices) * (lambdas[i] - lambdas[i + 1]) / nums_intergrate)
                s_d_int = np.sum((lambda_indices - lambdas[i])/ fn_lambda(lambda_indices) * (lambdas[i] - lambdas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(lambdas[i] - lambdas[i - 1])
                dd_x0 = 2 * (d_x0 - old_d_x0)/(lambdas[i] - lambdas[i - 2])
                x = r_alphas * r_fn * x + alphas[i + 1] * (1 - r_fn) * x0 \
                    + alphas[i + 1] * (lambdas[i + 1] - lambdas[i] + s_int * fn_lambda(lambdas[i + 1])) * d_x0 \
                    + alphas[i + 1] * ((lambdas[i + 1] - lambdas[i])**2/2 + s_d_int * fn_lambda(lambdas[i + 1])) * dd_x0 + noise

                old_d_x0 = d_x0
            old_x0 = x0
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








