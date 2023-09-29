import numpy as np
import torch
import math

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

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

class ER_SDE_Solver_Diffusion:
    def __init__(
        self,
        num_timesteps,
        schedule_name,
    ):  
        """
        Only VP and noise preiction are support in this version.
        The remaining types will be added later.
        """
        self.num_timesteps = num_timesteps
        if schedule_name in ["linear", "cosine"]:
            # uniform time step, +1 in order to stop at the moment colse to 0.
            use_timesteps = set(np.linspace(0, 1000, num_timesteps + 1, endpoint=False, dtype = np.int32))
            self.times = np.linspace(0, 1000, num_timesteps + 1, endpoint=False, dtype = np.int32).astype(np.float64)[::-1]

            # Orignal betas
            betas = get_named_beta_schedule(schedule_name, 1000)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            last_alpha_cumprod = 1
            # Create new betas
            new_betas = []
            for i, alpha_cumprod in enumerate(alphas_cumprod):
                if i in use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
            new_betas = np.array(new_betas)
            new_alphas = 1 - new_betas 
            alphas_cumprod = np.cumprod(new_alphas, axis=0)
            # vp_alphas, sqrt_alphas_cumprod
            self.vp_alphas = np.sqrt(alphas_cumprod)[::-1]
            # vp_sigmas, sqrt_one_minus_alphas_cumprod
            self.vp_sigmas = np.sqrt(1.0 - alphas_cumprod)[::-1]
            # vp_lambdas
            self.lambdas = np.sqrt(1.0 / alphas_cumprod - 1.0)[::-1]

    @torch.no_grad()
    def vp_nosie_1_order(
        self,
        model,
        shape,
        noise=None,
        model_kwargs=None,
        device=None,
        fn_lambda = customized_func,
        cond_fn = None,
        progress=False,
    ):  
        """
        Euler Method, 1-order ER-SDE Solver.
        Support vp-type and model which predicts the noise.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))
        s_in = img.new_ones([shape[0]]).to(device)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            eps = model(img, self.times[i] * s_in, **model_kwargs)
            eps = torch.split(eps, 3, dim=1)[0]
            if cond_fn:
                eps = eps - self.vp_sigmas[i] * cond_fn(img, self.times[i] * s_in, **model_kwargs)
            x0 = (img - self.vp_sigmas[i] * eps)/ self.vp_alphas[i]
            r_fn = fn_lambda(self.lambdas[i + 1]) / fn_lambda(self.lambdas[i])
            r_alphas = self.vp_alphas[i + 1] / self.vp_alphas[i]
            noise = torch.randn_like(img) * np.sqrt(self.numerical_clip(self.lambdas[i + 1]**2 - self.lambdas[i]**2 * r_fn**2)) * self.vp_alphas[i + 1]     
            img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * x0 + noise
        return img

    @torch.no_grad()
    def vp_nosie_2_order_taylor(
        self,
        model,
        shape,
        noise=None,
        model_kwargs=None,
        device=None,
        fn_lambda = customized_func,
        cond_fn = None,
        progress=False,
    ):
        """
        Taylor Method, 2-order ER-SDE Solver.
        Support vp-type and model which predicts the noise.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))
        s_in = img.new_ones([shape[0]]).to(device)

        nums_intergrate = 100.0
        nums_indices = np.arange(nums_intergrate, dtype=np.float64)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        old_x0 = None
        for i in indices:
            eps = model(img, self.times[i] * s_in, **model_kwargs)
            eps = torch.split(eps, 3, dim=1)[0]
            if cond_fn:
                eps = eps - self.vp_sigmas[i] * cond_fn(img, self.times[i] * s_in, **model_kwargs) 
            x0 = (img - self.vp_sigmas[i] * eps) / self.vp_alphas[i]
            r_fn = fn_lambda(self.lambdas[i + 1]) / fn_lambda(self.lambdas[i])
            r_alphas = self.vp_alphas[i + 1] / self.vp_alphas[i]
            noise = torch.randn_like(img) * np.sqrt(self.numerical_clip(self.lambdas[i + 1]**2 - self.lambdas[i]**2 * r_fn**2)) * self.vp_alphas[i + 1]
            if old_x0 == None:
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * x0 + noise
            else:
                lambda_indices = self.lambdas[i + 1] + nums_indices/ nums_intergrate*(self.lambdas[i] - self.lambdas[i + 1])
                s_int = np.sum(1.0 / fn_lambda(lambda_indices) * (self.lambdas[i] - self.lambdas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(self.lambdas[i] - self.lambdas[i - 1])
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * x0 + self.vp_alphas[i + 1] * (self.lambdas[i + 1] - self.lambdas[i] + s_int * fn_lambda(self.lambdas[i + 1])) * d_x0 + noise
            old_x0 = x0
        return img

    @torch.no_grad()
    def vp_nosie_3_order_taylor(
        self,
        model,
        shape,
        noise=None,
        model_kwargs=None,
        device=None,
        fn_lambda = customized_func,
        cond_fn = None,
        progress=False,
    ):
        """
        Taylor Method, 3-order ER-SDE Solver.
        Support vp-type and model which predicts the noise.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))
        s_in = img.new_ones([shape[0]]).to(device)

        nums_intergrate = 100.0
        nums_indices = np.arange(nums_intergrate, dtype=np.float64)
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        old_x0 = None
        old_d_x0 = None
        for i in indices:
            eps = model(img, self.times[i] * s_in, **model_kwargs)
            eps = torch.split(eps, 3, dim=1)[0]
            if cond_fn:
                eps = eps - self.vp_sigmas[i] * cond_fn(img, self.times[i] * s_in, **model_kwargs) 
            x0 = (img - self.vp_sigmas[i] * eps)/ self.vp_alphas[i]
            r_fn = fn_lambda(self.lambdas[i + 1]) / fn_lambda(self.lambdas[i])
            r_alphas = self.vp_alphas[i + 1] / self.vp_alphas[i]
            noise = torch.randn_like(img) * np.sqrt(self.numerical_clip(self.lambdas[i + 1]**2 - self.lambdas[i]**2 * r_fn**2)) * self.vp_alphas[i + 1]
            if old_x0 == None:
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * x0 + noise
                old_x0 = x0
            elif (old_x0 != None) and (old_d_x0 == None):
                lambda_indices = self.lambdas[i + 1] + nums_indices/ nums_intergrate*(self.lambdas[i] - self.lambdas[i + 1])
                s_int = np.sum(1.0 / fn_lambda(lambda_indices) * (self.lambdas[i] - self.lambdas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(self.lambdas[i] - self.lambdas[i - 1])
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * x0 + self.vp_alphas[i + 1] * (self.lambdas[i + 1] - self.lambdas[i] + s_int * fn_lambda(self.lambdas[i + 1])) * d_x0 + noise
                old_x0 = x0
                old_d_x0 = d_x0
            else:
                lambda_indices = self.lambdas[i + 1] + nums_indices/ nums_intergrate*(self.lambdas[i] - self.lambdas[i + 1])
                s_int = np.sum(1.0 / fn_lambda(lambda_indices) * (self.lambdas[i] - self.lambdas[i + 1]) / nums_intergrate)
                s_d_int = np.sum((lambda_indices - self.lambdas[i])/ fn_lambda(lambda_indices) * (self.lambdas[i] - self.lambdas[i + 1]) / nums_intergrate)
                d_x0 = (x0 - old_x0)/(self.lambdas[i] - self.lambdas[i - 1])
                dd_x0 = 2 * (d_x0 - old_d_x0)/(self.lambdas[i] - self.lambdas[i - 2])
                img = r_alphas * r_fn * img + self.vp_alphas[i + 1] * (1 - r_fn) * x0 \
                    + self.vp_alphas[i + 1] * (self.lambdas[i + 1] - self.lambdas[i] + s_int * fn_lambda(self.lambdas[i + 1])) * d_x0 \
                    + self.vp_alphas[i + 1] * ((self.lambdas[i + 1] - self.lambdas[i])**2/2 + s_d_int * fn_lambda(self.lambdas[i + 1])) * dd_x0 + noise
                old_x0 = x0
                old_d_x0 = d_x0
        return img

    def numerical_clip(self, x, eps = 1e-6):
        """
        Correct some numerical errors.
        Preventing negative numbers due to computer calculation accuracy errors.
        """
        if np.abs(x) < eps:
            return 0.0
        else:
            return x













