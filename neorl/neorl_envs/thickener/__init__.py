from neorl.neorl_envs.thickener.thickener_env import thickener


def get_env(env_name):
    if env_name == "thickener_random":
        sys_mode = "random"
    elif env_name == "thickener_cy":
        sys_mode = "cy"
    elif env_name == "thickener_const":
        sys_mode = "const"
    elif env_name == "thickener_s":
        sys_mode = "s"
    else:
        raise NotImplementedError
    env = thickener(sys_mode)
    return env


thickener_envs = {
        "thickener": "thickener_random",
        "thickener_random": "thickener_random",
        "thickener_cy": "thickener_cy",
        "thickener_const": "thickener_const",
        "thickener_s": "thickener_s",
}