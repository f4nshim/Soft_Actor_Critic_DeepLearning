import math
import torch

def update_parameters(target, source, tau=1):
    target_params = target.parameters()
    source_params = source.parameters()

    

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def soft_update(target, source, tau):
    # 对应原文（Equation 9下面的那一段）：exponentially moving average of the value network weights, which has been shown to stabilize training
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source): # 等价于tau=1
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
