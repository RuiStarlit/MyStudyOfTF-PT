import torch
from collections import OrderedDict
from collections import defaultdict

def totorch(x):
    return torch.autograd.Variable(x)

def inner_update_simple(model, loss , inner_lr):
    loss.backward()
    for param in model.parameters():
        param.data -= inner_lr * param.grad.data

def inner_update(model, grads, lr):
    """ Simple update
    """
    def set_parameter(current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters
    for (name, parameter), grad in zip(model.named_parameters(), grads):
        parameter.detach_()
        set_parameter(model, name, parameter.add(grad, alpha=-lr))

def inner_update_MAML(fast_weights, loss, inner_lr):
    """ Inner Loop Update """
    grads = torch.autograd.grad(
        loss, fast_weights.values(), create_graph=True)
    # Perform SGD
    fast_weights = OrderedDict(
        (name, param - inner_lr * grad)
        for ((name, param), grad) in zip(fast_weights.items(), grads))
    return fast_weights

def inner_update_alt1(fast_weights, loss, inner_lr):
    # first-order MAML
    grads = torch.autograd.grad(
        loss, fast_weights.values(), create_graph=False)
    # Perform SGD
    fast_weights = OrderedDict(
        (name, param - inner_lr * grad)
        for ((name, param), grad) in zip(fast_weights.items(), grads))
    return fast_weights

def inner_update_alt2(fast_weights, loss, inner_lr):
    # first-order 
    grads = torch.autograd.grad(
        loss, list(fast_weights.values())[-2:], create_graph=True)
    # Split out the logits
    for ((name, param), grad) in zip(
        list(fast_weights.items())[-2:], grads):
        fast_weights[name] = param - inner_lr * grad
    return fast_weights

class SimpleSGD(torch.optim.SGD):
    def __init__(self, net, *args, **kwargs):
        # Just like SGD with momentum
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def step(self, grads):
        group = self.param_groups[0]
        lr = group['lr']

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            self.set_parameter(self.net, name, parameter.add(grad, alpha=-lr))

class MetaSGD(torch.optim.SGD):
    def __init__(self, net, *args, **kwargs):
        # Just like SGD with momentum
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def step(self, grads):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            if weight_decay != 0:
                grad_wd = grad.add(parameter, alpha=weight_decay)
                # wd = wd + (weight_decay * p)
            else:
                grad_wd = grad
            if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
                buffer = self.state[parameter]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
                # buf = buf * momentum + (1-dampening) * wd
            else:
                grad_b = grad_wd
            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b
            self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))


class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, k, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state
    
    def load_state_dict_old(self, state_dict: dict) -> None:
        super(Lookahead, self).load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)