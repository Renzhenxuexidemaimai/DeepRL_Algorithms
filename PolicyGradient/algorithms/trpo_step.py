#!/usr/bin/env python
# Created at 2020/1/22
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.distributions import kl_divergence

from Utils.torch_utils import device, set_flat_params, get_flat_parameters


def trpo_step(policy_net, policy_net_old, value_net, optimizer_value, optim_value_iternum, states, actions,
              returns, advantages, old_log_probs, max_kl, damping, l2_reg):
    """update critic"""
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        value_loss = nn.MSELoss()(values_pred, returns)
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update policy"""
    update_policy(policy_net, policy_net_old, states, actions, old_log_probs, advantages, max_kl, damping)


def conjugate_gradient(Avp_f, b, steps=10, rdotr_tol=1e-10):
    """
    reference <<Numerical Optimization>> Page 112
    :param Avp_f: function Avp_f(x) = A @ x
    :param b: equation
    :param steps: steps to run Conjugate Gradient Descent
    :param rdotr_tol: the threshold to stop algorithm
    :return: update direction
    """
    x = torch.zeros_like(b, device=device)  # initialization approximation of x
    r = -b.clone()  # Fvp(x) - b : residual
    p = b.clone()  # b - Fvp(x) : steepest descent direction
    rdotr = r.t() @ r  # r^T*r
    for i in range(steps):
        Avp = Avp_f(p)  # A @ p
        alpha = rdotr / (p.t() @ Avp)  # step length
        x += alpha * p  # update x
        r += alpha * Avp  # new residual
        new_rdotr = r.t() @ r
        betta = new_rdotr / rdotr  # beta
        p = -r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:  # satisfy the threshold
            break
    return x


def line_search(model, f, x, step_dir, expected_improve, max_backtracks=10, accept_ratio=0.1):
    """
    perform line search method for choosing step size
    :param model:
    :param f:
    :param x:
    :param step_dir: direction to update model parameters
    :param expected_improve:
    :param max_backtracks:
    :param accept_ratio:
    :return:
    """
    f_val = f(False).item()

    for step_coefficient in [.5 ** k for k in range(max_backtracks)]:
        x_new = x + step_coefficient * step_dir
        set_flat_params(model, x_new)
        f_val_new = f(False).item()
        actual_improve = f_val_new - f_val
        improve = expected_improve * step_coefficient
        ratio = actual_improve / improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


def update_policy(policy_net: nn.Module, policy_net_old: nn.Module, states, actions, old_log_probs, advantages, max_kl,
                  damping):
    def get_loss(grad=True):
        log_probs = policy_net.get_log_prob(states, actions)
        log_probs.requires_grad_(grad)
        ratio = torch.exp(log_probs - old_log_probs)
        loss = -(ratio * advantages).mean()
        return loss

    def Fvp(v):
        """
        compute vector product of second order derivative of KL_Divergence Hessian and v
        :param v: vector
        :return: \nabla \nabla H v
        """
        # compute kl divergence between current policy and old policy
        with torch.no_grad():
            dist_old = policy_net_old(states)
        dist_new = policy_net(states)
        kl = kl_divergence(dist_new, dist_old)
        kl = kl.mean()

        # first order gradient kl
        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()  # flag_grad_kl.t() @ v
        # second order gradient of kl
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping

    # compute first order approximation to Loss
    loss = get_loss()
    loss_grads = autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in loss_grads]).detach()  # g^T

    # conjugate gradient solve : H * x = g
    # apply vector product strategy here: Hx = H * x
    step_dir = conjugate_gradient(Fvp, loss_grad)  # approximation solution of H^(-1)g
    shs = loss_grad.t() @ step_dir  # g^T H^(-1) g; another implementation: Fvp(step_dir) @ step_dir
    lm = torch.sqrt(2 * max_kl / shs)
    step = lm * step_dir
    expected_improve = loss_grad.t() @ step

    """
    line search for step size 
    """
    current_flat_parameters = get_flat_parameters(policy_net)  # theta
    success, new_flat_parameters = line_search(policy_net, get_loss, current_flat_parameters, expected_improve, 10)
    set_flat_params(policy_net, new_flat_parameters)
    # success indicating whether TRPO works as expected
    return success
