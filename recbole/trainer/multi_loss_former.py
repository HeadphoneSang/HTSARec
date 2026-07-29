import torch
from recbole.utils.wechat import save_result_to_file
from recbole.utils.wechat import dict_extract_keys_recursive
from collections import deque


class AbstractLossFrom:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.cur_step = 0

    def form(self, losses):
        """
        Takes in a list of losses (tensors)
        Returns a list of weights and is_backward
        Args:
            losses(tuple): losses of all tasks


        Returns:
            loss: total loss
            loss_tuple: scalar values of all processed task losses as a tuple
            is_backward(Bool): whether backpropagation has been performed
        """
        raise NotImplementedError

    def save_weights_records(self):
        return NotImplementedError

    def after_update_w(self):
        return NotImplementedError

    def __call__(self, losses):
        res = self.form(losses)
        self.cur_step += 1
        return res


class EmptyLossForm(AbstractLossFrom):
    def __init__(self, model, config):
        super(EmptyLossForm, self).__init__(model, config)

    def form(self, losses):
        return sum(losses), losses, False

    def save_weights_records(self):
        return None

    def after_update_w(self):
        return None


class GradNormLossForm(AbstractLossFrom):
    def __init__(self, model, config):
        super(GradNormLossForm, self).__init__(model, config)
        self.share_layers = model.get_shared_layers()
        self.L0_list = []  # record the initial loss of each loss function
        self.weight_records = []  # record all updated weight values for visualization and computing static weights
        self.G_records = []
        self.loss_records = []
        self.r_records = []
        self.a = config['a_value']
        self.norm_step = config['norm_step']
        self.eps = 1e-12
        # Initialize the initial weights for multi-task losses
        weights_list = config['loss_weight']
        self.loss_weights = torch.tensor(weights_list, dtype=torch.float64, device='cuda', requires_grad=True)
        self.T = torch.sum(self.loss_weights, dim=0).item()
        self.lr = config['w_lr']
        # record the grad loss
        self.grad_loss_his = []
        # Parameters related to early stopping
        self.stop_step = config['stop_time_step']
        self.cur_check_step = 0
        self.check_len = config['check_len']
        self.valid_delta = 0
        self.valid_0 = [weight.item() for weight in self.loss_weights]
        self.stop_check = config['stop_check']
        self.use_change = True
        self.max_delta_weights = self.valid_0.copy()

    def static_form(self, losses):
        weighted_losses = tuple([self.loss_weights[i].item() * loss for i, loss in enumerate(losses)])
        return sum(weighted_losses), weighted_losses, False

    def early_stop(self):
        if self.cur_step == 0:
            return True
        # Need to compute the mean for comparison
        if self.cur_step % self.check_len == 0:
            begin = self.cur_step - self.check_len + 1
            end = self.cur_step + 1  # +1 because of weight0
            avg_weights = [sum(weights[begin: end]) / self.check_len for weights in self.weight_records]
            cur_delta = [abs(weight - self.valid_0[i]) for i, weight in enumerate(avg_weights)]
            cur_delta = sum(cur_delta) / len(cur_delta)
            if cur_delta > self.valid_delta:
                self.valid_delta = cur_delta
                self.cur_check_step = 0
                self.max_delta_weights = avg_weights
            else:
                self.cur_check_step += 1
                # trigger early stopping
                if self.cur_check_step >= self.stop_step:
                    with torch.no_grad():
                        final_weights = [(avg_weight + max_delta_weight) / 2 for avg_weight, max_delta_weight in
                                         zip(avg_weights, self.max_delta_weights)]
                        for param, final_weight in zip(self.loss_weights, final_weights):
                            param.data = torch.tensor(final_weight, dtype=torch.float64, device='cuda',
                                                      requires_grad=False)
                    return False
                else:
                    return True
        return True

    def __call__(self, losses):
        if self.use_change:
            res = self.form(losses)
        else:
            res = self.static_form(losses)
            for i, loss in enumerate(losses):
                # record weights
                self.weight_records[i].append(self.loss_weights[i].item())
                self.loss_records[i].append(loss.item())
        if self.use_change and self.stop_check:
            self.use_change = self.early_stop()
        self.cur_step += 1
        return res

    def form(self, losses):
        """

        Args:
            losses(tuple(Tensor)): tuple of Tensors composing the loss functions of all tasks

        Returns:
            the sum of all tasks' losses as a (tensor) (1,), the Tensor (T,) of all weighted task losses, and whether backpropagation has been performed
        """
        weighted_losses = tuple([self.loss_weights[i].item() * loss for i, loss in enumerate(losses)])
        if self.cur_step % self.norm_step != 0:
            return sum(weighted_losses), weighted_losses, False
        assert len(losses) == self.loss_weights.shape[0], "the number of initial loss weights does not match the number of loss functions"
        # Compute the weighted values of all loss functions, used for return; the weights here are constants without gradient updates
        # If this is the first time, record L0
        if self.cur_step == 0:
            for i, loss in enumerate(losses):
                self.L0_list.append(loss.detach())
                self.weight_records.append([])
        # Compute G_W_i
        G_w_i = []
        L_hat_i = []
        for i, loss in enumerate(losses):
            # record weights
            self.weight_records[i].append(self.loss_weights[i].item())
            # Compute G and L
            grads = torch.autograd.grad(loss, inputs=self.share_layers, retain_graph=True)
            grads = torch.cat([g.view(-1) for g in grads], dim=0)
            L2_i = torch.norm(grads, p=2)
            G_w_i.append(self.loss_weights[i] * L2_i.detach())
            L_hat_i.append(loss / (self.L0_list[i] + self.eps))
        E_G = (sum(G_w_i) / len(G_w_i))
        E_L_hat = (sum(L_hat_i) / len(L_hat_i)) + self.eps
        r_i = [L_hat / E_L_hat for L_hat in L_hat_i]
        G_targets = [(E_G * (r ** self.a)).detach() for r in r_i]
        G_targets = torch.stack(G_targets, dim=0)
        G_w_i = torch.stack(G_w_i, dim=0)
        weight_loss = torch.sum(torch.abs(G_w_i - G_targets), dim=0)
        # Record the value of the weight update loss
        self.grad_loss_his.append(weight_loss.item())
        if torch.isnan(weight_loss):
            print("weight is nan")
            print(self.loss_weights)
        grads = torch.autograd.grad(weight_loss, inputs=self.loss_weights, retain_graph=False)
        with torch.no_grad():
            for param, grad in zip(self.loss_weights, grads[0]):
                param.data -= self.lr * grad  # manual update
        return sum(weighted_losses), weighted_losses, False

    def nor_weights(self):
        """
        Normalize the weights
        Returns:

        """

        normalize_co = self.T / torch.sum(self.loss_weights, dim=0)
        with torch.no_grad():
            self.loss_weights.mul_(normalize_co)  # in-place multiplication

    def after_update_w(self):
        if self.use_change:
            return self.nor_weights()

    def save_weights_records(self):
        """
        Record all changes of the dynamic weights
        Returns: whether saving succeeded
        """
        task_num = len(self.weight_records)
        E_weights = [0] * task_num
        step_num = len(self.weight_records[0])
        for task_i in range(task_num):
            E_weights[task_i] = sum(self.weight_records[task_i]) / step_num
        save_dict = {
            'weight_records': self.weight_records,
            "G_i_records": self.G_records,
            "loss_records": self.loss_records,
            "grad_losses": self.grad_loss_his,
            "r_records": self.r_records,
            'GradNorm_Static_weights': E_weights
        }
        wanted_list = ['weight_records', 'G_i_records', "loss_records", 'GradNorm_Static_weights', 'r_records',
                       'grad_losses']
        save_result_to_file(dict_extract_keys_recursive(save_dict, wanted_list))
        return None


class StaticLossForm(AbstractLossFrom):
    def __init__(self, model, config):
        super(StaticLossForm, self).__init__(model, config)
        weights_list = config['loss_weight']
        self.loss_weights = torch.tensor(weights_list, dtype=torch.float64, device='cuda')

    def form(self, losses):
        weighted_losses = tuple([self.loss_weights[i].item() * loss for i, loss in enumerate(losses)])
        return sum(weighted_losses), weighted_losses, False

    def after_update_w(self):
        return None


class GradNormLossAvgForm(GradNormLossForm):

    def __init__(self, model, config):
        super(GradNormLossAvgForm, self).__init__(model, config)
        self.his_len = int(config['win_len'])
        self.history_loss = None

    def init_history_loss(self, losses_num, losses0):
        """
        Initialize the history-loss queue according to the number of tasks and the loss tensors from the first task
        Args:
            losses_num: number of tasks
            losses0: loss tensors of all tasks from the first iteration

        Returns: None

        """
        self.history_loss = [deque([losses0[i].item()], maxlen=self.his_len) for i in range(0, losses_num)]
        for i in range(losses_num):
            self.weight_records.append([])
        return True

    def form(self, losses):
        """

        Args:
            losses(tuple(Tensor)): tuple of Tensors composing the loss functions of all tasks

        Returns:
            the sum of all tasks' losses as a (tensor) (1,), the Tensor (T,) of all weighted task losses, and whether backpropagation has been performed
        """
        weighted_losses = tuple([self.loss_weights[i].item() * loss for i, loss in enumerate(losses)])
        # If this is the first time, record L0
        if self.cur_step == 0 or self.history_loss is None:
            self.init_history_loss(len(losses), losses)
            for i in range(len(losses)):
                self.G_records.append([])
                self.loss_records.append([])
                self.r_records.append([])
        else:
            for i, loss in enumerate(losses):
                self.history_loss[i].append(loss.item())
        # if self.cur_step % self.norm_step != 0 or self.cur_step < self.his_len:
        #     return sum(weighted_losses), weighted_losses, False
        assert len(losses) == self.loss_weights.shape[0], "the number of initial loss weights does not match the number of loss functions"
        # Compute the weighted values of all loss functions, used for return; the weights here are constants without gradient updates
        # Compute G_W_i
        G_w_i = []
        L_hat_i = []
        self.L0_list = [sum(q) / len(q) for q in self.history_loss]
        for i, loss in enumerate(losses):
            # record weights
            self.weight_records[i].append(self.loss_weights[i].item())
            # Compute G and L
            grads = torch.autograd.grad(loss, inputs=self.share_layers, retain_graph=True)
            grads = torch.cat([g.view(-1) for g in grads], dim=0)
            L2_i = torch.norm(grads, p=2)
            G_w_i.append(self.loss_weights[i] * L2_i.detach())
            self.G_records[i].append(G_w_i[i].item())
            L_hat_i.append(loss / (self.L0_list[i] + self.eps))
        E_G = (sum(G_w_i) / len(G_w_i))
        E_L_hat = (sum(L_hat_i) / len(L_hat_i)) + self.eps
        r_i = [L_hat / E_L_hat for L_hat in L_hat_i]
        for i, r in enumerate(r_i):  # record G_i for each task at each time step
            self.r_records[i].append(r.item())
        G_targets = [(E_G * (r ** self.a)).detach() for r in r_i]
        G_targets = torch.stack(G_targets, dim=0)
        G_w_i = torch.stack(G_w_i, dim=0)
        weight_loss = torch.sum(torch.abs(G_w_i - G_targets), dim=0)
        # Record the value of the weight update loss
        self.grad_loss_his.append(weight_loss.item())
        if torch.isnan(weight_loss):
            print("weight is nan")
            print(self.loss_weights)
        grads = torch.autograd.grad(weight_loss, inputs=self.loss_weights, retain_graph=False)
        with torch.no_grad():
            for param, grad in zip(self.loss_weights, grads[0]):
                param.data -= self.lr * grad  # manual update
        return sum(weighted_losses), weighted_losses, False


class GradNormLossAvgAugForm(GradNormLossAvgForm):

    def __init__(self, model, config):
        super(GradNormLossAvgAugForm, self).__init__(model, config)
        self.b = config["b_value"]

    def form(self, losses):
        """

        Args:
            losses(tuple(Tensor)): tuple of Tensors composing the loss functions of all tasks

        Returns:
            the sum of all tasks' losses as a (tensor) (1,), the Tensor (T,) of all weighted task losses, and whether backpropagation has been performed
        """
        weighted_losses = tuple([self.loss_weights[i].detach() * loss for i, loss in enumerate(losses)])
        # If this is the first time, record L0
        if self.cur_step == 0:
            self.init_history_loss(len(losses), losses)
        else:
            for i, loss in enumerate(losses):
                self.history_loss[i].append(loss.clone().detach())
        if self.cur_step % self.norm_step != 0:
            return sum(weighted_losses), weighted_losses, False
        assert len(losses) == self.loss_weights.shape[0], "the number of initial loss weights does not match the number of loss functions"
        # Compute the weighted values of all loss functions, used for return; the weights here are constants without gradient updates
        # Compute L_hat_all, the ratio of all losses to L0
        self.L0_list = [sum(q) / len(q) for q in self.history_loss]
        L_hat_all = torch.stack([loss / (self.L0_list[i] + self.eps) for i, loss in enumerate(losses)])
        E_L_hat = L_hat_all.mean() + self.eps
        r_all = L_hat_all / E_L_hat
        #(T,)
        grads_all = []
        for i, loss in enumerate(losses):
            # record weights
            self.weight_records[i].append(self.loss_weights[i].item())
            # Compute G and L
            grads = torch.autograd.grad(loss, inputs=self.share_layers, retain_graph=True)
            grads = torch.cat([g.view(-1) for g in grads], dim=0)
            grads_all.append(grads)
        grads_all = torch.stack(grads_all, dim=0)  #shape(T,N).
        L2_all = torch.norm(grads_all, p=2, dim=1)  #(T,)
        GW_all = L2_all.detach() * self.loss_weights  #(T)
        E_G = GW_all.mean()
        # Compute gamma
        q_r = r_all - 1  #(T)
        q_g = GW_all / (E_G + self.eps) - 1  #(T)
        pos_all = torch.sqrt(torch.clamp_min(q_r, 0) * torch.clamp_min(q_g, 0))
        neg_all = torch.sqrt(torch.clamp_min(-1 * q_r, 0) * torch.clamp_min(-1 * q_g, 0))
        gammas = (1 + pos_all) / (1 + neg_all)  #(T)
        # gammas = gammas / gammas.mean()
        gammas = gammas ** self.b

        # Compute the target value
        G_targets = E_G * gammas * (r_all ** self.a)
        G_targets = G_targets.detach()  #(T)
        weight_loss = torch.sum(torch.abs(GW_all - G_targets), dim=0)
        if torch.isnan(weight_loss):
            print("weight is nan")
            print(self.loss_weights)
        grads = torch.autograd.grad(weight_loss, inputs=self.loss_weights, retain_graph=False)
        with torch.no_grad():
            for param, grad in zip(self.loss_weights, grads[0]):
                param.data -= self.lr * grad  # manual update
        return sum(weighted_losses), weighted_losses, False


class GradNormLossFinalForm(GradNormLossAvgForm):

    def __init__(self, model, config):
        super(GradNormLossFinalForm, self).__init__(model, config)
        self.b = config["b_value"]

    def form(self, losses):
        """

        Args:
            losses(tuple(Tensor)): tuple of Tensors composing the loss functions of all tasks

        Returns:
            the sum of all tasks' losses as a (tensor) (1,), the Tensor (T,) of all weighted task losses, and whether backpropagation has been performed
        """
        weighted_losses = tuple([self.loss_weights[i].detach() * loss for i, loss in enumerate(losses)])
        # Compute the current EMA
        if self.cur_step == 0:  # if this is the first step, EMA equals the current loss
            for i in range(len(losses)):
                self.L0_list.append(losses[i].item())
                self.weight_records.append([])
                self.G_records.append([])
                self.loss_records.append([])
                self.r_records.append([])
        if self.cur_step % self.norm_step != 0:
            return sum(weighted_losses), weighted_losses, False
        assert len(losses) == self.loss_weights.shape[0], "the number of initial loss weights does not match the number of loss functions"
        # Compute the weighted values of all loss functions, used for return; the weights here are constants without gradient updates
        L_hat_all = torch.stack([loss / (self.L0_list[i] + self.eps) for i, loss in enumerate(losses)])
        E_L_hat = L_hat_all.mean() + self.eps
        r_all = L_hat_all / E_L_hat
        #(T,)
        grads_all = []
        for i, loss in enumerate(losses):
            # record weights
            self.weight_records[i].append(self.loss_weights[i].item())
            self.loss_records[i].append(loss.item())
            # Compute G and L
            grads = torch.autograd.grad(loss, inputs=self.share_layers, retain_graph=True)
            grads = torch.cat([g.view(-1) for g in grads], dim=0)
            grads_all.append(grads)
        grads_all = torch.stack(grads_all, dim=0)  #shape(T,N).
        L2_all = torch.norm(grads_all, p=2, dim=1)  #(T,)
        GW_all = L2_all.detach() * self.loss_weights  #(T)
        for i, g_i in enumerate(GW_all):  # record G_i for each task at each time step
            self.G_records[i].append(g_i.item())
            self.r_records[i].append(r_all[i].item())
        E_G = GW_all.mean()
        # Compute gamma
        q_r = r_all - 1  #(T)
        q_g = GW_all / (E_G + self.eps) - 1  #(T)
        pos_all = torch.sqrt(torch.clamp_min(q_r, 0) * torch.clamp_min(q_g, 0))
        neg_all = torch.sqrt(torch.clamp_min(-1 * q_r, 0) * torch.clamp_min(-1 * q_g, 0))
        gammas = (1 + pos_all) / (1 + neg_all)  #(T)
        # gammas = gammas / gammas.mean()
        gammas = gammas ** self.b

        # Compute the target value
        G_targets = E_G * gammas * (r_all ** self.a)
        G_targets = G_targets.detach()  #(T)
        weight_loss = torch.sum(torch.abs(GW_all - G_targets), dim=0)
        # Record the value of the weight update loss
        self.grad_loss_his.append(weight_loss.item())
        grads = torch.autograd.grad(weight_loss, inputs=self.loss_weights, retain_graph=False)
        with torch.no_grad():
            self.loss_weights.data -= self.lr * grads[0]
        return sum(weighted_losses), weighted_losses, False
