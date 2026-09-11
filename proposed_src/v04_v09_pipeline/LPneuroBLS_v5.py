import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import argparse
import math
import random
import pickle
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import global_mean_pool, GATConv
from torch.distributions import Categorical
from pyscipopt import Model as ScipModel

torch.set_float32_matmul_precision('high')

# ==========================================
# 0. LP Relaxation Solver & Batch Converter
# ==========================================
def solve_lp_relaxation(A, b):
    model = ScipModel("LP_Relaxation")
    model.hideOutput()
    num_vars = A.shape[1]
    x_vars = [model.addVar(vtype="C", lb=0.0, ub=1.0, name=f"x_{i}") for i in range(num_vars)]
    for j in range(A.shape[0]):
        row_expr = sum(A[j, i] * x_vars[i] for i in range(num_vars) if A[j, i] != 0)
        model.addCons(row_expr == b[j])
    model.setObjective(0, "minimize")
    model.optimize()
    if model.getStatus() == "optimal":
        return np.array([model.getVal(v) for v in x_vars], dtype=np.float32), True
    return np.full(num_vars, 0.5, dtype=np.float32), False

class BatchConverter:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch_data):
        list_xv, list_xc = [], []
        list_xlp_v, list_xlp_c = [], []
        list_ev2c, list_ec2v = [], []
        list_target, list_mask = [], []
        list_batch_idx_v, list_batch_idx_c = [], []

        offset_var, offset_check = 0, 0

        for i, (xv, xc, xlp_v, xlp_c, ev2c, ec2v, tgt, msk) in enumerate(batch_data):
            num_v, num_c = xv.size(0), xc.size(0)
            list_xv.append(xv)
            list_xc.append(xc)
            list_xlp_v.append(xlp_v)
            list_xlp_c.append(xlp_c)
            list_batch_idx_v.append(torch.full((num_v,), i, dtype=torch.long, device=self.device))
            list_batch_idx_c.append(torch.full((num_c,), i, dtype=torch.long, device=self.device))
            if tgt is not None:
                list_target.append(tgt)
            if msk is not None:
                list_mask.append(msk)
            cur_ev2c = ev2c.clone()
            cur_ev2c[0, :] += offset_var
            cur_ev2c[1, :] += offset_check
            list_ev2c.append(cur_ev2c)
            cur_ec2v = ec2v.clone()
            cur_ec2v[0, :] += offset_check
            cur_ec2v[1, :] += offset_var
            list_ec2v.append(cur_ec2v)
            offset_var += num_v
            offset_check += num_c

        return (
            torch.cat(list_xv, dim=0),
            torch.cat(list_xc, dim=0),
            torch.cat(list_xlp_v, dim=0),
            torch.cat(list_xlp_c, dim=0),
            torch.cat(list_ev2c, dim=1),
            torch.cat(list_ec2v, dim=1),
            torch.cat(list_target, dim=0) if list_target else None,
            torch.cat(list_mask, dim=0) if list_mask else None,
            torch.cat(list_batch_idx_v, dim=0),
            torch.cat(list_batch_idx_c, dim=0),
        )

# ==========================================
# 1. GNN Model (v4: adds feasibility_head)
# ==========================================
class BipartiteGNN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, noise_std=0.05):
        super(BipartiteGNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

        self.var_emb = nn.Embedding(3, self.hidden_dim)
        self.check_emb = nn.Embedding(5, self.hidden_dim)
        self.var_feat_proj = nn.Linear(self.hidden_dim + 1, self.hidden_dim)
        self.check_feat_proj = nn.Linear(self.hidden_dim + 1, self.hidden_dim)

        self.ln_var = nn.LayerNorm(self.hidden_dim)
        self.ln_check = nn.LayerNorm(self.hidden_dim)

        self.mlp_v2c = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.mlp_c2v = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))

        self.gat_v2c = GATConv((self.hidden_dim, self.hidden_dim), self.hidden_dim, add_self_loops=False)
        self.gat_c2v = GATConv((self.hidden_dim, self.hidden_dim), self.hidden_dim, add_self_loops=False)

        self.lstm_var = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_check = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.selection_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.assign_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        # Feasibility head: predicts P(instance is feasible) using global graph representation
        self.feasibility_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x_var, x_check, x_lp_var, x_lp_check, edge_v2c, edge_c2v, batch_v=None, batch_c=None):
        h_var_emb = self.var_emb(x_var)
        # x_check is 1-D (check_status values 0-4); use directly to avoid non-contiguous slice
        h_check_emb = self.check_emb(x_check)

        h_var = torch.cat([h_var_emb, x_lp_var.unsqueeze(-1)], dim=-1)
        h_var = self.var_feat_proj(h_var)
        h_check = torch.cat([h_check_emb, x_lp_check.unsqueeze(-1)], dim=-1)
        h_check = self.check_feat_proj(h_check)

        if self.training and getattr(self, 'noise_std', 0.0) > 0.0:
            h_var = h_var + torch.randn_like(h_var) * self.noise_std

        c_var = torch.zeros_like(h_var)
        c_check = torch.zeros_like(h_check)
        h_var_init = h_var
        h_check_init = h_check

        for _ in range(self.num_layers):
            msg_v = self.mlp_v2c(h_var)
            aggr_v2c = F.elu(self.gat_v2c((msg_v, h_check), edge_v2c)).contiguous()
            h_check, c_check = self.lstm_check(aggr_v2c, (h_check, c_check))
            h_check = self.ln_check(h_check)

            msg_c = self.mlp_c2v(h_check)
            aggr_c2v = F.elu(self.gat_c2v((msg_c, h_var), edge_c2v)).contiguous()
            h_var, c_var = self.lstm_var(aggr_c2v, (h_var, c_var))
            h_var = self.ln_var(h_var)

        h_var = h_var + h_var_init
        h_check = h_check + h_check_init

        h_var_final = torch.cat([h_var, x_lp_var.unsqueeze(-1)], dim=-1)
        select_logits = self.selection_head(h_var_final).squeeze(-1)
        assign_logits = self.assign_head(h_var_final)

        pool_var = global_mean_pool(h_var, batch_v) if batch_v is not None else torch.mean(h_var, dim=0, keepdim=True)
        pool_check = global_mean_pool(h_check, batch_c) if batch_c is not None else torch.mean(h_check, dim=0, keepdim=True)
        pool_cat = torch.cat([pool_var, pool_check], dim=1)
        value_logits = self.value_head(pool_cat)
        feas_logit = self.feasibility_head(pool_cat).squeeze(-1)  # (B,) or scalar

        return select_logits, assign_logits, value_logits, feas_logit

# ==========================================
# 2. Environment
# ==========================================
class FastBinaryEnv:
    def __init__(self, A, b, assignment=None, is_invalid=False):
        self.A = A
        self.b = b
        self.num_vars = A.shape[1]
        self.num_checks = A.shape[0]
        self.assignment = assignment if assignment is not None else np.full(self.num_vars, -1, dtype=np.int8)
        self.is_invalid = is_invalid

    def get_tensor_state(self, device):
        # Explicit int64 cast before CUDA transfer avoids potential int8 promotion issues
        x_var = torch.tensor(self.assignment.astype(np.int64) + 1, dtype=torch.long, device=device)
        temp_assign = np.maximum(self.assignment, 0)
        curr_sum = self.A.dot(temp_assign)
        diff = curr_sum - self.b
        check_status = np.full(self.num_checks, 2, dtype=np.int64)
        threshold = 2
        check_status[diff < -threshold] = 0
        check_status[(diff >= -threshold) & (diff < 0)] = 1
        check_status[(diff > 0) & (diff <= threshold)] = 3
        check_status[diff > threshold] = 4
        # Return 1-D contiguous array — model only needs check_status for the embedding
        x_check = torch.tensor(check_status, dtype=torch.long, device=device)
        return x_var, x_check

    def is_terminal(self):
        if self.is_invalid:
            return True
        return -1 not in self.assignment

    def get_accuracy(self):
        if self.is_invalid:
            return 0.0
        temp_assign = np.maximum(self.assignment, 0)
        is_solved = np.sum(np.abs(self.A.dot(temp_assign) - self.b)) == 0
        return 1.0 if is_solved else 0.0

    def get_reward(self):
        if self.is_invalid or not self.is_terminal():
            return 0.0
        return self.get_accuracy()

    def step_inplace(self, var_idx, val):
        self.assignment[var_idx] = val
        return self

    def propagate_constraints(self):
        if self.is_invalid:
            return self
        changed = True
        while changed and not self.is_terminal():
            changed = False
            temp_assign = np.maximum(self.assignment, 0)
            curr_sum = self.A.dot(temp_assign)
            remaining_b = self.b - curr_sum
            unassigned_bit = (self.assignment == -1).astype(np.int32)
            num_unassigned = self.A.dot(unassigned_bit)

            if np.any(remaining_b < 0) or np.any(num_unassigned < remaining_b):
                self.is_invalid = True
                return self

            active = num_unassigned > 0
            force_zero_rows = active & (remaining_b == 0)
            force_one_rows  = active & (remaining_b == num_unassigned)

            if not (force_zero_rows.any() or force_one_rows.any()):
                break

            unassigned = (self.assignment == -1)
            to_zero = (self.A[force_zero_rows] > 0).any(axis=0) & unassigned \
                      if force_zero_rows.any() else np.zeros(self.num_vars, dtype=bool)
            to_one  = (self.A[force_one_rows]  > 0).any(axis=0) & unassigned \
                      if force_one_rows.any()  else np.zeros(self.num_vars, dtype=bool)

            if (to_zero & to_one).any():
                self.is_invalid = True
                return self

            if to_zero.any():
                self.assignment[to_zero] = 0
                changed = True
            if to_one.any():
                self.assignment[to_one] = 1
                changed = True

        return self

    def apply_probing(self):
        if self.is_invalid or self.is_terminal():
            return self
        changed = True
        while changed and not self.is_terminal():
            changed = False
            unassigned_vars = np.where(self.assignment == -1)[0]
            for v in unassigned_vars:
                for val in [0, 1]:
                    env_try = FastBinaryEnv(self.A, self.b, self.assignment.copy(), self.is_invalid)
                    env_try.assignment[v] = val
                    env_try.propagate_constraints()
                    if env_try.is_invalid:
                        self.assignment[v] = 1 - val
                        self.propagate_constraints()
                        if self.is_invalid:
                            return self
                        changed = True
                        break
                if changed:
                    break
        return self

    def apply_local_search(self):
        if self.is_terminal() and self.get_accuracy() == 0.0 and not self.is_invalid:
            temp_assign = np.maximum(self.assignment, 0)
            diff = self.A.dot(temp_assign) - self.b
            violated_checks = np.where(diff != 0)[0]
            if len(violated_checks) == 0:
                return self
            involved_vars = set()
            for c_idx in violated_checks:
                involved_vars.update(np.where(self.A[c_idx] != 0)[0])
            for v in involved_vars:
                new_assign = temp_assign.copy()
                new_assign[v] = 1 - new_assign[v]
                new_diff = self.A.dot(new_assign) - self.b
                if np.sum(np.abs(new_diff)) == 0:
                    self.assignment = new_assign
                    self.is_invalid = False
                    return self
        return self

# ==========================================
# 3. MCTS
# ==========================================
class MCTSNode:
    def __init__(self, parent=None, prior_p=0.0, node_type='root', var_idx=None, action=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_p = prior_p
        self.node_type = node_type
        self.var_idx = var_idx
        self.action = action

    def value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, device, num_simulations=50, c_puct=1.5, training=True, dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.training = training
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.cache = {}

    def _softmax(self, logits):
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum() if exp.sum() > 0 else np.ones_like(logits) / len(logits)

    def run(self, env, edge_v2c, edge_c2v, x_lp_var, x_lp_check):
        if env.is_terminal():
            return None, None, None, None, None, None
        unassigned = np.where(env.assignment == -1)[0]
        if len(unassigned) == 0:
            return None, None, None, None, None, None

        state_key = tuple(env.assignment)
        if state_key not in self.cache:
            with torch.no_grad():
                x_v, x_c = env.get_tensor_state(self.device)
                select_logits, assign_logits, value_logit, _ = self.model(x_v, x_c, x_lp_var, x_lp_check, edge_v2c, edge_c2v)
                select_logits_np = select_logits.cpu().numpy()
                assign_logits_np = assign_logits.cpu().numpy()
                v_value = torch.sigmoid(value_logit).item()
                self.cache[state_key] = (select_logits_np, assign_logits_np, v_value)
        else:
            select_logits_np, assign_logits_np, _ = self.cache[state_key]
            if select_logits_np is None or assign_logits_np is None:
                with torch.no_grad():
                    x_v, x_c = env.get_tensor_state(self.device)
                    select_logits, assign_logits, value_logit, _ = self.model(x_v, x_c, x_lp_var, x_lp_check, edge_v2c, edge_c2v)
                    select_logits_np = select_logits.cpu().numpy()
                    assign_logits_np = assign_logits.cpu().numpy()
                    v_value = torch.sigmoid(value_logit).item()
                    self.cache[state_key] = (select_logits_np, assign_logits_np, v_value)

        var_scores = select_logits_np[unassigned]
        var_priors = self._softmax(var_scores)
        if self.training and len(unassigned) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(unassigned))
            var_priors = (1.0 - self.dirichlet_eps) * var_priors + self.dirichlet_eps * noise
        root = MCTSNode(prior_p=1.0, node_type='root')
        for idx, var in enumerate(unassigned):
            var_node = MCTSNode(parent=root, prior_p=var_priors[idx], node_type='var', var_idx=var)
            root.children[var] = var_node
            action_scores = assign_logits_np[var]
            action_priors = self._softmax(action_scores)
            for val in [0, 1]:
                var_node.children[val] = MCTSNode(parent=var_node, prior_p=action_priors[val], node_type='action', var_idx=var, action=val)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            while node.node_type != 'action':
                best_score = -float('inf')
                best_child = None
                for child in node.children.values():
                    u = self.c_puct * child.prior_p * math.sqrt(max(1, node.visit_count)) / (1 + child.visit_count)
                    score = child.value() + u
                    if score > best_score:
                        best_score = score
                        best_child = child
                node = best_child
                search_path.append(node)

            chosen_var = node.var_idx
            chosen_action = node.action
            curr_env = FastBinaryEnv(env.A, env.b, env.assignment.copy(), env.is_invalid)
            curr_env.step_inplace(chosen_var, chosen_action)
            curr_env.propagate_constraints()

            if curr_env.is_terminal():
                reward = curr_env.get_reward()
            else:
                rollout_env = FastBinaryEnv(curr_env.A, curr_env.b, curr_env.assignment.copy())
                for _ in range(3):
                    if rollout_env.is_terminal():
                        break
                    sk = tuple(rollout_env.assignment)
                    if sk not in self.cache or self.cache[sk][1] is None:
                        with torch.no_grad():
                            xv, xc = rollout_env.get_tensor_state(self.device)
                            sl, al, vl, _ = self.model(xv, xc, x_lp_var, x_lp_check, edge_v2c, edge_c2v)
                            self.cache[sk] = (sl.cpu().numpy(), al.cpu().numpy(), torch.sigmoid(vl).item())
                    _, assign_cached, _ = self.cache[sk]
                    remaining = np.where(rollout_env.assignment == -1)[0]
                    if len(remaining) == 0:
                        break
                    diffs = np.abs(assign_cached[remaining, 0] - assign_cached[remaining, 1])
                    best_var = remaining[np.argmax(diffs)]
                    best_val = int(np.argmax(assign_cached[best_var]))
                    rollout_env.step_inplace(best_var, best_val)
                    rollout_env.propagate_constraints()
                    rollout_env.apply_probing()

                if rollout_env.is_terminal():
                    reward = rollout_env.get_reward()
                else:
                    next_state_key = tuple(rollout_env.assignment)
                    if next_state_key not in self.cache:
                        with torch.no_grad():
                            xv, xc = rollout_env.get_tensor_state(self.device)
                            _, _, vl, _ = self.model(xv, xc, x_lp_var, x_lp_check, edge_v2c, edge_c2v)
                            self.cache[next_state_key] = (None, None, torch.sigmoid(vl).item())
                    reward = self.cache[next_state_key][2]

            for ancestor in reversed(search_path):
                ancestor.visit_count += 1
                ancestor.value_sum += reward

        var_visits = np.array([root.children[var].visit_count for var in unassigned], dtype=np.float32)
        if var_visits.sum() == 0:
            pi_var = np.ones_like(var_visits) / len(var_visits)
        else:
            pi_var = var_visits ** 2
            pi_var = pi_var / pi_var.sum()
        pi_var_full = np.zeros(select_logits_np.shape[0], dtype=np.float32)
        pi_var_full[unassigned] = pi_var

        chosen_var = np.random.choice(unassigned, p=pi_var)
        action_node = root.children[chosen_var].children
        action_visits = np.array([action_node[0].visit_count, action_node[1].visit_count], dtype=np.float32)
        if action_visits.sum() == 0:
            pi_act = np.array([0.5, 0.5], dtype=np.float32)
        else:
            pi_act = action_visits ** 2
            pi_act = pi_act / pi_act.sum()

        return pi_var_full, pi_act, chosen_var, np.array([action_node[0].prior_p, action_node[1].prior_p], dtype=np.float32), var_scores, assign_logits_np

# ==========================================
# 4. Trainer
# ==========================================
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(self.args.device)
        self.model = BipartiteGNN(hidden_dim=self.args.hidden_dim, num_layers=self.args.num_layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
        self.mcts = MCTS(self.model, self.device, num_simulations=args.num_simulations, training=True)
        self.batch_converter = BatchConverter(self.device)
        self.history = {
            'loss': [], 'policy_loss': [], 'value_loss': [], 'feas_loss': [],
            'rewards': [], 'solved': [], 'feas_acc': []
        }

        all_data = self.load_data(self.args.data_dir)
        feasible_data = [d for d in all_data if d[5] is True]
        infeasible_data = [d for d in all_data if d[5] is False]
        print(f"  Loaded: {len(all_data)} total  ({len(feasible_data)} feasible, {len(infeasible_data)} infeasible)")

        random.shuffle(feasible_data)
        split_idx = int(len(feasible_data) * 0.2)
        self.imitation_data = feasible_data[:split_idx]
        self.pretrain_data = feasible_data[split_idx:]
        self.rl_data = feasible_data[split_idx:] + infeasible_data

        now = datetime.now()
        self.save_dir = Path(self.args.save_dir) / (now.strftime("%y%m%d-%H%M") + "_LP_Hybrid_v5")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, dir_path):
        path_obj = Path(dir_path)
        if not path_obj.exists():
            return []
        files = sorted(path_obj.glob('*.json'))
        data = []
        print("💡 Caching LP Relaxation constraints...")
        for f in tqdm(files):
            with open(f, 'r') as fp:
                d = json.load(fp)
                A, b = np.array(d['A']), np.array(d['b'])
                gt_feasible = bool(d.get('feasible', True))
                x_gt = np.array(d['x']) if d.get('x') is not None else None
                x_lp, lp_is_feasible = solve_lp_relaxation(A, b)
                row_degrees = np.maximum(np.sum(A != 0, axis=1), 1)
                x_lp_check = (b - A.dot(x_lp)) / row_degrees
                x_lp_var_ts = torch.tensor(x_lp, dtype=torch.float, device=self.device)
                x_lp_check_ts = torch.tensor(x_lp_check, dtype=torch.float, device=self.device)
                data.append((A, b, x_gt, x_lp_var_ts, x_lp_check_ts, gt_feasible, lp_is_feasible))
        return data

    def matrix_to_graph(self, A):
        rows, cols = np.where(A == 1)
        ev = torch.tensor(np.array([cols, rows]), dtype=torch.long, device=self.device)
        ec = torch.tensor(np.array([rows, cols]), dtype=torch.long, device=self.device)
        return ev, ec

    def save_checkpoint(self, epoch, tag=""):
        path = self.save_dir / f"checkpoint_ep{epoch}{tag}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)

    def save_history(self, epoch, prefix=""):
        path = self.save_dir / f"training_history_{prefix}_ep{epoch}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(self.history, f)

    def plot_metrics(self, epoch, prefix=""):
        if len(self.history['loss']) == 0:
            return
        n_plots = 7
        figw = 4 * n_plots
        def plot_with_trendline(subplot_index, data, title):
            if len(data) == 0:
                return
            x_epochs = range(1, len(data) + 1)
            plt.subplot(1, n_plots, subplot_index)
            plt.plot(x_epochs, data, label='Data')
            if len(data) > 1:
                z = np.polyfit(x_epochs, data, 1)
                p = np.poly1d(z)
                plt.plot(x_epochs, p(x_epochs), "r--", label='Trend')
            plt.title(title)
        plt.figure(figsize=(figw, 4))
        plot_with_trendline(1, self.history['loss'], 'Total Loss')
        plot_with_trendline(2, self.history['policy_loss'], 'Policy Loss')
        plot_with_trendline(3, self.history['value_loss'], 'Value Loss')
        plot_with_trendline(4, self.history.get('feas_loss', []), 'Feas Loss')
        plot_with_trendline(5, self.history['rewards'], 'Avg Solved Rate')
        plot_with_trendline(6, self.history['solved'], 'Solved Count')
        plot_with_trendline(7, self.history.get('feas_acc', []), 'Feas Acc')
        save_path = self.save_dir / f'training_metrics_{prefix}_ep{epoch}.png'
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def pretrain(self):
        if self.args.pretrain_epochs <= 0:
            return
        print("\n=== Starting GNN Pre-training (Masked Variable Prediction) ===")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.001
        prev_noise = getattr(self.model, 'noise_std', 0.05)
        self.model.noise_std = 0.0
        self.model.train()
        valid_instances = self.pretrain_data  # feasible only
        if not valid_instances:
            return
        BATCH_SIZE = self.args.batch_size
        num_batches = int(np.ceil(len(valid_instances) / BATCH_SIZE))
        best_masked_acc = -1.0
        pretrain_patience_counter = 0
        for epoch in range(1, self.args.pretrain_epochs + 1):
            random.shuffle(valid_instances)
            total_loss, correct_vars, total_masked_count = 0.0, 0, 0
            pbar = tqdm(range(num_batches), desc=f"[Pretrain] Ep {epoch}")
            start_mask_ratio = 0.2
            end_mask_ratio = 0.8
            progress = min(1.0, (epoch - 1) / self.args.pretrain_minimum_epoch)
            current_max_ratio = start_mask_ratio + (end_mask_ratio - start_mask_ratio) * progress
            for b_idx in pbar:
                batch_data = []
                start = b_idx * BATCH_SIZE
                end = min((b_idx + 1) * BATCH_SIZE, len(valid_instances))
                for i in range(start, end):
                    A, b_vec, x_gt, x_lp_var, x_lp_check, _, _ = valid_instances[i]
                    num_vars = len(x_gt)
                    mask_ratio = random.random() * current_max_ratio
                    num_masked = max(1, int(num_vars * mask_ratio))
                    num_reveal = num_vars - num_masked
                    current_assignment = np.full(num_vars, -1, dtype=np.int8)
                    mask = np.ones(num_vars, dtype=bool)
                    if num_reveal > 0:
                        reveal_indices = np.random.choice(num_vars, num_reveal, replace=False)
                        current_assignment[reveal_indices] = x_gt[reveal_indices]
                        mask[reveal_indices] = False
                    env = FastBinaryEnv(A, b_vec, current_assignment)
                    env.propagate_constraints()
                    current_assignment = env.assignment
                    mask = (current_assignment == -1)
                    min_mask = max(1, int(0.05 * num_vars))
                    if mask.sum() < min_mask:
                        assigned_indices = np.where(current_assignment != -1)[0]
                        num_to_unassign = min_mask - int(mask.sum())
                        if len(assigned_indices) > 0:
                            num_to_unassign = min(num_to_unassign, len(assigned_indices))
                            to_unassign = np.random.choice(assigned_indices, num_to_unassign, replace=False)
                            for idx in to_unassign:
                                current_assignment[idx] = -1
                            env = FastBinaryEnv(A, b_vec, current_assignment)
                            env.propagate_constraints()
                            current_assignment = env.assignment
                            mask = (current_assignment == -1)
                    xv, xc = env.get_tensor_state(self.device)
                    ev2c, ec2v = self.matrix_to_graph(A)
                    tgt = torch.tensor(x_gt, dtype=torch.long, device=self.device)
                    msk = torch.tensor(mask, dtype=torch.bool, device=self.device)
                    batch_data.append((xv, xc, x_lp_var, x_lp_check, ev2c, ec2v, tgt, msk))
                if not batch_data:
                    continue
                bx_v, bx_c, b_xlpv, b_xlpc, bev2c, bec2v, b_tgt, b_msk, b_idx_v, b_idx_c = self.batch_converter(batch_data)
                self.optimizer.zero_grad()
                _, assign_logits, _, feas_logits = self.model(bx_v, bx_c, b_xlpv, b_xlpc, bev2c, bec2v, b_idx_v, b_idx_c)
                if b_msk.sum() > 0:
                    masked_logits = assign_logits[b_msk]
                    masked_labels = b_tgt[b_msk]
                    if masked_labels.numel() > 0:
                        class_counts = torch.bincount(masked_labels, minlength=2).float()
                        total = class_counts.sum()
                        class_weights = (total / (class_counts + 1e-6))
                        class_weights = class_weights / class_weights.sum() * 2.0
                        class_weights = class_weights.to(self.device)
                        ce_per = F.cross_entropy(masked_logits, masked_labels, reduction='none', weight=class_weights)
                        probs = F.softmax(masked_logits, dim=1)
                        pt = probs.gather(1, masked_labels.unsqueeze(1)).squeeze()
                        focal_factor = (1.0 - pt) ** 2.0
                        masked_loss = (ce_per * focal_factor).mean()
                    else:
                        masked_loss = F.cross_entropy(masked_logits, masked_labels)
                    # All pretrain instances are feasible → target=1.0; helps initialize feasibility head
                    feas_target = torch.ones(feas_logits.shape[0], dtype=torch.float32, device=self.device)
                    feas_loss = F.binary_cross_entropy_with_logits(feas_logits, feas_target)
                    masked_loss = masked_loss + 0.1 * feas_loss
                else:
                    continue
                if torch.isnan(masked_loss):
                    continue
                masked_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += masked_loss.item()
                with torch.no_grad():
                    preds = assign_logits.argmax(dim=1)
                    correct_vars += (preds[b_msk] == b_tgt[b_msk]).sum().item()
                    total_masked_count += b_msk.sum().item()
                if b_idx % 5 == 0:
                    curr_acc = 100 * correct_vars / total_masked_count if total_masked_count > 0 else 0
                    pbar.set_postfix({'Loss': f"{masked_loss.item():.3f}", 'MaskAcc': f"{curr_acc:.1f}%"})
            avg_acc = 100 * correct_vars / total_masked_count if total_masked_count > 0 else 0
            print(f"[GNN-Pretrain] Ep {epoch} | Loss: {total_loss / num_batches:.4f} | MaskAcc: {avg_acc:.2f}%")
            if avg_acc > best_masked_acc and epoch > self.args.pretrain_minimum_epoch:
                best_masked_acc = avg_acc
                pretrain_patience_counter = 0
            elif epoch > self.args.pretrain_minimum_epoch:
                pretrain_patience_counter += 1
            if pretrain_patience_counter >= self.args.pretrain_patience_limit:
                print("🛑 Early Stopping triggered for Pre-training.")
                break
        self.model.noise_std = prev_noise
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr
        torch.save(self.model.state_dict(), self.save_dir / "checkpoint_pretrained.pt")

    def run_imitation_learning(self, num_epochs):
        if num_epochs <= 0 or not self.imitation_data:
            return
        print("\n=== Starting Imitation Learning (Teacher Forcing) ===")
        prev_noise = getattr(self.model, 'noise_std', 0.05)
        self.model.noise_std = 0.0
        self.model.train()
        BATCH_SIZE = self.args.batch_size
        num_batches = int(np.ceil(len(self.imitation_data) / BATCH_SIZE))
        for epoch in range(1, num_epochs + 1):
            random.shuffle(self.imitation_data)
            epoch_loss = epoch_var_loss = epoch_act_loss = epoch_val_loss = 0.0
            valid_update_count = 0
            pbar = tqdm(range(num_batches), desc=f"[IL] Ep {epoch}")
            for b_idx in pbar:
                start = b_idx * BATCH_SIZE
                end = min((b_idx + 1) * BATCH_SIZE, len(self.imitation_data))
                states = []
                for i in range(start, end):
                    A, b_vec, x_gt, x_lp_var, x_lp_check, _, _ = self.imitation_data[i]
                    env = FastBinaryEnv(A, b_vec).propagate_constraints().apply_probing()
                    while not env.is_terminal():
                        unassigned = np.where(env.assignment == -1)[0]
                        if len(unassigned) == 0:
                            break
                        lp_vals = x_lp_var.cpu().numpy()
                        frac = np.abs(lp_vals[unassigned] - 0.5)
                        target_var = int(unassigned[np.argmin(frac)])
                        target_act = int(x_gt[target_var])
                        xv, xc = env.get_tensor_state(self.device)
                        ev2c, ec2v = self.matrix_to_graph(A)
                        states.append({
                            'xv': xv, 'xc': xc,
                            'xlp_v': x_lp_var, 'xlp_c': x_lp_check,
                            'ev2c': ev2c, 'ec2v': ec2v,
                            'target_var': target_var,
                            'target_act': target_act,
                        })
                        env.step_inplace(target_var, target_act)
                        env.propagate_constraints().apply_probing()
                if not states:
                    continue
                for chunk_start in range(0, len(states), self.args.batch_size):
                    chunk = states[chunk_start:chunk_start + self.args.batch_size]
                    batch_input = []
                    list_target_node_mask = []
                    list_act_target = []
                    list_val_target = []
                    total_node_offset = 0
                    for item in chunk:
                        num_nodes = item['xv'].size(0)
                        batch_input.append((item['xv'], item['xc'], item['xlp_v'], item['xlp_c'], item['ev2c'], item['ec2v'], None, None))
                        list_act_target.append(item['target_act'])
                        list_val_target.append([1.0])
                        list_target_node_mask.append(total_node_offset + item['target_var'])
                        total_node_offset += num_nodes
                    bx_v, bx_c, b_xlpv, b_xlpc, bev2c, bec2v, _, _, b_idx_v, b_idx_c = self.batch_converter(batch_input)
                    b_act_target = torch.tensor(list_act_target, dtype=torch.long, device=self.device)
                    b_val_target = torch.tensor(np.array(list_val_target), dtype=torch.float, device=self.device).view(-1)
                    self.optimizer.zero_grad()
                    select_logits, assign_logits, v_pred, _ = self.model(bx_v, bx_c, b_xlpv, b_xlpc, bev2c, bec2v, b_idx_v, b_idx_c)
                    relevant_assign_logits = assign_logits[list_target_node_mask]
                    action_loss = F.cross_entropy(relevant_assign_logits, b_act_target)
                    var_loss = 0.0
                    offset = 0
                    for item in chunk:
                        num_nodes = item['xv'].size(0)
                        logits = select_logits[offset:offset + num_nodes]
                        mask = (bx_v[offset:offset + num_nodes] == 0)
                        masked_logits = logits.masked_fill(~mask, -1e9)
                        log_probs = F.log_softmax(masked_logits, dim=0)
                        var_loss = var_loss - log_probs[item['target_var']]
                        offset += num_nodes
                    var_loss = var_loss / len(chunk)
                    value_loss = F.mse_loss(v_pred.view(-1), b_val_target)
                    loss = var_loss + action_loss + (2.0 * value_loss)
                    if not torch.isnan(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        epoch_loss += loss.item()
                        epoch_var_loss += var_loss.item()
                        epoch_act_loss += action_loss.item()
                        epoch_val_loss += value_loss.item()
                        valid_update_count += 1
                pbar.set_postfix({'Loss': f"{epoch_loss / max(1, valid_update_count):.3f}", 'Updates': valid_update_count})
            if valid_update_count > 0:
                print(f"[IL] Ep {epoch} | Loss: {epoch_loss / valid_update_count:.4f} | VarLoss: {epoch_var_loss / valid_update_count:.4f} | ActLoss: {epoch_act_loss / valid_update_count:.4f} | ValLoss: {epoch_val_loss / valid_update_count:.4f}")
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(epoch, tag="_IL")
                self.save_history(epoch, prefix="IL")
                self.plot_metrics(epoch, prefix="IL")
        self.model.noise_std = prev_noise

    def _run_mcts_phase(self, num_epochs, is_il_phase):
        phase_name = "Imitation Learning" if is_il_phase else "Reinforcement Learning"
        phase_prefix = phase_name[:2].upper()
        phase_data = self.imitation_data if is_il_phase else self.rl_data
        MCTS_INSTANCE_BATCH = 64
        NETWORK_BATCH_SIZE = self.args.batch_size
        prev_noise = getattr(self.model, 'noise_std', 0.05)
        if not is_il_phase:
            self.model.noise_std = 0.01
        gamma = 0.97
        for epoch in range(1, num_epochs + 1):
            self.state_visits = {}
            random.shuffle(phase_data)
            epoch_data = phase_data[:self.args.samples_per_epoch]
            num_batches = int(np.ceil(len(epoch_data) / MCTS_INSTANCE_BATCH))
            epoch_loss = epoch_p_loss = epoch_v_loss = epoch_feas_loss = 0.0
            epoch_acc = 0.0
            epoch_solved = 0
            epoch_feas_correct = epoch_feas_total = 0
            valid_update_count = 0
            feasible_batch_count = 0
            pbar = tqdm(range(num_batches), desc=f"[{phase_name}] Ep {epoch}")

            for b_idx in pbar:
                start = b_idx * MCTS_INSTANCE_BATCH
                end = min((b_idx + 1) * MCTS_INSTANCE_BATCH, len(epoch_data))
                current_batch_data = epoch_data[start:end]

                collected_trajectories = []
                feas_roots = []   # (root_state, gt_feasible) per instance
                batch_acc = []
                batch_feas_correct = []

                for A, b, x_gt, x_lp_var, x_lp_check, gt_feasible, _ in current_batch_data:
                    env = FastBinaryEnv(A, b)
                    edge_v2c, edge_c2v = self.matrix_to_graph(A)
                    current_env = env.propagate_constraints().apply_probing()

                    # Feasibility prediction at root (no_grad for accuracy tracking)
                    with torch.no_grad():
                        xv_root, xc_root = current_env.get_tensor_state(self.device)
                        _, _, _, feas_logit_root = self.model(
                            xv_root, xc_root, x_lp_var, x_lp_check, edge_v2c, edge_c2v
                        )
                        pred_feas = (torch.sigmoid(feas_logit_root).item() >= 0.5)
                    batch_feas_correct.append(pred_feas == gt_feasible)

                    # Store root state for feasibility BCE gradient update
                    feas_roots.append({
                        'xv': xv_root, 'xc': xc_root,
                        'xlp_v': x_lp_var, 'xlp_c': x_lp_check,
                        'ev2c': edge_v2c, 'ec2v': edge_c2v,
                        'gt_feasible': float(gt_feasible)
                    })

                    if not gt_feasible:
                        # Infeasible: skip MCTS, feasibility BCE is the only loss
                        self.mcts.cache.clear()
                        continue

                    # Feasible: run MCTS trajectory
                    trajectory = []
                    while True:
                        pi_var, pi_act, tgt_var, prior_act, var_scores, assign_logits_np = self.mcts.run(
                            current_env, edge_v2c, edge_c2v, x_lp_var, x_lp_check
                        )
                        if pi_var is None:
                            break
                        if is_il_phase and x_gt is not None:
                            action = int(x_gt[tgt_var])
                            target_act = np.array([1.0, 0.0] if action == 0 else [0.0, 1.0], dtype=np.float32)
                        else:
                            target_act = pi_act
                            action = int(np.random.choice([0, 1], p=pi_act))
                        trajectory.append((current_env.assignment.copy(), tgt_var, target_act, pi_var, x_gt is not None))
                        current_env.step_inplace(tgt_var, action)
                        current_env.propagate_constraints().apply_probing()

                    current_env = current_env.apply_local_search()
                    self.mcts.cache.clear()
                    final_acc = current_env.get_accuracy()
                    batch_acc.append(final_acc)
                    if final_acc == 1.0:
                        epoch_solved += 1

                    if not is_il_phase and final_acc < 1.0:
                        temp_final = np.maximum(current_env.assignment, 0)
                        sat_ratio = float(np.sum(A.dot(temp_final) == b)) / A.shape[0]
                        partial_credit = sat_ratio * 0.3
                    else:
                        partial_credit = 0.0

                    for step_idx, (state, target_var, target_act, pi_var, has_gt) in enumerate(trajectory):
                        steps_to_end = len(trajectory) - step_idx - 1
                        binary_reward = float(final_acc == 1.0)
                        if not is_il_phase:
                            state_key = tuple(state)
                            self.state_visits[state_key] = self.state_visits.get(state_key, 0) + 1
                            visit_count = self.state_visits[state_key]
                            intrinsic_reward = 0.2 * math.sqrt(1.0 / (visit_count + 1.0)) if binary_reward > 0 else 0.0
                        else:
                            intrinsic_reward = 0.0
                        final_value_target = np.clip(
                            binary_reward * (gamma ** steps_to_end) + intrinsic_reward + partial_credit, 0.0, 1.0
                        )
                        temp_env = FastBinaryEnv(A, b, state)
                        xv, xc = temp_env.get_tensor_state(self.device)
                        collected_trajectories.append({
                            'xv': xv, 'xc': xc,
                            'xlp_v': x_lp_var, 'xlp_c': x_lp_check,
                            'ev2c': edge_v2c, 'ec2v': edge_c2v,
                            'target_var': target_var,
                            'target_act': target_act,
                            'pi_var': pi_var,
                            'final_value_target': final_value_target,
                        })

                # ── Policy / Value gradient update ──────────────────────────────
                if collected_trajectories:
                    random.shuffle(collected_trajectories)
                    for chunk_start in range(0, len(collected_trajectories), NETWORK_BATCH_SIZE):
                        chunk = collected_trajectories[chunk_start:chunk_start + NETWORK_BATCH_SIZE]
                        batch_input, list_act_target, list_val_target = [], [], []
                        list_target_node_mask = []
                        total_node_offset = 0
                        for sample_idx, item in enumerate(chunk):
                            num_nodes = item['xv'].size(0)
                            batch_input.append((item['xv'], item['xc'], item['xlp_v'], item['xlp_c'], item['ev2c'], item['ec2v'], None, None))
                            list_act_target.append(item['target_act'])
                            list_val_target.append([item['final_value_target']])
                            list_target_node_mask.append(total_node_offset + item['target_var'])
                            total_node_offset += num_nodes
                        bx_v, bx_c, b_xlpv, b_xlpc, bev2c, bec2v, _, _, b_idx_v, b_idx_c = self.batch_converter(batch_input)
                        b_act_target = torch.tensor(np.array(list_act_target), dtype=torch.float, device=self.device)
                        b_val_target = torch.tensor(np.array(list_val_target), dtype=torch.float, device=self.device).view(-1)
                        self.optimizer.zero_grad()
                        select_logits, assign_logits, v_pred, _ = self.model(bx_v, bx_c, b_xlpv, b_xlpc, bev2c, bec2v, b_idx_v, b_idx_c)
                        relevant_assign_logits = assign_logits[list_target_node_mask]
                        batch_size = len(chunk)
                        num_vars = bx_v.size(0) // batch_size
                        assert bx_v.size(0) == batch_size * num_vars, "Batch variable size not uniform"
                        select_logits_reshaped = select_logits.view(batch_size, num_vars)
                        bx_v_reshaped = bx_v.view(batch_size, num_vars)
                        unassigned_mask = (bx_v_reshaped == 0)
                        masked_logits_sel = select_logits_reshaped.masked_fill(~unassigned_mask, -1e9)
                        select_log_probs = F.log_softmax(masked_logits_sel, dim=1)
                        # AlphaZero-style policy target: cross-entropy against the full MCTS visit distribution
                        b_pi_var = torch.tensor(np.stack([item['pi_var'] for item in chunk]), dtype=torch.float, device=self.device)
                        var_ce = -(b_pi_var * select_log_probs).sum(dim=1)
                        act_log_probs = F.log_softmax(relevant_assign_logits, dim=1)
                        act_ce = -(b_act_target * act_log_probs).sum(dim=1)
                        v_loss = F.mse_loss(v_pred.view(-1), b_val_target)
                        dist = Categorical(logits=relevant_assign_logits)
                        ent = dist.entropy().mean()
                        p_loss = (var_ce + act_ce).mean()
                        loss = p_loss + (2.0 * v_loss) - (self.args.entropy_coef * ent)
                        if not torch.isnan(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            epoch_loss += loss.item()
                            epoch_v_loss += v_loss.item()
                            epoch_p_loss += p_loss.item()
                            valid_update_count += 1

                # ── Feasibility BCE gradient update ─────────────────────────────
                if feas_roots:
                    random.shuffle(feas_roots)
                    feas_update_count = 0
                    for chunk_start in range(0, len(feas_roots), NETWORK_BATCH_SIZE):
                        chunk_f = feas_roots[chunk_start:chunk_start + NETWORK_BATCH_SIZE]
                        batch_input_f = [
                            (item['xv'], item['xc'], item['xlp_v'], item['xlp_c'], item['ev2c'], item['ec2v'], None, None)
                            for item in chunk_f
                        ]
                        feas_targets = torch.tensor(
                            [item['gt_feasible'] for item in chunk_f], dtype=torch.float32, device=self.device
                        )
                        bxv_f, bxc_f, bxlpv_f, bxlpc_f, bev2c_f, bec2v_f, _, _, bidxv_f, bidxc_f = \
                            self.batch_converter(batch_input_f)
                        self.optimizer.zero_grad()
                        _, _, _, feas_logits = self.model(
                            bxv_f, bxc_f, bxlpv_f, bxlpc_f, bev2c_f, bec2v_f, bidxv_f, bidxc_f
                        )
                        feas_loss = F.binary_cross_entropy_with_logits(feas_logits, feas_targets)
                        if not torch.isnan(feas_loss):
                            feas_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            epoch_feas_loss += feas_loss.item()
                            feas_update_count += 1

                # ── Per-batch metrics ────────────────────────────────────────────
                epoch_feas_correct += sum(batch_feas_correct)
                epoch_feas_total += len(batch_feas_correct)
                if batch_acc:
                    epoch_acc += np.mean(batch_acc)
                    feasible_batch_count += 1

                disp_loss = epoch_loss / max(1, valid_update_count)
                disp_acc = epoch_acc / max(1, feasible_batch_count)
                disp_feas = epoch_feas_correct / max(1, epoch_feas_total)
                pbar.set_postfix({
                    'Loss': f"{disp_loss:.3f}",
                    'SolvedRate': f"{disp_acc:.2f}",
                    'FeasAcc': f"{disp_feas:.2f}",
                    'Solved': f"{epoch_solved}"
                })

            # ── Epoch summary ────────────────────────────────────────────────────
            if valid_update_count > 0:
                self.history['loss'].append(epoch_loss / valid_update_count)
                self.history['policy_loss'].append(epoch_p_loss / valid_update_count)
                self.history['value_loss'].append(epoch_v_loss / valid_update_count)
            if epoch_feas_loss > 0:
                self.history['feas_loss'].append(epoch_feas_loss / max(1, num_batches))
            self.history['rewards'].append(epoch_acc / max(1, feasible_batch_count))
            self.history['solved'].append(epoch_solved)
            self.history['feas_acc'].append(epoch_feas_correct / max(1, epoch_feas_total))

            epoch_feas_acc_pct = 100.0 * epoch_feas_correct / max(1, epoch_feas_total)
            print(
                f"[{phase_name}] Ep {epoch} | "
                f"Loss: {epoch_loss / max(1, valid_update_count):.4f} | "
                f"PL: {epoch_p_loss / max(1, valid_update_count):.4f} | "
                f"VL: {epoch_v_loss / max(1, valid_update_count):.4f} | "
                f"FL: {epoch_feas_loss / max(1, num_batches):.4f} | "
                f"SolvedRate: {epoch_acc / max(1, feasible_batch_count):.3f} | "
                f"FeasAcc: {epoch_feas_acc_pct:.1f}% | "
                f"Solved: {epoch_solved}"
            )

            if not is_il_phase:
                self.scheduler.step()
            if epoch % self.args.save_interval == 0:
                tag = f"_{phase_prefix}_FA{epoch_feas_acc_pct:.0f}_RE{100*epoch_acc/max(1,feasible_batch_count):.0f}"
                self.save_checkpoint(epoch, tag=tag)
                self.save_history(epoch, prefix=phase_prefix)
                self.plot_metrics(epoch, prefix=phase_prefix)
        self.model.noise_std = prev_noise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./train_instances_dns_10x25_10000')
    parser.add_argument('--save_dir', type=str, default='./runs')
    parser.add_argument('--pretrain_epochs', type=int, default=200)
    parser.add_argument('--pretrain_minimum_epoch', type=int, default=20)
    parser.add_argument('--pretrain_patience_limit', type=int, default=30)
    parser.add_argument('--il_epochs', type=int, default=0)
    parser.add_argument('--rl_epochs', type=int, default=10000)
    parser.add_argument('--samples_per_epoch', type=int, default=512)
    parser.add_argument('--num_simulations', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_decay_step', type=int, default=30)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--init_from', type=str, default=None, help='Warm-start model weights (e.g. a pretrained checkpoint) before any phase')
    args = parser.parse_args()
    trainer = Trainer(args)
    if args.init_from:
        state = torch.load(args.init_from, map_location=trainer.device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        trainer.model.load_state_dict(state)
        print(f"Warm-started weights from {args.init_from}")
    trainer.pretrain()
    if args.il_epochs > 0:
        trainer.run_imitation_learning(args.il_epochs)
    if args.rl_epochs > 0:
        trainer._run_mcts_phase(args.rl_epochs, is_il_phase=False)

if __name__ == '__main__':
    main()
