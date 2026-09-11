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
from torch_geometric.utils import softmax as segment_softmax
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

def kernel_pump_stats(x0, A, b, A_pinv, max_iter=200, seed=0):
    # Alternating projection between rounding and the affine subspace {x: Ax=b};
    # returns (solution|None, solved, stats[4]) where stats summarize the failure
    # trajectory: final/min normalized residual, mean fractionality, cycle rate
    rng = np.random.default_rng(seed)
    Af = A.astype(np.float64)
    bf = b.astype(np.float64)
    m_rows = A.shape[0]
    x = np.clip(x0.astype(np.float64), 0.0, 1.0)
    prev = None
    min_res = np.inf
    frac_sum = 0.0
    cycles = 0
    it = 0
    res = np.inf
    for it in range(1, max_iter + 1):
        x_r = np.round(np.clip(x, 0.0, 1.0))
        res = np.abs(Af.dot(x_r) - bf).sum()
        min_res = min(min_res, res)
        if res == 0:
            return x_r.astype(np.int8), True, np.array([0.0, 0.0, 0.0, it / max_iter], dtype=np.float32)
        if prev is not None and np.array_equal(x_r, prev):
            cycles += 1
            flip = rng.choice(x_r.shape[0], size=max(1, x_r.shape[0] // 5), replace=False)
            x_r[flip] = 1.0 - x_r[flip]
        prev = x_r.copy()
        x = x_r - A_pinv.dot(Af.dot(x_r) - bf)
        frac_sum += np.abs(x - np.round(x)).mean()
    stats = np.array([
        res / m_rows,
        min_res / m_rows,
        frac_sum / max(1, it),
        cycles / max(1, it),
    ], dtype=np.float32)
    return None, False, stats


def policy_dfs(A, b, root_assignment, var_order, val_order, node_budget=20000):
    # Complete backtracking from the propagated root with policy-given ordering;
    # exhausting the tree without a solution proves infeasibility
    nodes = 0
    stack = [(root_assignment.copy(), 0)]
    while stack:
        assignment, _ = stack.pop()
        nodes += 1
        if nodes > node_budget:
            return None, 'budget', nodes
        env = FastBinaryEnv(A, b, assignment.copy())
        env.propagate_constraints()
        if env.is_invalid:
            continue
        if env.is_terminal():
            if env.get_accuracy() == 1.0:
                return env.assignment.copy(), 'solved', nodes
            continue
        unassigned = np.where(env.assignment == -1)[0]
        branch_var = -1
        for v in var_order:
            if env.assignment[v] == -1:
                branch_var = v
                break
        if branch_var == -1:
            branch_var = int(unassigned[0])
        first = int(val_order[branch_var])
        for val in (1 - first, first):
            child = env.assignment.copy()
            child[branch_var] = val
            stack.append((child, 0))
    return None, 'infeasible', nodes


def complement_flip(A, b, x_gt, x_lp):
    # Complementation symmetry x -> 1-x maps (A, b) to (A, rowsum - b) with a
    # bijection between solution sets: feasibility and difficulty are preserved
    # exactly while every b/LP input the network sees changes
    rowsum = A.sum(axis=1).astype(b.dtype)
    b2 = rowsum - b
    x_gt2 = (1 - x_gt) if x_gt is not None else None
    x_lp2 = 1.0 - x_lp
    return b2, x_gt2, x_lp2


def gen_planted_instance(m, n, rng):
    # Fresh feasible instance (market-split profile): feasibility is guaranteed
    # by construction, so on-the-fly generation needs no solver certification
    while True:
        A = (rng.random((m, n)) < 0.5).astype(np.int8)
        if A.sum(axis=1).min() >= 2 and A.sum(axis=0).min() >= 1:
            break
    x = np.zeros(n, dtype=np.int8)
    x[rng.choice(n, n // 2, replace=False)] = 1
    b = A.astype(np.int32).dot(x.astype(np.int32))
    return A, b, x


class BatchConverter:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch_data):
        list_xv, list_xc = [], []
        list_xlp_v, list_xlp_c, list_cx = [], [], []
        list_ev2c, list_ec2v = [], []
        list_target, list_mask = [], []
        list_batch_idx_v, list_batch_idx_c = [], []

        offset_var, offset_check = 0, 0

        for i, (xv, xc, xlp_v, xlp_c, cx, ev2c, ec2v, tgt, msk) in enumerate(batch_data):
            num_v, num_c = xv.size(0), xc.size(0)
            list_xv.append(xv)
            list_xc.append(xc)
            list_xlp_v.append(xlp_v)
            list_xlp_c.append(xlp_c)
            list_cx.append(cx)
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
            torch.cat(list_cx, dim=0),
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
        self.var_feat_proj = nn.Linear(self.hidden_dim + 2, self.hidden_dim)
        self.check_feat_proj = nn.Linear(self.hidden_dim + 4, self.hidden_dim)

        self.ln_var = nn.LayerNorm(self.hidden_dim)
        self.ln_check = nn.LayerNorm(self.hidden_dim)

        self.mlp_v2c = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.mlp_c2v = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))

        self.gat_v2c = GATConv((self.hidden_dim, self.hidden_dim), self.hidden_dim, add_self_loops=False)
        self.gat_c2v = GATConv((self.hidden_dim, self.hidden_dim), self.hidden_dim, add_self_loops=False)

        self.lstm_var = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_check = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.selection_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.assign_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        # Feasibility head: global graph representation + pump-failure stats (4)
        # + assignment-marginal summary stats (2)
        self.feasibility_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 6, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x_var, x_check, x_lp_var, x_lp_check, edge_v2c, edge_c2v, batch_v=None, batch_c=None, pump_feats=None, check_extra=None, mp_iters=None):
        h_var_emb = self.var_emb(x_var)
        # x_check is 1-D (check_status values 0-4); use directly to avoid non-contiguous slice
        h_check_emb = self.check_emb(x_check)

        if check_extra is None:
            check_extra = torch.zeros(x_check.size(0), 3, device=x_check.device)
        h_var = torch.cat([h_var_emb, x_lp_var], dim=-1)
        h_var = self.var_feat_proj(h_var)
        h_check = torch.cat([h_check_emb, x_lp_check.unsqueeze(-1), check_extra], dim=-1)
        h_check = self.check_feat_proj(h_check)

        if self.training and getattr(self, 'noise_std', 0.0) > 0.0:
            h_var = h_var + torch.randn_like(h_var) * self.noise_std

        c_var = torch.zeros_like(h_var)
        c_check = torch.zeros_like(h_check)
        h_var_init = h_var
        h_check_init = h_check

        for _ in range(mp_iters if mp_iters is not None else self.num_layers):
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

        h_var_final = torch.cat([h_var, x_lp_var], dim=-1)
        select_logits = self.selection_head(h_var_final).squeeze(-1)
        assign_logits = self.assign_head(h_var_final)

        pool_var = global_mean_pool(h_var, batch_v) if batch_v is not None else torch.mean(h_var, dim=0, keepdim=True)
        pool_check = global_mean_pool(h_check, batch_c) if batch_c is not None else torch.mean(h_check, dim=0, keepdim=True)
        pool_cat = torch.cat([pool_var, pool_check], dim=1)
        value_logits = self.value_head(pool_cat)

        # Marginal summary stats: mean confidence and mean entropy of the
        # per-variable assignment marginals (inconsistent marginals signal infeasibility)
        marg = F.softmax(assign_logits, dim=-1)[:, 1]
        conf = (marg - 0.5).abs().unsqueeze(-1)
        mc = marg.clamp(1e-6, 1.0 - 1e-6)
        ent = -(mc.log() * mc + (1.0 - mc).log() * (1.0 - mc)).unsqueeze(-1)
        if batch_v is not None:
            m_conf = global_mean_pool(conf, batch_v)
            m_ent = global_mean_pool(ent, batch_v)
        else:
            m_conf = conf.mean(dim=0, keepdim=True)
            m_ent = ent.mean(dim=0, keepdim=True)
        if pump_feats is None:
            pump_feats = torch.zeros(pool_cat.size(0), 4, device=pool_cat.device)
        feas_in = torch.cat([pool_cat, m_conf, m_ent, pump_feats], dim=1)
        feas_logit = self.feasibility_head(feas_in).squeeze(-1)  # (B,) or scalar

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
        # Row-degree-normalized residual keeps the bucket semantics invariant
        # across instance sizes (0.16 matches the old absolute threshold 2 at 10x25)
        row_deg = np.maximum(self.A.sum(axis=1), 1)
        diff = (curr_sum - self.b) / row_deg
        check_status = np.full(self.num_checks, 2, dtype=np.int64)
        threshold = 0.16
        check_status[diff < -threshold] = 0
        check_status[(diff >= -threshold) & (diff < 0)] = 1
        check_status[(diff > 0) & (diff <= threshold)] = 3
        check_status[diff > threshold] = 4
        # State-dependent, size-invariant check features (aligned with propagation)
        unassigned_bit = (self.assignment == -1).astype(np.float64)
        un_frac = self.A.dot(unassigned_bit) / row_deg
        slack = un_frac + diff
        extra = np.stack([diff, un_frac, slack], axis=-1)
        x_check = torch.tensor(check_status, dtype=torch.long, device=device)
        x_extra = torch.tensor(extra, dtype=torch.float, device=device)
        return x_var, x_check, x_extra

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
        # Dense terminal reward: 1.0 for an exact solution, else a shaped credit
        # from the satisfied-constraint ratio (keeps MCTS backups informative on
        # hard instances where exact solutions are rare)
        if self.is_invalid or not self.is_terminal():
            return 0.0
        temp = np.maximum(self.assignment, 0)
        sat = float(np.sum(self.A.dot(temp) == self.b)) / self.num_checks
        return 1.0 if sat == 1.0 else 0.5 * sat * sat

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
    def __init__(self, model, device, num_simulations=50, c_puct=1.5, training=True, dirichlet_alpha=0.3, dirichlet_eps=0.25, mp_iters=None):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.training = training
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.mp_iters = mp_iters
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
                x_v, x_c, x_e = env.get_tensor_state(self.device)
                select_logits, assign_logits, value_logit, _ = self.model(x_v, x_c, x_lp_var, x_lp_check, edge_v2c, edge_c2v, check_extra=x_e, mp_iters=self.mp_iters)
                select_logits_np = select_logits.cpu().numpy()
                assign_logits_np = assign_logits.cpu().numpy()
                v_value = torch.sigmoid(value_logit).item()
                self.cache[state_key] = (select_logits_np, assign_logits_np, v_value)
        else:
            select_logits_np, assign_logits_np, _ = self.cache[state_key]
            if select_logits_np is None or assign_logits_np is None:
                with torch.no_grad():
                    x_v, x_c, x_e = env.get_tensor_state(self.device)
                    select_logits, assign_logits, value_logit, _ = self.model(x_v, x_c, x_lp_var, x_lp_check, edge_v2c, edge_c2v, check_extra=x_e, mp_iters=self.mp_iters)
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
                            xv, xc, xe = rollout_env.get_tensor_state(self.device)
                            sl, al, vl, _ = self.model(xv, xc, x_lp_var, x_lp_check, edge_v2c, edge_c2v, check_extra=xe, mp_iters=self.mp_iters)
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
                            xv, xc, xe = rollout_env.get_tensor_state(self.device)
                            _, _, vl, _ = self.model(xv, xc, x_lp_var, x_lp_check, edge_v2c, edge_c2v, check_extra=xe, mp_iters=self.mp_iters)
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
            'rewards': [], 'solved': [], 'feas_acc': [],
            'grad_norm': [], 'sel_entropy': []
        }

        self.rng = np.random.default_rng(self.args.seed)
        self.infeasible_pool = self.load_infeasible(self.args.data_dir)
        self.sizes = sorted({d['A'].shape for d in self.infeasible_pool}) or [(10, 25)]
        self.feat_cache = {}
        self.imitation_data = []
        print(f"  Infeasible pool: {len(self.infeasible_pool)}  |  sizes: {self.sizes}  |  feasible: fresh-generated per epoch")

        now = datetime.now()
        self.save_dir = Path(self.args.save_dir) / (now.strftime("%y%m%d-%H%M") + "_LP_Hybrid_v7")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def load_infeasible(self, dir_paths):
        pool = []
        for dir_path in dir_paths:
            path_obj = Path(dir_path)
            if not path_obj.exists():
                continue
            for f in sorted(path_obj.glob('*.json')):
                with open(f, 'r') as fp:
                    d = json.load(fp)
                if bool(d.get('feasible', True)):
                    continue
                pool.append({'A': np.array(d['A'], dtype=np.int8), 'b': np.array(d['b'], dtype=np.int32)})
        return pool

    def featurize(self, A, b, x_gt, gt_feasible):
        # LP relaxation, kernel-pump failure statistics, and graph tensors for
        # one instance; fresh feasible instances pass through here once and are
        # discarded after the epoch
        x_lp, lp_ok = solve_lp_relaxation(A, b)
        A_pinv = np.linalg.pinv(A.astype(np.float64))
        _, pumped, pstats = kernel_pump_stats(x_lp, A, b, A_pinv)
        row_deg = np.maximum(np.sum(A != 0, axis=1), 1)
        x_lp_check = (b - A.dot(x_lp)) / row_deg
        ev2c, ec2v = self.matrix_to_graph(A)
        col_deg = (A != 0).sum(axis=0) / max(1, A.shape[0])
        xlp_v_arr = np.stack([x_lp, col_deg], axis=-1)
        return {
            'A': A, 'b': b, 'x_gt': x_gt, 'gt_feasible': gt_feasible,
            'xlp_v': torch.tensor(xlp_v_arr, dtype=torch.float, device=self.device),
            'xlp_c': torch.tensor(x_lp_check, dtype=torch.float, device=self.device),
            'pump': torch.tensor(pstats, dtype=torch.float, device=self.device),
            'pump_solved': pumped, 'lp_ok': lp_ok,
            'ev2c': ev2c, 'ec2v': ec2v,
        }

    def sample_fresh_feasible(self, size=None):
        m, n = size if size is not None else self.sizes[int(self.rng.integers(len(self.sizes)))]
        A, b, x = gen_planted_instance(m, n, self.rng)
        return self.featurize(A, b, x, True)

    def sample_pool_infeasible(self):
        # Fresh certified infeasible for cheap sizes (prevents the head from
        # memorizing a fixed pool); pooled + complement view for expensive sizes
        m_, n_ = self.sizes[int(self.rng.integers(len(self.sizes)))]
        if m_ <= 20:
            from generate_dns_marketsplit import gen_infeasible
            A, b = gen_infeasible(m_, n_, self.rng, time_limit=5.0, max_tries=10)
            if A is not None:
                return self.featurize(A, b, None, False)
        pool_idx = [i for i, d in enumerate(self.infeasible_pool) if d['A'].shape == (m_, n_)]
        if not pool_idx:
            pool_idx = list(range(len(self.infeasible_pool)))
        idx = int(pool_idx[int(self.rng.integers(len(pool_idx)))])
        comp = bool(self.rng.random() < 0.5)
        key = (idx, comp)
        if key not in self.feat_cache:
            d = self.infeasible_pool[idx]
            A, b = d['A'], d['b'].copy()
            if comp:
                b = A.sum(axis=1).astype(b.dtype) - b
            self.feat_cache[key] = self.featurize(A, b, None, False)
        return self.feat_cache[key]

    def run_feas_finetune(self, num_epochs):
        # Head-only fine-tuning on fresh feasible + fresh certified infeasible;
        # the trunk and policy/value heads stay frozen so search is untouched
        if num_epochs <= 0:
            return
        print("\n=== Starting feasibility-head fine-tuning (head-only, fresh both classes) ===")
        self.model.train()
        head_opt = optim.Adam(self.model.feasibility_head.parameters(), lr=0.001)
        BATCH = min(64, self.args.batch_size)
        for epoch in range(1, num_epochs + 1):
            data = self.sample_epoch(self.args.samples_per_epoch, feas_ratio=0.5)
            correct = total = 0
            ep_loss = 0.0
            nb = 0
            for cs in range(0, len(data), BATCH):
                chunk = data[cs:cs + BATCH]
                batch_input = []
                for item in chunk:
                    env = FastBinaryEnv(item['A'], item['b']).propagate_constraints().apply_probing()
                    xv, xc, cx = env.get_tensor_state(self.device)
                    batch_input.append((xv, xc, item['xlp_v'], item['xlp_c'], cx, item['ev2c'], item['ec2v'], None, None))
                bxv, bxc, bxlpv, bxlpc, bcx, bev2c, bec2v, _, _, bidxv, bidxc = self.batch_converter(batch_input)
                b_pump = torch.stack([item['pump'] for item in chunk])
                targets = torch.tensor([float(item['gt_feasible']) for item in chunk], dtype=torch.float32, device=self.device)
                head_opt.zero_grad()
                _, _, _, feas_logits = self.model(bxv, bxc, bxlpv, bxlpc, bev2c, bec2v, bidxv, bidxc, pump_feats=b_pump, check_extra=bcx)
                bce = F.binary_cross_entropy_with_logits(feas_logits, targets, reduction='none')
                p_t = torch.exp(-bce)
                loss = ((1.0 - p_t) ** 2.0 * bce).mean()
                loss.backward()
                head_opt.step()
                ep_loss += loss.item()
                nb += 1
                with torch.no_grad():
                    preds = (torch.sigmoid(feas_logits) >= 0.5).float()
                    correct += (preds == targets).sum().item()
                    total += len(chunk)
            print(f"[Feas-FT] Ep {epoch} | Loss: {ep_loss / max(1, nb):.4f} | Acc: {100 * correct / max(1, total):.1f}%")
            if epoch % self.args.save_interval == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, tag=f"_FEASFT_A{100 * correct / max(1, total):.0f}")

    def sample_epoch(self, n_samples, feas_ratio=0.8):
        n_feas = int(n_samples * feas_ratio)
        data = [self.sample_fresh_feasible() for _ in range(n_feas)]
        if self.infeasible_pool:
            data += [self.sample_pool_infeasible() for _ in range(n_samples - n_feas)]
        random.shuffle(data)
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
        n_plots = 9
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
        plot_with_trendline(8, self.history.get('grad_norm', []), 'Grad Norm')
        plot_with_trendline(9, self.history.get('sel_entropy', []), 'Select Entropy')
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
        BATCH_SIZE = self.args.batch_size
        num_batches = max(1, self.args.samples_per_epoch // BATCH_SIZE)
        best_masked_acc = -1.0
        pretrain_patience_counter = 0
        for epoch in range(1, self.args.pretrain_epochs + 1):
            total_loss, correct_vars, total_masked_count = 0.0, 0, 0
            pbar = tqdm(range(num_batches), desc=f"[Pretrain] Ep {epoch}")
            start_mask_ratio = 0.2
            end_mask_ratio = 0.8
            progress = min(1.0, (epoch - 1) / self.args.pretrain_minimum_epoch)
            current_max_ratio = start_mask_ratio + (end_mask_ratio - start_mask_ratio) * progress
            for b_idx in pbar:
                batch_data = []
                list_pump = []
                fresh = [self.sample_fresh_feasible() for _ in range(BATCH_SIZE)]
                for item in fresh:
                    A, b_vec, x_gt = item['A'], item['b'], item['x_gt']
                    x_lp_var, x_lp_check = item['xlp_v'], item['xlp_c']
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
                    min_mask = max(1, int(0.10 * num_vars))
                    if mask.sum() < min_mask:
                        # Re-open variables WITHOUT re-propagating: propagation on
                        # the dense planted family would immediately re-solve them
                        assigned_indices = np.where(current_assignment != -1)[0]
                        num_to_unassign = min_mask - int(mask.sum())
                        if len(assigned_indices) > 0:
                            num_to_unassign = min(num_to_unassign, len(assigned_indices))
                            to_unassign = np.random.choice(assigned_indices, num_to_unassign, replace=False)
                            for idx in to_unassign:
                                current_assignment[idx] = -1
                            env = FastBinaryEnv(A, b_vec, current_assignment)
                            mask = (current_assignment == -1)
                    xv, xc, cx = env.get_tensor_state(self.device)
                    ev2c, ec2v = item['ev2c'], item['ec2v']
                    tgt = torch.tensor(x_gt, dtype=torch.long, device=self.device)
                    msk = torch.tensor(mask, dtype=torch.bool, device=self.device)
                    batch_data.append((xv, xc, x_lp_var, x_lp_check, cx, ev2c, ec2v, tgt, msk))
                    list_pump.append(item['pump'])
                if not batch_data:
                    continue
                bx_v, bx_c, b_xlpv, b_xlpc, b_cx, bev2c, bec2v, b_tgt, b_msk, b_idx_v, b_idx_c = self.batch_converter(batch_data)
                b_pump = torch.stack(list_pump)
                self.optimizer.zero_grad()
                _, assign_logits, _, feas_logits = self.model(bx_v, bx_c, b_xlpv, b_xlpc, bev2c, bec2v, b_idx_v, b_idx_c, pump_feats=b_pump, check_extra=b_cx)
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
        MCTS_INSTANCE_BATCH = 64
        NETWORK_BATCH_SIZE = self.args.batch_size
        prev_noise = getattr(self.model, 'noise_std', 0.05)
        if not is_il_phase:
            self.model.noise_std = 0.01
        gamma = 0.97
        for epoch in range(1, num_epochs + 1):
            self.state_visits = {}
            epoch_data = self.sample_epoch(self.args.samples_per_epoch)
            num_batches = int(np.ceil(len(epoch_data) / MCTS_INSTANCE_BATCH))
            epoch_loss = epoch_p_loss = epoch_v_loss = epoch_feas_loss = 0.0
            epoch_grad_norm = epoch_sel_ent = 0.0
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

                for item in current_batch_data:
                    A, b, x_gt, gt_feasible = item['A'], item['b'], item['x_gt'], item['gt_feasible']
                    x_lp_var, x_lp_check = item['xlp_v'], item['xlp_c']
                    edge_v2c, edge_c2v = item['ev2c'], item['ec2v']

                    # Clean root (no curriculum reveal) for the feasibility head, so
                    # the head cannot shortcut on "revealed variables imply feasible"
                    clean_env = FastBinaryEnv(A, b).propagate_constraints().apply_probing()

                    env = FastBinaryEnv(A, b)
                    # Reverse curriculum: reveal part of the planted solution early in
                    # training so self-play reaches solved terminals; anneal to zero
                    if x_gt is not None and not is_il_phase:
                        reveal_cap = max(0.0, 0.9 * (1.0 - (epoch - 1) / max(1.0, 0.7 * num_epochs)))
                        k_rev = int(len(x_gt) * self.rng.uniform(0.0, reveal_cap))
                        if k_rev > 0:
                            idxs = self.rng.choice(len(x_gt), k_rev, replace=False)
                            env.assignment[idxs] = x_gt[idxs]
                    current_env = env.propagate_constraints().apply_probing()

                    # Feasibility prediction at root (no_grad for accuracy tracking)
                    with torch.no_grad():
                        xv_root, xc_root, cx_root = clean_env.get_tensor_state(self.device)
                        _, _, _, feas_logit_root = self.model(
                            xv_root, xc_root, x_lp_var, x_lp_check, edge_v2c, edge_c2v,
                            pump_feats=item['pump'].unsqueeze(0), check_extra=cx_root
                        )
                        pred_feas = (torch.sigmoid(feas_logit_root).item() >= 0.5)
                    batch_feas_correct.append(pred_feas == gt_feasible)

                    # Store root state for feasibility focal-BCE gradient update
                    feas_roots.append({
                        'xv': xv_root, 'xc': xc_root, 'cx': cx_root,
                        'xlp_v': x_lp_var, 'xlp_c': x_lp_check,
                        'ev2c': edge_v2c, 'ec2v': edge_c2v,
                        'pump': item['pump'],
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

                    base_reward = current_env.get_reward()

                    for step_idx, (state, target_var, target_act, pi_var, has_gt) in enumerate(trajectory):
                        steps_to_end = len(trajectory) - step_idx - 1
                        if not is_il_phase and final_acc == 1.0:
                            state_key = tuple(state)
                            self.state_visits[state_key] = self.state_visits.get(state_key, 0) + 1
                            visit_count = self.state_visits[state_key]
                            intrinsic_reward = 0.2 * math.sqrt(1.0 / (visit_count + 1.0))
                        else:
                            intrinsic_reward = 0.0
                        final_value_target = np.clip(
                            base_reward * (gamma ** steps_to_end) + intrinsic_reward, 0.0, 1.0
                        )
                        temp_env = FastBinaryEnv(A, b, state)
                        xv, xc, cx = temp_env.get_tensor_state(self.device)
                        collected_trajectories.append({
                            'xv': xv, 'xc': xc, 'cx': cx,
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
                            batch_input.append((item['xv'], item['xc'], item['xlp_v'], item['xlp_c'], item['cx'], item['ev2c'], item['ec2v'], None, None))
                            list_act_target.append(item['target_act'])
                            list_val_target.append([item['final_value_target']])
                            list_target_node_mask.append(total_node_offset + item['target_var'])
                            total_node_offset += num_nodes
                        bx_v, bx_c, b_xlpv, b_xlpc, b_cx, bev2c, bec2v, _, _, b_idx_v, b_idx_c = self.batch_converter(batch_input)
                        b_act_target = torch.tensor(np.array(list_act_target), dtype=torch.float, device=self.device)
                        b_val_target = torch.tensor(np.array(list_val_target), dtype=torch.float, device=self.device).view(-1)
                        self.optimizer.zero_grad()
                        select_logits, assign_logits, v_pred, _ = self.model(bx_v, bx_c, b_xlpv, b_xlpc, bev2c, bec2v, b_idx_v, b_idx_c, check_extra=b_cx)
                        relevant_assign_logits = assign_logits[list_target_node_mask]
                        batch_size = len(chunk)
                        # Segment softmax over each instance's own variable set —
                        # batches may mix instance sizes freely
                        unassigned_mask = (bx_v == 0)
                        masked_logits_sel = select_logits.masked_fill(~unassigned_mask, -1e9)
                        sel_probs = segment_softmax(masked_logits_sel, b_idx_v)
                        sel_log_probs = torch.log(sel_probs.clamp_min(1e-12))
                        # AlphaZero-style policy target: cross-entropy against the full MCTS visit distribution
                        b_pi_flat = torch.tensor(np.concatenate([item['pi_var'] for item in chunk]), dtype=torch.float, device=self.device)
                        ce_per_node = -(b_pi_flat * sel_log_probs)
                        var_ce = torch.zeros(batch_size, device=self.device).index_add_(0, b_idx_v, ce_per_node)
                        act_log_probs = F.log_softmax(relevant_assign_logits, dim=1)
                        act_ce = -(b_act_target * act_log_probs).sum(dim=1)
                        # Search reads sigmoid(value_logit), so train in the same space
                        v_loss = F.mse_loss(torch.sigmoid(v_pred.view(-1)), b_val_target)
                        dist = Categorical(logits=relevant_assign_logits)
                        ent = dist.entropy().mean()
                        sel_ent_nodes = -(sel_probs * sel_log_probs)
                        sel_ent = torch.zeros(batch_size, device=self.device).index_add_(0, b_idx_v, sel_ent_nodes).mean()
                        p_loss = (var_ce + act_ce).mean()
                        loss = p_loss + (2.0 * v_loss) - (self.args.entropy_coef * ent)
                        if not torch.isnan(loss):
                            loss.backward()
                            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            epoch_loss += loss.item()
                            epoch_v_loss += v_loss.item()
                            epoch_p_loss += p_loss.item()
                            epoch_grad_norm += float(gn)
                            epoch_sel_ent += sel_ent.item()
                            valid_update_count += 1

                # ── Feasibility BCE gradient update ─────────────────────────────
                if feas_roots:
                    random.shuffle(feas_roots)
                    feas_update_count = 0
                    for chunk_start in range(0, len(feas_roots), NETWORK_BATCH_SIZE):
                        chunk_f = feas_roots[chunk_start:chunk_start + NETWORK_BATCH_SIZE]
                        batch_input_f = [
                            (item['xv'], item['xc'], item['xlp_v'], item['xlp_c'], item['cx'], item['ev2c'], item['ec2v'], None, None)
                            for item in chunk_f
                        ]
                        feas_targets = torch.tensor(
                            [item['gt_feasible'] for item in chunk_f], dtype=torch.float32, device=self.device
                        )
                        bxv_f, bxc_f, bxlpv_f, bxlpc_f, bcx_f, bev2c_f, bec2v_f, _, _, bidxv_f, bidxc_f = \
                            self.batch_converter(batch_input_f)
                        b_pump_f = torch.stack([item['pump'] for item in chunk_f])
                        self.optimizer.zero_grad()
                        _, _, _, feas_logits = self.model(
                            bxv_f, bxc_f, bxlpv_f, bxlpc_f, bev2c_f, bec2v_f, bidxv_f, bidxc_f,
                            pump_feats=b_pump_f, check_extra=bcx_f
                        )
                        # Focal BCE: concentrates gradient on the hard boundary cases
                        bce = F.binary_cross_entropy_with_logits(feas_logits, feas_targets, reduction='none')
                        p_t = torch.exp(-bce)
                        feas_loss = ((1.0 - p_t) ** 2.0 * bce).mean()
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
                self.history['grad_norm'].append(epoch_grad_norm / valid_update_count)
                self.history['sel_entropy'].append(epoch_sel_ent / valid_update_count)
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
                f"Solved: {epoch_solved} | "
                f"GradNorm: {epoch_grad_norm / max(1, valid_update_count):.3f} | "
                f"SelEnt: {epoch_sel_ent / max(1, valid_update_count):.3f}"
            )

            if not is_il_phase:
                self.scheduler.step()
            if epoch % self.args.save_interval == 0:
                tag = f"_{phase_prefix}_FA{epoch_feas_acc_pct:.0f}_RE{100*epoch_acc/max(1,feasible_batch_count):.0f}"
                self.save_checkpoint(epoch, tag=tag)
                self.save_history(epoch, prefix=phase_prefix)
                self.plot_metrics(epoch, prefix=phase_prefix)
        self.model.noise_std = prev_noise

    def expert_probe_scores(self, env):
        # Propagation-impact expert (strong-branching analogue for feasibility):
        # probe both values of each unassigned variable; contradictions rank
        # highest (fail-first), otherwise score by propagation-fixed count
        n = env.num_vars
        base_assigned = int((env.assignment != -1).sum())
        scores = np.zeros(n, dtype=np.float64)
        best_val = np.zeros(n, dtype=np.int64)
        unassigned = np.where(env.assignment == -1)[0]
        for v in unassigned:
            fixed = [0, 0]
            contra = [False, False]
            for val in (0, 1):
                e = FastBinaryEnv(env.A, env.b, env.assignment.copy())
                e.assignment[v] = val
                e.propagate_constraints()
                contra[val] = e.is_invalid
                fixed[val] = int((e.assignment != -1).sum()) - base_assigned
            if contra[0] and contra[1]:
                scores[v] = 2.0 * n
            elif contra[0] or contra[1]:
                scores[v] = n + fixed[0 if contra[1] else 1]
                best_val[v] = 0 if contra[1] else 1
            else:
                scores[v] = max(fixed)
                best_val[v] = int(np.argmax(fixed))
        return scores, best_val, unassigned

    def run_expert_imitation(self, num_epochs):
        # Imitation of the propagation-impact expert on fresh instances drawn
        # from a continuous size distribution — the size-transfer learning
        # signal that replaces self-play (learn2branch pattern)
        if num_epochs <= 0:
            return
        print("\n=== Starting expert imitation (propagation-impact oracle) ===")
        self.model.train()
        prev_noise = getattr(self.model, 'noise_std', 0.05)
        self.model.noise_std = 0.0
        gamma = 0.97
        NETWORK_BATCH = self.args.batch_size
        for epoch in range(1, num_epochs + 1):
            states = []
            solved_cnt = 0
            n_inst = max(1, self.args.samples_per_epoch // 4)
            pbar = tqdm(range(n_inst), desc=f"[Expert-IL] Ep {epoch}")
            for _ in pbar:
                item = self.sample_fresh_feasible()
                A, b, x_gt = item['A'], item['b'], item['x_gt']
                env = FastBinaryEnv(A, b).propagate_constraints().apply_probing()
                # Half the rollouts follow the planted solution (assignment targets
                # correct by construction, positive value states); half follow the
                # expert greedily (fail-path value signal)
                follow_solution = bool(self.rng.random() < 0.5)
                traj = []
                while not env.is_terminal():
                    scores, best_val, unassigned = self.expert_probe_scores(env)
                    if len(unassigned) == 0:
                        break
                    sc = scores[unassigned]
                    pi = sc - sc.min() + 1e-3
                    pi = pi / pi.sum()
                    pi_full = np.zeros(env.num_vars, dtype=np.float32)
                    pi_full[unassigned] = pi
                    tgt_var = int(unassigned[int(np.argmax(sc))])
                    tgt_act = int(x_gt[tgt_var]) if follow_solution else int(best_val[tgt_var])
                    xv, xc, cx = env.get_tensor_state(self.device)
                    traj.append({'xv': xv, 'xc': xc, 'cx': cx, 'pi_var': pi_full,
                        'target_var': tgt_var, 'target_act': tgt_act})
                    env.step_inplace(tgt_var, tgt_act)
                    env.propagate_constraints()
                base_reward = env.get_reward()
                solved_cnt += int(env.get_accuracy() == 1.0)
                for k, st in enumerate(traj):
                    steps_to_end = len(traj) - k - 1
                    st['val_target'] = float(np.clip(base_reward * (gamma ** steps_to_end), 0.0, 1.0))
                    st['xlp_v'] = item['xlp_v']
                    st['xlp_c'] = item['xlp_c']
                    st['ev2c'] = item['ev2c']
                    st['ec2v'] = item['ec2v']
                    states.append(st)
            random.shuffle(states)
            ep_loss = ep_pl = ep_vl = 0.0
            nb = 0
            for cs in range(0, len(states), NETWORK_BATCH):
                chunk = states[cs:cs + NETWORK_BATCH]
                batch_input = []
                list_target_node = []
                offset = 0
                for st in chunk:
                    batch_input.append((st['xv'], st['xc'], st['xlp_v'], st['xlp_c'], st['cx'], st['ev2c'], st['ec2v'], None, None))
                    list_target_node.append(offset + st['target_var'])
                    offset += st['xv'].size(0)
                bx_v, bx_c, b_xlpv, b_xlpc, b_cx, bev2c, bec2v, _, _, b_idx_v, b_idx_c = self.batch_converter(batch_input)
                b_act = torch.tensor([st['target_act'] for st in chunk], dtype=torch.long, device=self.device)
                b_val = torch.tensor([st['val_target'] for st in chunk], dtype=torch.float, device=self.device)
                self.optimizer.zero_grad()
                select_logits, assign_logits, v_pred, _ = self.model(bx_v, bx_c, b_xlpv, b_xlpc, bev2c, bec2v, b_idx_v, b_idx_c, check_extra=b_cx)
                unassigned_mask = (bx_v == 0)
                masked_sel = select_logits.masked_fill(~unassigned_mask, -1e9)
                sel_probs = segment_softmax(masked_sel, b_idx_v)
                sel_logp = torch.log(sel_probs.clamp_min(1e-12))
                b_pi = torch.tensor(np.concatenate([st['pi_var'] for st in chunk]), dtype=torch.float, device=self.device)
                var_ce = torch.zeros(len(chunk), device=self.device).index_add_(0, b_idx_v, -(b_pi * sel_logp))
                act_ce = F.cross_entropy(assign_logits[list_target_node], b_act)
                v_loss = F.mse_loss(torch.sigmoid(v_pred.view(-1)), b_val)
                loss = var_ce.mean() + act_ce + 2.0 * v_loss
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    ep_loss += loss.item()
                    ep_pl += var_ce.mean().item() + act_ce.item()
                    ep_vl += v_loss.item()
                    nb += 1
            print(f"[Expert-IL] Ep {epoch} | Loss: {ep_loss/max(1,nb):.4f} | PL: {ep_pl/max(1,nb):.4f} | VL: {ep_vl/max(1,nb):.4f} | ExpertSolved: {solved_cnt}/{n_inst} | States: {len(states)}")
            if epoch % self.args.save_interval == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, tag=f"_EXIL_S{100*solved_cnt/max(1,n_inst):.0f}")
        self.model.noise_std = prev_noise

    def run_dfs_finetune(self, num_epochs):
        # Role expansion of the GNN: REINFORCE on a Plackett-Luce ordering
        # distribution sampled from the selection head; episodes are budgeted
        # complete backtracking runs and the reward favors small proofs/solutions
        if num_epochs <= 0:
            return
        print("\n=== Starting DFS-ordering fine-tuning (Plackett-Luce REINFORCE) ===")
        self.model.train()
        prev_noise = getattr(self.model, 'noise_std', 0.05)
        self.model.noise_std = 0.0
        K = 12
        budget = self.args.dfs_budget_train
        baseline = 0.5
        for epoch in range(1, num_epochs + 1):
            data = self.sample_epoch(self.args.dfs_samples, feas_ratio=0.6)
            rewards = []
            losses = []
            self.optimizer.zero_grad()
            pbar = tqdm(data, desc=f"[DFS-FT] Ep {epoch}")
            for item in pbar:
                A, b = item['A'], item['b']
                env = FastBinaryEnv(A, b).propagate_constraints().apply_probing()
                if env.is_invalid or env.is_terminal():
                    continue
                xv, xc, cx = env.get_tensor_state(self.device)
                select_logits, assign_logits, _, _ = self.model(
                    xv, xc, item['xlp_v'], item['xlp_c'], item['ev2c'], item['ec2v'], check_extra=cx
                )
                unassigned = torch.tensor(env.assignment == -1, device=self.device)
                masked = select_logits.masked_fill(~unassigned, -1e9)
                gumbel = -torch.log(-torch.log(torch.rand_like(masked).clamp_min(1e-9)).clamp_min(1e-9))
                order = torch.argsort(-(masked + gumbel)).cpu().numpy()
                val_order = assign_logits.argmax(dim=1).cpu().numpy()
                _, status, nodes = policy_dfs(A, b, env.assignment, order, val_order, node_budget=budget)
                if status == 'budget':
                    r = 0.0
                else:
                    r = max(0.0, 1.0 - math.log(max(2, nodes)) / math.log(budget))
                rewards.append(r)
                # Plackett-Luce log-probability of the first K sampled positions
                terms = []
                rem = unassigned.clone()
                depth = 0
                for v in order:
                    if depth >= K:
                        break
                    if not rem[v]:
                        continue
                    idx = torch.nonzero(rem).squeeze(1)
                    terms.append(masked[v] - torch.logsumexp(masked[idx], dim=0))
                    rem = rem.clone()
                    rem[v] = False
                    depth += 1
                if terms:
                    logp = torch.stack(terms).sum()
                    losses.append(-(r - baseline) * logp)
            if losses:
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            mean_r = float(np.mean(rewards)) if rewards else 0.0
            baseline = 0.9 * baseline + 0.1 * mean_r
            print(f"[DFS-FT] Ep {epoch} | MeanReward: {mean_r:.3f} | Baseline: {baseline:.3f} | Episodes: {len(rewards)}")
            if epoch % self.args.save_interval == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, tag=f"_DFSFT_R{100*mean_r:.0f}")
        self.model.noise_std = prev_noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, nargs='+', default=['./train_instances_dnsms_10x25_10000'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dfs_epochs', type=int, default=0)
    parser.add_argument('--feas_epochs', type=int, default=0)
    parser.add_argument('--imitation_epochs', type=int, default=0)
    parser.add_argument('--size_range', type=int, nargs=2, default=None, help='Continuous training size range m_lo m_hi (n = 2.5m)')
    parser.add_argument('--dfs_samples', type=int, default=128)
    parser.add_argument('--dfs_budget_train', type=int, default=3000)
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
    if args.size_range:
        lo, hi = args.size_range
        trainer.sizes = [(mm, int(round(2.5 * mm))) for mm in range(lo, hi + 1)]
        print(f"  Continuous size distribution: m in [{lo},{hi}], n = 2.5m")
    if args.init_from:
        state = torch.load(args.init_from, map_location=trainer.device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        trainer.model.load_state_dict(state)
        print(f"Warm-started weights from {args.init_from}")
    trainer.pretrain()
    if args.il_epochs > 0:
        trainer.run_imitation_learning(args.il_epochs)
    if args.imitation_epochs > 0:
        trainer.run_expert_imitation(args.imitation_epochs)
    if args.rl_epochs > 0:
        trainer._run_mcts_phase(args.rl_epochs, is_il_phase=False)
    if args.dfs_epochs > 0:
        trainer.run_dfs_finetune(args.dfs_epochs)
    if args.feas_epochs > 0:
        trainer.run_feas_finetune(args.feas_epochs)

if __name__ == '__main__':
    main()
