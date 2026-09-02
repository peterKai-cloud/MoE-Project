import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.collections import PolyCollection 
import torch
import torch.nn.functional as F
import numpy as np
import traceback
import warnings

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, Isomap
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from scipy.spatial import ConvexHull
    from scipy.stats import pearsonr
    import scipy.ndimage  # 用于连通域分析和高斯抗锯齿平滑
except ImportError:
    print("⚠️ 警告: 缺少 sklearn 或 scipy 库，部分图表可能无法生成。请使用 pip install scikit-learn scipy 安装。")

try:
    matplotlib.use('TkAgg')
except:
    pass

# 配置论文所需字体，支持中文以及数学符号的显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False 
warnings.filterwarnings('ignore') 

DT = 0.1 
MAX_JERK = 5.0 

class VisualizationPlot:
    def __init__(self, dataset, model, stats):
        self.dataset = dataset
        self.model = model
        self.stats = stats
        self.device = next(model.parameters()).device
        
        self.mean_t = torch.tensor(stats['mean'], device=self.device, dtype=torch.float32)
        self.std_t = torch.tensor(stats['std'], device=self.device, dtype=torch.float32)
        
        self.total_samples = len(dataset)
        
        print(f">> 可视化工具启动: 共加载 {self.total_samples} 个测试片段。")
        
        self.expert_names = ['Mamba', 'Gipps', 'GM', 'FVD', 'Wiedemann', 'NETSIM', 'Mod_NETSIM', 'Fritzsche', 'Trans(Shared)', 'IDM(Shared)']
        self.private_expert_names = self.expert_names[:8]
        
        self.custom_colors = plt.cm.get_cmap('tab10').colors
        self.scene_colors = ['#5DADE2', '#F4D03F', '#E74C3C', '#8E44AD']
        self.scene_names = {
            0: "Conservative and evasive following",
            1: "Steady-state cruising balanced following",
            2: "Aggressive high-risk interactive following",
            3: "Stops and starts, jolts and congestion",
        }
        
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.canvas.manager.set_window_title('Car Following MoE Visualization (Paper Ready)')
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1.2, 1])
        
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])
        
        plt.subplots_adjust(bottom=0.1, right=0.82, hspace=0.35)
        
        self._interactive_loop()
        
    def _run_single_strategy(self, hist_t, gt_np, strategy):
        history_len = len(hist_t)
        traj_len = len(gt_np)
        
        curr_window_norm = hist_t.unsqueeze(0).clone()
        hist_phys = hist_t * self.std_t + self.mean_t
        
        last_frame_phys = hist_phys[-1]
        curr_spacing = float(last_frame_phys[0])
        curr_sv_spd = float(last_frame_phys[1])
        curr_rel_spd = float(last_frame_phys[2])
        
        if history_len >= 2:
            prev_spd = float(hist_phys[-2, 1])
            curr_acc = (curr_sv_spd - prev_spd) / DT
        else:
            curr_acc = 0.0
        
        gt_lv_spd_seq = gt_np[:, 3]
        
        pred_spacing_list = [curr_spacing]
        pred_speed_list = [curr_sv_spd]
        pred_acc_list = [curr_acc]
        
        expert_weights_list = []
        
        with torch.no_grad():
            for t in range(traj_len):
                if strategy == 'MoE':
                    pred_acc_norm, diagnostics = self.model(
                        curr_window_norm, return_diagnostics=True
                    )
                    weights = diagnostics['contribution_share']
                    raw_acc = pred_acc_norm.item()
                    expert_weights_list.append(weights.squeeze(0).detach().cpu().numpy())
                else:
                    x_phys_full = curr_window_norm * self.std_t + self.mean_t
                    last_frame = x_phys_full[:, -1, :]
                    gap = last_frame[:, 0]
                    v_ego = last_frame[:, 1]
                    v_lead = last_frame[:, 3]
                    
                    if strategy == 'Trans(Shared)':
                        raw_acc = self.model.shared_transformer(v_ego, v_lead, gap, x_phys_full).item()
                    elif strategy == 'IDM(Shared)':
                        raw_acc = self.model.shared_idm(v_ego, v_lead, gap, last_frame).item()
                    else:
                        e_idx = self.expert_names.index(strategy)
                        expert = self.model.private_experts[e_idx]
                        if strategy == 'Mamba':
                            raw_acc = expert(v_ego, v_lead, gap, x_phys_full).item()
                        else:
                            raw_acc = expert(v_ego, v_lead, gap, last_frame).item()
                
                target_acc = max(-4.0, min(3.0, raw_acc))
                max_delta = MAX_JERK * DT
                delta_acc = max(-max_delta, min(max_delta, target_acc - curr_acc))
                final_acc = curr_acc + delta_acc
                curr_acc = final_acc
                
                next_sv_spd = max(0.001, curr_sv_spd + final_acc * DT)
                next_lv_spd = gt_lv_spd_seq[t]
                next_rel_spd = next_lv_spd - next_sv_spd
                next_spacing = curr_spacing + DT * (curr_rel_spd + next_rel_spd) / 2.0
                
                pred_spacing_list.append(next_spacing)
                pred_speed_list.append(next_sv_spd)
                pred_acc_list.append(final_acc)
                
                next_frame_phys = torch.tensor(
                    [next_spacing, next_sv_spd, next_rel_spd, next_lv_spd], 
                    device=self.device, dtype=torch.float32 
                )
                next_frame_norm = (next_frame_phys - self.mean_t) / self.std_t
                curr_window_norm = torch.cat(
                    [curr_window_norm[:, 1:, :], next_frame_norm.view(1, 1, 4)], dim=1
                )
                curr_spacing = next_spacing
                curr_sv_spd = next_sv_spd
                curr_rel_spd = next_rel_spd
                
        return {
            'pred_spacing': np.array(pred_spacing_list),
            'pred_speed': np.array(pred_speed_list),
            'pred_acc': np.array(pred_acc_list),
            'expert_weights': np.array(expert_weights_list) if strategy == 'MoE' else None
        }

    def plot_sample(self, idx):
        if idx < 0 or idx >= self.total_samples: return
        
        if hasattr(self.dataset, 'samples'):
            s = self.dataset.samples[idx]
            hist_np = s['history']
            gt_np = s['future']
            hist_t = torch.tensor((hist_np - self.stats['mean']) / self.stats['std'], dtype=torch.float32).to(self.device)
        else:
            try:
                hist_t, gt_t, _ = self.dataset[idx]
                hist_t = hist_t.to(self.device)
                gt_np = gt_t.numpy()
                if gt_np.ndim == 3: gt_np = gt_np[0]
            except Exception as e:
                return

        all_strategies = ['MoE'] + self.expert_names
        results = {}
        for strategy in all_strategies:
            results[strategy] = self._run_single_strategy(hist_t, gt_np, strategy)
            
        gt_spacing = np.concatenate(([results['MoE']['pred_spacing'][0]], gt_np[:, 0]))
        gt_speed = np.concatenate(([results['MoE']['pred_speed'][0]], gt_np[:, 1]))
        
        time_axis = np.arange(len(gt_spacing)) * DT
        valid_len = len(time_axis)
        
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()
        
        colors = plt.cm.get_cmap('tab10').colors
        
        self.ax1.plot(time_axis, gt_spacing[:valid_len], color='black', linestyle='-', linewidth=4, label='Ground Truth', zorder=5)
        for i, exp in enumerate(self.expert_names):
            self.ax1.plot(time_axis, results[exp]['pred_spacing'], color=colors[i%len(colors)], linestyle='--', linewidth=1.5, label=exp, alpha=0.85, zorder=6)
        self.ax1.plot(time_axis, results['MoE']['pred_spacing'], color='#e74c3c', linestyle='-', linewidth=3.5, label='MoE (Proposed)', zorder=10)
        self.ax1.set_ylabel('Spacing (m)', fontsize=13)
        self.ax1.set_title(f'Multi-Expert Closed-Loop Rollout Comparison (Sample ID: {idx})', fontsize=15, fontweight='bold')
        self.ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
        self.ax1.grid(True, linestyle=':', alpha=0.7)
        
        self.ax2.plot(time_axis, gt_speed[:valid_len], color='black', linestyle='-', linewidth=4, label='Ground Truth Speed', zorder=5)
        for i, exp in enumerate(self.expert_names):
            self.ax2.plot(time_axis, results[exp]['pred_speed'], color=colors[i%len(colors)], linestyle='--', linewidth=1.5, label=exp, alpha=0.85, zorder=6)
        self.ax2.plot(time_axis, results['MoE']['pred_speed'], color='#e74c3c', linestyle='-', linewidth=3.5, label='MoE (Proposed)', zorder=10)
        self.ax2.set_ylabel('Speed (m/s)', fontsize=13)
        self.ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
        self.ax2.grid(True, linestyle=':', alpha=0.7)
        
        moe_weights = results['MoE']['expert_weights'] 
        if moe_weights is not None and len(moe_weights) > 0:
            moe_weights_T = moe_weights.T 
            for e_idx, name in enumerate(self.expert_names):
                self.ax3.plot(time_axis[1:], moe_weights_T[e_idx], label=name, color=colors[e_idx%len(colors)], linewidth=2, alpha=0.9)
            self.ax3.set_ylabel('MoE Weights', fontsize=13)
            self.ax3.set_xlabel('Time (s)', fontsize=13)
            self.ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, ncol=2)
            self.ax3.grid(True, linestyle=':', alpha=0.7)

        self.fig.canvas.draw()
        
        save_filename = f"trajectory_comparison_sample_{idx}.png"
        self.fig.savefig(save_filename, bbox_inches='tight', dpi=300)
        print(f"✅ 成功渲染车辆 {idx} 的多专家闭环轨迹对比图！已自动保存至: {save_filename}")

    def _collect_rollout_frames(self):
        print(f"\n⏳ 正在启动全量测试集闭环推演模拟器，自动收集动态数据...")
        indices = np.arange(self.total_samples)
        all_v_ego, all_v_lead, all_gap = [], [], []
        all_scene_ids, all_dominant_experts, all_phys_feats = [], [], []
        
        batch_size = 256
        frames_collected = 0
        
        with torch.no_grad():
            for i in range(0, self.total_samples, batch_size):
                chunk_indices = indices[i:i+batch_size]
                hist_list, future_list = [], []
                for idx in chunk_indices:
                    if hasattr(self.dataset, 'samples'):
                        hist_np = self.dataset.samples[idx]['history']
                        fut_np = self.dataset.samples[idx]['future']
                        hist_t = torch.tensor((hist_np - self.stats['mean']) / self.stats['std'], dtype=torch.float32)
                        fut_t = torch.tensor(fut_np, dtype=torch.float32)
                    else:
                        hist_t, fut_t, _ = self.dataset[idx]
                        if fut_t.ndim == 3: fut_t = fut_t[0]
                    hist_list.append(hist_t)
                    future_list.append(fut_t)
                    
                max_len = max(len(f) for f in future_list)
                N = len(chunk_indices)
                
                hist_batch = torch.stack(hist_list).to(self.device)
                gt_batch = torch.zeros(N, max_len, 4, device=self.device)
                mask = torch.zeros(N, max_len, device=self.device)
                
                for j, fut in enumerate(future_list):
                    length = len(fut)
                    gt_batch[j, :length, :] = fut.to(self.device)
                    mask[j, :length] = 1.0
                    if length < max_len: gt_batch[j, length:, 3] = fut[-1, 3] 
                        
                gt_lv_spd_seq = gt_batch[:, :, 3]
                curr_window_norm = hist_batch.clone()
                last_frame_phys = hist_batch[:, -1, :] * self.std_t + self.mean_t
                curr_spacing, curr_sv_spd, curr_rel_spd = last_frame_phys[:, 0], last_frame_phys[:, 1], last_frame_phys[:, 2]
                
                prev_spd = hist_batch[:, -2, 1] * self.std_t[1] + self.mean_t[1]
                curr_acc = (curr_sv_spd - prev_spd) / DT
                
                for t in range(max_len):
                    active_mask = mask[:, t]
                    if active_mask.sum() == 0: break
                    active_indices = active_mask.nonzero(as_tuple=True)[0]
                    
                    curr_phys = curr_window_norm * self.std_t + self.mean_t
                    gap = curr_phys[:, -1, 0]
                    v_ego = curr_phys[:, -1, 1]
                    avg_speed = torch.mean(curr_phys[:, :, 1], dim=1)
                    acc_seq = (curr_phys[:, 1:, 1] - curr_phys[:, :-1, 1]) / DT
                    avg_acc = torch.mean(acc_seq, dim=1) if acc_seq.shape[1] > 0 else torch.zeros_like(avg_speed)
                    std_speed = torch.sqrt(torch.clamp(torch.var(curr_phys[:, :, 1], dim=1, unbiased=False) + 1e-5, min=1e-5))
                    std_acc = torch.sqrt(torch.clamp(torch.var(acc_seq, dim=1, unbiased=False) + 1e-5, min=1e-5)) if acc_seq.shape[1] > 1 else torch.zeros_like(avg_acc)
                    
                    if curr_phys.shape[1] > 2:
                        jerk_seq = (acc_seq[:, 1:] - acc_seq[:, :-1]) / DT
                        avg_jerk = torch.mean(torch.abs(jerk_seq), dim=1)
                        std_jerk = torch.sqrt(torch.clamp(torch.var(jerk_seq, dim=1, unbiased=False) + 1e-5, min=1e-5))
                    else:
                        avg_jerk = std_jerk = torch.zeros_like(avg_acc)
                        
                    v_ego_safe = torch.clamp(v_ego, min=0.1)
                    gap_safe = torch.clamp(gap, min=0.1)
                    gap_seq_safe = torch.clamp(curr_phys[:, :, 0], min=0.1)
                    
                    time_headway = torch.clamp(gap_safe / v_ego_safe, max=10.0)
                    
                    inv_ttc = F.relu(-curr_phys[:, -1, 2]) / gap_safe
                    inv_ttc_seq = F.relu(-curr_phys[:, :, 2]) / gap_seq_safe
                    delta_inv_ttc = inv_ttc_seq[:, -1] - inv_ttc_seq[:, 0]
                    
                    phys_feats = torch.stack([avg_speed / 30.0, std_speed / 5.0, avg_acc / 3.0, inv_ttc, avg_jerk / 5.0, time_headway / 2.0, delta_inv_ttc / 0.5, std_jerk / 5.0, std_acc / 3.0], dim=1)
                    phys_feats = torch.nan_to_num(phys_feats, nan=0.0, posinf=10.0, neginf=-10.0)
                    phys_feats = torch.clamp(phys_feats, min=-15.0, max=15.0)
                    
                    scene_probs = self.model.scene_perception(phys_feats)
                    scene_ids = torch.argmax(scene_probs, dim=1)
                    
                    pred_acc_norm, diagnostics = self.model(
                        curr_window_norm, return_diagnostics=True
                    )
                    raw_acc = pred_acc_norm.squeeze(-1) if pred_acc_norm.ndim > 1 else pred_acc_norm
                    private_weights = diagnostics['private_router_weights']
                    dom_expert = torch.argmax(private_weights, dim=1)
                    
                    all_gap.extend(curr_spacing[active_indices].cpu().numpy())
                    all_v_ego.extend(curr_sv_spd[active_indices].cpu().numpy())
                    all_v_lead.extend(gt_lv_spd_seq[active_indices, t].cpu().numpy())
                    all_dominant_experts.extend(dom_expert[active_indices].cpu().numpy())
                    all_scene_ids.extend(scene_ids[active_indices].cpu().numpy())
                    all_phys_feats.extend(phys_feats[active_indices].cpu().numpy())
                    
                    frames_collected += len(active_indices)
                    
                    target_acc = torch.clamp(raw_acc, -4.0, 3.0)
                    max_delta = MAX_JERK * DT
                    delta = torch.clamp(target_acc - curr_acc, -max_delta, max_delta)
                    final_acc = curr_acc + delta
                    curr_acc = final_acc
                    
                    next_sv_spd = torch.clamp(curr_sv_spd + final_acc * DT, min=0.001)
                    next_lv_spd = gt_lv_spd_seq[:, t]
                    next_rel_spd = next_lv_spd - next_sv_spd
                    next_spacing = curr_spacing + DT * (curr_rel_spd + next_rel_spd) / 2.0
                    
                    next_frame_phys = torch.stack([next_spacing, next_sv_spd, next_rel_spd, next_lv_spd], dim=1)
                    next_frame_norm = (next_frame_phys - self.mean_t) / self.std_t
                    next_frame_norm = torch.clamp(next_frame_norm, min=-25.0, max=25.0)
                    
                    curr_window_norm = torch.cat([curr_window_norm[:, 1:, :], next_frame_norm.unsqueeze(1)], dim=1)
                    curr_spacing, curr_sv_spd, curr_rel_spd = next_spacing, next_sv_spd, next_rel_spd
                    
                print(f"     => 已处理 {(i+len(chunk_indices))} / {self.total_samples} 辆车 | 累计捕获 {frames_collected} 帧...")

        return {
            'v_ego': np.array(all_v_ego), 'v_lead': np.array(all_v_lead), 'gap': np.array(all_gap),
            'scene_ids': np.array(all_scene_ids), 'dominant_experts': np.array(all_dominant_experts), 'phys_feats': np.array(all_phys_feats)
        }

    def _filter_valid_data(self, data):
        valid_mask = (data['gap'] <= 30.0)
        filtered_data = {
            'v_ego': data['v_ego'][valid_mask],
            'v_lead': data['v_lead'][valid_mask],
            'gap': data['gap'][valid_mask],
            'scene_ids': data['scene_ids'][valid_mask],
            'dominant_experts': data['dominant_experts'][valid_mask],
            'phys_feats': data['phys_feats'][valid_mask]
        }
        print(f"   >> 数据清洗: 已剔除 Spacing > 30m 的游离帧，剩余 {len(filtered_data['gap'])} 核心有效帧。")
        return filtered_data

    def _get_sync_sampled_data(self, data, max_pts=2500):
        """统一采样函数：确保多个可视化模块使用完全一致的随机样本子集"""
        idx_list = []
        dominant_experts = data['dominant_experts']
        for e_idx in range(8):
            mask = (dominant_experts == e_idx)
            count = np.sum(mask)
            indices = np.where(mask)[0]
            if count > max_pts:
                # 只抽一次，将锁定的子集送给后续模块
                sampled = np.random.choice(indices, max_pts, replace=False)
                idx_list.extend(sampled)
            else:
                idx_list.extend(indices)
        
        idx_array = np.array(idx_list)
        return {
            'v_ego': data['v_ego'][idx_array],
            'v_lead': data['v_lead'][idx_array],
            'gap': data['gap'][idx_array],
            'scene_ids': data['scene_ids'][idx_array],
            'dominant_experts': data['dominant_experts'][idx_array],
            'phys_feats': data['phys_feats'][idx_array]
        }

    def _draw_confidence_ellipse(self, x, y, ax, color, n_std=2.5):
        if len(x) < 5: return
        cov = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * n_std * np.sqrt(vals)
        ell_fill = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=theta, color=color, alpha=0.1, zorder=0)
        ell_edge = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=theta, color=color, fill=False, linewidth=1.5, zorder=1)
        ax.add_patch(ell_fill)
        ax.add_patch(ell_edge)
        
    def _fast_kde_2d_gpu(self, train_pts, eval_pts, bw=0.3, chunk_size=10000):
        device = self.device
        train_t = torch.tensor(train_pts, dtype=torch.float32, device=device)
        eval_t = torch.tensor(eval_pts, dtype=torch.float32, device=device)
        
        d, N = train_t.shape
        fact = 1.0 / (N - 1)
        train_c = train_t - torch.mean(train_t, dim=1, keepdim=True)
        cov = fact * train_c.matmul(train_c.T)
        cov += torch.eye(d, device=device) * 1e-5
        
        inv_cov = torch.inverse(cov) / (bw ** 2)
        det_cov = torch.det(cov) * (bw ** 4)
        norm_const = 1.0 / (2.0 * np.pi * torch.sqrt(det_cov))
        
        densities = []
        for i in range(0, eval_t.shape[1], chunk_size):
            chunk = eval_t[:, i:i+chunk_size] 
            diff = chunk.unsqueeze(2) - train_t.unsqueeze(1)
            dist2 = torch.einsum('cmn, cd, dmn -> mn', diff, inv_cov, diff)
            density = norm_const * torch.exp(-0.5 * dist2).mean(dim=1)
            densities.append(density)
            
        return torch.cat(densities).cpu().numpy()

    def plot_scene_advanced_analysis(self):
        raw_data = self._collect_rollout_frames()
        data = self._filter_valid_data(raw_data)
        phys_feats_np = data['phys_feats']
        scene_ids = data['scene_ids']
        
        feature_names = ['Avg Speed', 'Speed Std', 'Avg Acc', 'Inv TTC', 'Avg Jerk', 'Time Headway', 'Delta Inv TTC', 'Jerk Std', 'Acc Std']
        
        unique_scenes, counts = np.unique(scene_ids, return_counts=True)
        
        # 🌟 核心修改 1：注入具有明确学术物理语义的场景标签
        scene_names_map = {
            s_id: self.scene_names.get(int(s_id), f'Scene {int(s_id) + 1}')
            for s_id in unique_scenes
        }

        print("\n📊 真实场景分布统计 (全量核心预测帧):")
        for s, c in zip(unique_scenes, counts):
            print(f"   >> {scene_names_map[s]} (底层模型原始ID:{s}): {c} 帧")

        target_total = 10000
        n_classes = len(unique_scenes)
        target_per_class = target_total // n_classes

        idx_sub = []
        for s_id in unique_scenes:
            s_indices = np.where(scene_ids == s_id)[0]
            if len(s_indices) > target_per_class:
                sampled = np.random.choice(s_indices, target_per_class, replace=False)
            else:
                sampled = s_indices
            idx_sub.extend(sampled.tolist())

        current_len = len(idx_sub)
        if current_len < target_total and current_len < len(scene_ids):
            remaining_indices = list(set(range(len(scene_ids))) - set(idx_sub))
            needed = min(target_total - current_len, len(remaining_indices))
            if needed > 0:
                idx_sub.extend(np.random.choice(remaining_indices, needed, replace=False).tolist())

        idx_sub = np.array(idx_sub)
        feats_sub = phys_feats_np[idx_sub]
        scenes_sub = scene_ids[idx_sub]

        print(f"⏳ 已使用【分层均衡抽样】抓取 {len(idx_sub)} 个代表点，开始计算 t-SNE (非线性聚类岛屿)...")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='pca', random_state=42, learning_rate='auto')
        feats_tsne = tsne.fit_transform(feats_sub)

        print("⏳ 正在提取 t-SNE 空间下的特征向量映射梯度 (t-SNE Feature Biplot)...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feats_std = scaler.fit_transform(feats_sub)
        
        loadings = np.zeros((len(feature_names), 2))
        for i in range(len(feature_names)):
            rx, _ = pearsonr(feats_std[:, i], feats_tsne[:, 0])
            ry, _ = pearsonr(feats_std[:, i], feats_tsne[:, 1])
            loadings[i, 0] = rx
            loadings[i, 1] = ry

        print("⏳ 正在计算 Rank-Based 特征相对优势度 (雷达图排序得分)...")
        raw_profiles = []
        for s_id in unique_scenes:
            mask = (scene_ids == s_id)
            raw_profiles.append(np.mean(phys_feats_np[mask], axis=0))
        raw_profiles = np.array(raw_profiles)
        
        mapped_profiles = np.zeros_like(raw_profiles)
        for j in range(len(feature_names)):
            col = raw_profiles[:, j]
            ranks = col.argsort().argsort() 
            if n_classes > 1:
                mapped_values = 0.25 + (0.75 / (n_classes - 1)) * ranks
            else:
                mapped_values = np.array([1.0])
            mapped_profiles[:, j] = mapped_values
            
        radar_profiles = {s_id: mapped_profiles[i] for i, s_id in enumerate(unique_scenes)}

        fig_tsne, ax_tsne = plt.subplots(figsize=(9, 7))
        fig_tsne.canvas.manager.set_window_title('t-SNE Clusters')
        
        fig_biplot, ax_biplot = plt.subplots(figsize=(9, 7))
        fig_biplot.canvas.manager.set_window_title('t-SNE Biplot Analysis')
        
        fig_radar = plt.figure(figsize=(9, 7))
        fig_radar.canvas.manager.set_window_title('Radar Chart Analysis')
        ax_radar = fig_radar.add_subplot(111, projection='polar')

        legend_elements = []
        
        for i, s_id in enumerate(unique_scenes):
            mask = (scenes_sub == s_id)
            if np.sum(mask) < 2: continue 
                
            color = self.scene_colors[i % len(self.scene_colors)]
            scene_label = scene_names_map[s_id]
            legend_elements.append(mpatches.Patch(color=color, label=scene_label))
            
            pts_t = feats_tsne[mask]
            
            ax_tsne.scatter(pts_t[:, 0], pts_t[:, 1], c=[color], alpha=0.7, s=8, edgecolors='none')
            ax_biplot.scatter(pts_t[:, 0], pts_t[:, 1], c=[color], alpha=0.6, s=8, edgecolors='none')

        ax_tsne.set_title('t-SNE Manifold of Car-Following Scenes', fontsize=16, fontweight='bold', pad=15)
        ax_tsne.set_xlabel('t-SNE Component 1', fontsize=14)
        ax_tsne.set_ylabel('t-SNE Component 2', fontsize=14)
        ax_tsne.grid(True, linestyle=':', alpha=0.5)
        # 🌟 核心修改：将 t-SNE 聚类图的图例移至图片正下方，排列成 2 列增加紧凑性
        ax_tsne.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=11, ncol=2)
        
        fig_tsne.tight_layout()
        fig_tsne.savefig("scene_adv_1_tsne.png", bbox_inches='tight', dpi=300)

        # 🌟 核心修改 2：更新轴向的多维复合物理空间名称定义
        ax_biplot.set_title('t-SNE Feature Projection Biplot', fontsize=16, fontweight='bold', pad=15)
        ax_biplot.set_xlabel('Interaction risk and response dimension', fontsize=14)
        ax_biplot.set_ylabel('Kinematic state and velocity domain', fontsize=14)
        ax_biplot.grid(True, linestyle=':', alpha=0.5)
        # 🌟 核心修改：将 Biplot 图的图例移至图片正下方，避免遮挡数据点并保持水平对称，两列布局
        ax_biplot.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=11, ncol=2)
        
        max_coord = np.max(np.abs(feats_tsne))
        max_load = np.max(np.abs(loadings))
        scale = (max_coord / max_load) * 0.85 
        
        target_features_to_draw = ['Avg Speed', 'Avg Acc', 'Time Headway', 'Inv TTC']
        
        for i, feature in enumerate(feature_names):
            if feature in target_features_to_draw:
                x_vec = loadings[i, 0] * scale
                y_vec = loadings[i, 1] * scale
                
                head_w = max_coord * 0.035
                ax_biplot.arrow(0, 0, x_vec, y_vec, color='#C0392B', alpha=0.85, width=head_w*0.1, head_width=head_w, zorder=5)
                ax_biplot.text(x_vec * 1.1, y_vec * 1.1, feature, color='black', fontsize=11, fontweight='bold',
                               ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
                        
        fig_biplot.tight_layout()
        fig_biplot.savefig("scene_adv_2_tsne_biplot.png", bbox_inches='tight', dpi=300)

        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        angles += angles[:1] 
        
        ax_radar.set_title('Feature Profile Radar Chart (Rank-Based Dominance)', fontsize=16, fontweight='bold', pad=25)
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(feature_names, fontsize=11)
        
        ticks = [0.25, 0.50, 0.75, 1.00] if n_classes == 4 else np.linspace(0.25, 1.0, n_classes)
        ax_radar.set_yticks(ticks)
        ax_radar.set_yticklabels([f'{t:.2f}' for t in ticks], color="grey", size=9)
        ax_radar.set_ylim(0, 1.05)
        
        for i, s_id in enumerate(unique_scenes):
            if s_id not in radar_profiles: continue
            color = self.scene_colors[i % len(self.scene_colors)]
            vals = radar_profiles[s_id].tolist()
            vals += vals[:1] 
            ax_radar.plot(angles, vals, color=color, linewidth=2.5, linestyle='solid', label=scene_names_map[s_id])
            ax_radar.fill(angles, vals, color=color, alpha=0.15)
            
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=12)

        fig_radar.tight_layout()
        fig_radar.savefig("scene_adv_3_radar.png", bbox_inches='tight', dpi=300)

        print("\n✅ 成功渲染完美分离的场景聚类分析！已分别保存至:")
        print("   - scene_adv_1_tsne.png")
        print("   - scene_adv_2_tsne_biplot.png")
        print("   - scene_adv_3_radar.png")
        
        fig_tsne.show()
        fig_biplot.show()
        fig_radar.show()

    def plot_scene_3d(self):
        raw_data = self._collect_rollout_frames()
        data = self._filter_valid_data(raw_data)
        phys_feats_np = data['phys_feats']
        scene_ids = data['scene_ids']
        total_valid_frames = len(scene_ids)
        
        from sklearn.decomposition import PCA
        from scipy.spatial import ConvexHull
        pca = PCA(n_components=3)
        feats_3d = pca.fit_transform(phys_feats_np)
        explained_var = pca.explained_variance_ratio_ * 100
        
        fig3d = plt.figure(figsize=(12, 9))
        ax3d = fig3d.add_subplot(111, projection='3d')
        
        unique_scenes = np.unique(scene_ids)
        scene_names_map = {
            s_id: self.scene_names.get(int(s_id), f'Scene {int(s_id) + 1}')
            for s_id in unique_scenes
        }
        colors = plt.cm.get_cmap('tab10').colors
        
        for i, s_id in enumerate(unique_scenes):
            mask = (scene_ids == s_id)
            count = np.sum(mask)
            color = colors[i % len(colors)]
            if count == 0: continue
                
            pts = feats_3d[mask]
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                         c=[color], label=scene_names_map[s_id],
                         s=3, alpha=0.3, edgecolors='none')
                         
            if count >= 4:
                try:
                    hull_pts = pts if count <= 15000 else pts[np.random.choice(count, 15000, replace=False)]
                    hull_pts_jitter = hull_pts + np.random.normal(0, 1e-4, hull_pts.shape)
                    hull = ConvexHull(hull_pts_jitter)
                    
                    face_c = mcolors.to_rgba(color, alpha=0.1)
                    edge_c = mcolors.to_rgba(color, alpha=0.6)
                    ax3d.plot_trisurf(hull_pts[:, 0], hull_pts[:, 1], hull_pts[:, 2], triangles=hull.simplices,
                                      facecolors=face_c, edgecolors=edge_c, linewidth=0.3, shade=False)
                except Exception:
                    pass
                         
        ax3d.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1f}%)', fontsize=12, labelpad=8)
        ax3d.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1f}%)', fontsize=12, labelpad=8)
        ax3d.set_zlabel(f'Principal Component 3 ({explained_var[2]:.1f}%)', fontsize=12, labelpad=8)
        ax3d.set_title(f'9D Feature Space Clustering & Boundaries (Predicting {total_valid_frames} Frames)', fontsize=16, fontweight='bold', pad=20)
        
        ax3d.view_init(elev=30, azim=-135)
        ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
        ax3d.grid(True, linestyle=':', alpha=0.5)
        ax3d.legend(title='Identified Scene Regions', loc='upper right', bbox_to_anchor=(1.15, 0.9), fontsize=12)
        plt.tight_layout()
        fig3d.savefig("scene_perception_9D_boundary_cluster.png", bbox_inches='tight', dpi=300)
        fig3d.show()

    def plot_expert_domain_3d(self):
        raw_data = self._collect_rollout_frames()
        data = self._filter_valid_data(raw_data)
        x_vals, y_vals, z_vals = data['v_ego'], data['v_lead'], data['gap']
        dominant_experts = data['dominant_experts']
        total_valid_frames = len(dominant_experts)

        print("\n" + "★"*95)
        print("🏆 各大私有专家【擅长主导状态范围】定量分析报告 (基于 Spacing <= 30m)")
        print("-" * 95)
        print(f" {'专家名称 (Expert)':<20} | {'主导状态范围 (Dominant State Range)':<55} | {'管辖帧数'}")
        print("-" * 95)
        
        for e_idx in range(8):
            expert_name = self.private_expert_names[e_idx]
            mask = (dominant_experts == e_idx)
            count = np.sum(mask)
            
            if count > 10: 
                v_ego_pts, v_lead_pts, gap_pts = x_vals[mask], y_vals[mask], z_vals[mask]
                v_ego_min, v_ego_max = np.percentile(v_ego_pts, [2, 98])
                v_lead_min, v_lead_max = np.percentile(v_lead_pts, [2, 98])
                gap_min, gap_max = np.percentile(gap_pts, [2, 98])
                
                range_str = (f"V_lead: {int(v_lead_min):>2}~{int(v_lead_max):<2} m/s, "
                             f"Spacing: {int(gap_min):>2}~{int(gap_max):<3} m, "
                             f"v_ego: {int(v_ego_min):>2}~{int(v_ego_max):<2} m/s")
                print(f" {expert_name:<20} | {range_str:<55} | n={count}")
            else:
                print(f" {expert_name:<20} | {'暂无足够主导帧，未形成优势区间':<55} | n={count}")
                
        print("★"*95 + "\n")

        fig3d = plt.figure(figsize=(12, 9))
        ax3d = fig3d.add_subplot(111, projection='3d')

        from scipy.spatial import ConvexHull
        for e_idx in range(8):
            expert_name = self.private_expert_names[e_idx]
            mask = (dominant_experts == e_idx)
            count = np.sum(mask)
            color = self.custom_colors[e_idx % len(self.custom_colors)]
            if count == 0: continue
                
            pts_x, pts_y, pts_z = x_vals[mask], y_vals[mask], z_vals[mask]
            pts = np.column_stack((pts_x, pts_y, pts_z))
            
            ax3d.scatter(pts_x, pts_y, pts_z, c=[color], label=f'{expert_name}', s=3, alpha=0.3, edgecolors='none')
                         
            if count >= 4:
                try:
                    hull_pts = pts if count <= 15000 else pts[np.random.choice(count, 15000, replace=False)]
                    hull_pts_jitter = hull_pts + np.random.normal(0, 1e-4, hull_pts.shape)
                    hull = ConvexHull(hull_pts_jitter)
                    
                    face_c = mcolors.to_rgba(color, alpha=0.15)
                    edge_c = mcolors.to_rgba(color, alpha=0.7)
                    ax3d.plot_trisurf(hull_pts[:, 0], hull_pts[:, 1], hull_pts[:, 2], triangles=hull.simplices,
                                      facecolors=face_c, edgecolors=edge_c, linewidth=0.3, shade=False)
                except Exception:
                    pass

        ax3d.set_xlabel('Ego Speed ($v_{ego}$, m/s)', fontsize=12, labelpad=8)
        ax3d.set_ylabel('Lead Speed ($v_{lead}$, m/s)', fontsize=12, labelpad=8)
        ax3d.set_zlabel('Spacing ($gap$, m)', fontsize=12, labelpad=8)
        ax3d.set_title(f'Spatial Domains of Private Experts (Based on {total_valid_frames} Predicted Frames)', fontsize=16, fontweight='bold', pad=20)
        
        ax3d.view_init(elev=30, azim=-135) 
        ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
        ax3d.grid(True, linestyle=':', alpha=0.5)
        ax3d.legend(title='Dominant Private Expert', loc='upper right', bbox_to_anchor=(1.15, 0.9), fontsize=12)
        plt.tight_layout()
        fig3d.savefig("expert_domain_3D_boundary.png", bbox_inches='tight', dpi=300)
        fig3d.show()

    def plot_expert_kde_3d(self, provided_data=None):
        if provided_data is None:
            raw_data = self._collect_rollout_frames()
            data = self._filter_valid_data(raw_data) 
        else:
            data = provided_data
            
        x_vals, y_vals, z_vals = data['v_ego'], data['v_lead'], data['gap']
        
        rel_vals = y_vals - x_vals
        dominant_experts = data['dominant_experts']

        print("\n⏳ 正在为每个专家利用 GPU 飞速生成 3D 相空间 KDE 曲面图...")
        
        X_rel, Y_gap = np.mgrid[-8:8:50j, 0:35:50j]
        pos_rel_gap = np.vstack([X_rel.ravel(), Y_gap.ravel()])

        spacing_factor = 6.0 
        waterfall_data = {} 
        expert_surfaces = {}
        
        global_max_z_rel = 0
        global_max_z_gap = 0

        for e_idx in range(8):
            expert_name = self.private_expert_names[e_idx]
            mask = (dominant_experts == e_idx)
            count = np.sum(mask)
            color = self.custom_colors[e_idx % len(self.custom_colors)]
            
            if count < 10: continue
                
            v_ego_sub, v_lead_sub, gap_sub = x_vals[mask], y_vals[mask], z_vals[mask]
            rel_v_sub = rel_vals[mask]
            
            # 统一采样上限 2500，以保证和 2D scatter 使用绝对一致的数据量与子集
            if count > 2500:
                idx_s = np.random.choice(count, 2500, replace=False)
                rel_v_kde = rel_v_sub[idx_s]
                gap_kde = gap_sub[idx_s]
            else:
                rel_v_kde = rel_v_sub
                gap_kde = gap_sub
                
            rel_v_kde = rel_v_kde + np.random.normal(0, 1e-3, len(rel_v_kde))
            gap_kde = gap_kde + np.random.normal(0, 1e-3, len(gap_kde))

            try:
                fit_pts = np.vstack([rel_v_kde, gap_kde])
                density = self._fast_kde_2d_gpu(fit_pts, pos_rel_gap, bw=0.4)
                Z_1 = np.reshape(density, X_rel.shape)
                
                expert_surfaces[e_idx] = Z_1

                fig_exp_surf = plt.figure(figsize=(10, 8))
                fig_exp_surf.canvas.manager.set_window_title(f'KDE Surface - {expert_name}')
                ax_surf = fig_exp_surf.add_subplot(111, projection='3d')

                ax_surf.plot_surface(X_rel, Y_gap, Z_1, color=color, alpha=0.35, edgecolor='none', shade=False)
                ax_surf.plot_wireframe(X_rel, Y_gap, Z_1, color=color, alpha=0.7, linewidth=0.8)

                ax_surf.set_xlabel(r'Relative Speed ($\Delta v$, m/s)', fontsize=12, labelpad=10)
                ax_surf.set_ylabel('Spacing ($gap$, m)', fontsize=12, labelpad=10)
                ax_surf.set_zlabel('Kernel Density', fontsize=12, labelpad=10)
                
                ax_surf.set_xlim(-8, 8)
                ax_surf.set_title(fr'[{expert_name}] 3D KDE Phase Surface ($\Delta v$ vs Spacing)', fontsize=18, fontweight='bold', pad=25, color='#C0392B')

                ax_surf.view_init(elev=30, azim=-135) 
                ax_surf.xaxis.pane.fill = False; ax_surf.yaxis.pane.fill = False; ax_surf.zaxis.pane.fill = False
                ax_surf.grid(True, linestyle=':', alpha=0.5)
                ax_surf.set_zlim(bottom=0)

                plt.tight_layout()
                
                safe_name = expert_name.replace("(", "_").replace(")", "")
                surf_filename = f"expert_kde_surface_{safe_name}.png"
                fig_exp_surf.savefig(surf_filename, bbox_inches='tight', dpi=300)
                print(f"   >> ✅ 成功渲染专属专家曲面图: {surf_filename}")
                plt.close(fig_exp_surf) 
                
            except Exception as e:
                print(f"   >> ⚠️ 专家 {expert_name} 曲面图生成失败: {e}")
                traceback.print_exc()

            try:
                from scipy.stats import gaussian_kde
                
                kde_rel = gaussian_kde(rel_v_kde, bw_method=0.4)
                rel_grid = np.linspace(-8, 8, 500) 
                z_vals_rel = kde_rel(rel_grid)
                global_max_z_rel = max(global_max_z_rel, np.max(z_vals_rel))
                
                kde_z = gaussian_kde(gap_kde, bw_method=0.4)
                gap_grid = np.linspace(0, 30, 500)
                z_vals_gap = kde_z(gap_grid)
                global_max_z_gap = max(global_max_z_gap, np.max(z_vals_gap))
                
                waterfall_data[e_idx] = {
                    'rel_grid': rel_grid, 'z_vals_rel': z_vals_rel,
                    'gap_grid': gap_grid, 'z_vals_gap': z_vals_gap,
                    'color': color,
                    'peak_gap': gap_grid[np.argmax(z_vals_gap)] 
                }
            except Exception: 
                pass

        valid_e_idxs = list(expert_surfaces.keys())
        if len(valid_e_idxs) >= 4:
            valid_e_idxs.sort(key=lambda idx: waterfall_data[idx]['peak_gap'])
            step = len(valid_e_idxs) / 4.0
            selected_4 = [valid_e_idxs[int(i * step)] for i in range(4)]
        else:
            selected_4 = valid_e_idxs

        fig_comb4 = plt.figure(figsize=(12, 10))
        fig_comb4.canvas.manager.set_window_title('Combined 4 Experts KDE Surface')
        ax_comb4 = fig_comb4.add_subplot(111, projection='3d')
        comb4_patches = []

        for e_idx in selected_4:
            Z_1 = expert_surfaces[e_idx]
            color = waterfall_data[e_idx]['color']
            expert_name = self.private_expert_names[e_idx]
            
            ax_comb4.plot_surface(X_rel, Y_gap, Z_1, color=color, alpha=0.35, edgecolor='none', shade=False)
            ax_comb4.plot_wireframe(X_rel, Y_gap, Z_1, color=color, alpha=0.5, linewidth=0.6)
            comb4_patches.append(mpatches.Patch(color=color, label=f"{expert_name}"))

        ax_comb4.set_xlabel(r'Relative Speed ($\Delta v$, m/s)', fontsize=12, labelpad=10)
        ax_comb4.set_ylabel('Spacing ($gap$, m)', fontsize=12, labelpad=10)
        ax_comb4.set_zlabel('Kernel Density', fontsize=12, labelpad=10)
        ax_comb4.set_xlim(-8, 8)
        ax_comb4.set_title('Combined 3D KDE Surfaces (4 Distinct Experts)', fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
        ax_comb4.view_init(elev=30, azim=-135)
        ax_comb4.xaxis.pane.fill = False; ax_comb4.yaxis.pane.fill = False; ax_comb4.zaxis.pane.fill = False
        ax_comb4.grid(True, linestyle=':', alpha=0.5)
        ax_comb4.legend(handles=comb4_patches, loc='upper right', fontsize=12)
        fig_comb4.tight_layout()
        
        save_comb4 = "expert_kde_surface_combined_4.png"
        fig_comb4.savefig(save_comb4, bbox_inches='tight', dpi=300)
        print(f"   >> ✅ 成功渲染【精选防遮挡四大专家】合并相空间对比图: {save_comb4}")
        fig_comb4.show()

        fig_water_rel = plt.figure(figsize=(10, 10))
        fig_water_rel.canvas.manager.set_window_title('Waterfall Probability Curves - Relative Speed')
        ax_kde_rel = fig_water_rel.add_subplot(111, projection='3d')

        fig_water_gap = plt.figure(figsize=(10, 10))
        fig_water_gap.canvas.manager.set_window_title('Waterfall Probability Curves - Spacing')
        ax_kde_gap = fig_water_gap.add_subplot(111, projection='3d')

        waterfall_patches = []

        for e_idx in range(7, -1, -1):
            if e_idx not in waterfall_data: continue
            data_dict = waterfall_data[e_idx]
            color = data_dict['color']
            expert_name = self.private_expert_names[e_idx]
            
            waterfall_patches.append(mpatches.Patch(color=color, label=f"{expert_name}"))
            
            y_pos_line = e_idx * spacing_factor
            y_pos_poly = y_pos_line + 0.1
            
            rel_grid = data_dict['rel_grid']
            z_vals_rel = data_dict['z_vals_rel']

            ax_kde_rel.plot(rel_grid, np.full_like(rel_grid, y_pos_line), z_vals_rel, color=color, alpha=1.0, linewidth=2.0, zorder=10-e_idx)
            max_idx_rel = np.argmax(z_vals_rel)
            max_rel, max_z_rel = rel_grid[max_idx_rel], z_vals_rel[max_idx_rel]
            ax_kde_rel.scatter([max_rel], [y_pos_line], [max_z_rel], color=color, marker='o', s=30, zorder=10-e_idx)
            ax_kde_rel.text(max_rel, y_pos_line, max_z_rel + global_max_z_rel * 0.05, f'{max_rel:.1f}', ha='center', va='bottom', fontsize=10, zorder=10-e_idx)
            verts_rel = [(rel_grid[0], 0)] + list(zip(rel_grid, z_vals_rel)) + [(rel_grid[-1], 0)]
            poly_rel = PolyCollection([verts_rel], facecolors=mcolors.to_rgba(color, alpha=0.35), edgecolors='none', zorder=10-e_idx)
            ax_kde_rel.add_collection3d(poly_rel, zs=y_pos_poly, zdir='y')
            
            gap_grid = data_dict['gap_grid']
            z_vals_gap = data_dict['z_vals_gap']
            
            ax_kde_gap.plot(gap_grid, np.full_like(gap_grid, y_pos_line), z_vals_gap, color=color, alpha=1.0, linewidth=2.0, zorder=10-e_idx)
            max_idx_gap = np.argmax(z_vals_gap)
            max_gap, max_z_gap = gap_grid[max_idx_gap], z_vals_gap[max_idx_gap]
            ax_kde_gap.scatter([max_gap], [y_pos_line], [max_z_gap], color=color, marker='o', s=30, zorder=10-e_idx)
            ax_kde_gap.text(max_gap, y_pos_line, max_z_gap + global_max_z_gap * 0.05, f'{max_gap:.1f}', ha='center', va='bottom', fontsize=10, zorder=10-e_idx)
            verts_gap = [(gap_grid[0], 0)] + list(zip(gap_grid, z_vals_gap)) + [(gap_grid[-1], 0)]
            poly_gap = PolyCollection([verts_gap], facecolors=mcolors.to_rgba(color, alpha=0.35), edgecolors='none', zorder=10-e_idx)
            ax_kde_gap.add_collection3d(poly_gap, zs=y_pos_poly, zdir='y')

        y_ticks_pos = np.arange(8) * spacing_factor
        waterfall_elev = 30
        waterfall_azim = -35

        ax_kde_rel.set_title(r'Joyplot: Probability Density of Relative Speed ($\Delta v$)', fontsize=18, fontweight='bold', pad=25)
        ax_kde_rel.set_xlabel(r'Relative Speed ($\Delta v$, m/s)', fontsize=14, labelpad=15)
        ax_kde_rel.set_ylabel('') 
        ax_kde_rel.set_zlabel('Probability Density', fontsize=14, labelpad=15)
        ax_kde_rel.tick_params(axis='z', which='major', pad=5, labelsize=10) 
        ax_kde_rel.set_yticks(y_ticks_pos)
        ax_kde_rel.set_yticklabels(self.private_expert_names[:8], fontsize=12, rotation=-45, ha='left', va='center')
        ax_kde_rel.tick_params(axis='y', which='major', pad=5, length=8, direction='out') 
        ax_kde_rel.set_ylim(-2.0, 8 * spacing_factor) 
        ax_kde_rel.set_zlim(0, global_max_z_rel * 1.15)
        ax_kde_rel.set_xlim(-8, 8) 
        ax_kde_rel.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_kde_rel.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_kde_rel.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_kde_rel.grid(True, linestyle='--', alpha=0.3) 
        ax_kde_rel.view_init(elev=waterfall_elev, azim=waterfall_azim)
        
        ax_kde_gap.set_title('Joyplot: Probability Density of Spacing ($gap$)', fontsize=18, fontweight='bold', pad=25)
        ax_kde_gap.set_xlabel('Spacing ($gap$, m)', fontsize=14, labelpad=15)
        ax_kde_gap.set_ylabel('')
        ax_kde_gap.set_zlabel('Probability Density', fontsize=14, labelpad=15)
        ax_kde_gap.tick_params(axis='z', which='major', pad=5, labelsize=10) 
        ax_kde_gap.set_yticks(y_ticks_pos)
        ax_kde_gap.set_yticklabels(self.private_expert_names[:8], fontsize=12, rotation=-45, ha='left', va='center')
        ax_kde_gap.tick_params(axis='y', which='major', pad=5, length=8, direction='out') 
        ax_kde_gap.set_ylim(-2.0, 8 * spacing_factor) 
        ax_kde_gap.set_zlim(0, global_max_z_gap * 1.15)
        ax_kde_gap.set_xlim(0, 30)
        ax_kde_gap.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_kde_gap.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_kde_gap.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_kde_gap.grid(True, linestyle='--', alpha=0.3)
        ax_kde_gap.view_init(elev=waterfall_elev, azim=waterfall_azim)

        waterfall_patches.reverse()
        
        fig_water_rel.legend(handles=waterfall_patches, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12, frameon=True)
        fig_water_rel.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
        fig_water_rel.savefig("expert_kde_waterfall_rel_speed.png", bbox_inches='tight', dpi=300)

        fig_water_gap.legend(handles=waterfall_patches, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12, frameon=True)
        fig_water_gap.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
        fig_water_gap.savefig("expert_kde_waterfall_spacing.png", bbox_inches='tight', dpi=300)

        print("\n✅ 成功渲染极致学术感 Joyplot 分析！图表已分离保存至当前目录:")
        print("   -> 1. 多张各专家独立的相空间曲面图 (Delta V ∈ [-8, 8] vs Spacing): expert_kde_surface_*.png")
        print("   -> 2. 独立输出瀑布图 (Relative Speed): expert_kde_waterfall_rel_speed.png")
        print("   -> 3. 独立输出瀑布图 (Spacing): expert_kde_waterfall_spacing.png")
        print("   -> 4. 新增四专家合并无遮挡对比图: expert_kde_surface_combined_4.png")

        fig_water_rel.show()
        fig_water_gap.show()

    def plot_expert_scatter_2d(self, provided_data=None):
        if provided_data is None:
            raw_data = self._collect_rollout_frames()
            data = self._filter_valid_data(raw_data)
        else:
            data = provided_data
            
        rel_v = data['v_lead'] - data['v_ego']
        gap = data['gap']
        dominant_experts = data['dominant_experts']
        
        print("\n⏳ 正在利用【GPU KDE + 连通域碎片过滤 + 高斯抗锯齿寻界】渲染顶级论文相空间...")
        
        fig_regions, ax_regions = plt.subplots(figsize=(13, 9))
        fig_regions.canvas.manager.set_window_title('Phase Space Partition (Regions Only)')
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(13, 9))
        fig_scatter.canvas.manager.set_window_title('Phase Space Partition with Scatter')
        
        xx, yy = np.meshgrid(np.linspace(-8.5, 8.5, 400), np.linspace(0, 35, 300))
        grid_pts = np.vstack([xx.ravel(), yy.ravel()])
        probs = np.zeros((8, grid_pts.shape[1]))
        
        total_pts = len(rel_v)
        valid_experts = []
        
        for e_idx in range(8):
            mask = (dominant_experts == e_idx)
            count = np.sum(mask)
            if count < 10: continue
            
            valid_experts.append(e_idx)
            expert_pts = np.vstack([rel_v[mask], gap[mask]])
            
            if count > 2500:
                idx_s = np.random.choice(count, 2500, replace=False)
                expert_pts = expert_pts[:, idx_s]
            
            try:
                density = self._fast_kde_2d_gpu(expert_pts, grid_pts, bw=0.3)
                prior = count / total_pts
                probs[e_idx, :] = density * prior + 1e-12 * prior
            except Exception as e:
                traceback.print_exc()
                
        Z_raw = np.argmax(probs, axis=0).reshape(xx.shape)
        Z_cleaned = Z_raw.copy()
        
        transition_label = 8 
        
        for e_idx in valid_experts:
            mask = (Z_raw == e_idx)
            labeled_array, num_features = scipy.ndimage.label(mask)
            if num_features > 0:
                sizes = scipy.ndimage.sum(mask, labeled_array, range(1, num_features + 1))
                max_area = np.max(sizes)
                area_threshold = max_area * 0.5
                
                for i, size in enumerate(sizes):
                    if size < area_threshold:
                        small_island_mask = (labeled_array == i + 1)
                        Z_cleaned[small_island_mask] = transition_label
        
        cmap_colors = [self.custom_colors[i % len(self.custom_colors)] for i in range(8)]
        cmap_colors.append('#E8E8E8') 
        cmap = mcolors.ListedColormap(cmap_colors)
        bounds = np.arange(-0.5, 9.5, 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        for ax in [ax_regions, ax_scatter]:
            ax.pcolormesh(xx, yy, Z_cleaned, cmap=cmap, norm=norm, alpha=0.35, zorder=0, shading='nearest')
            
            for e_idx in valid_experts + [transition_label]:
                Z_binary = (Z_cleaned == e_idx).astype(float)
                if np.max(Z_binary) > 0:
                    Z_smooth = scipy.ndimage.gaussian_filter(Z_binary, sigma=1.5)
                    ax.contour(xx, yy, Z_smooth, levels=[0.5], colors='#7F8C8D', linewidths=0.8, linestyles='solid', alpha=0.6, zorder=1)
        
        # 散点图的采样绘制，由于底层的 KDE 已经使用了统一的 2500 样本，这里的 1000 散点只会从中继续下采，完美对应要求。
        max_samples_per_expert = 1000 
        
        for e_idx in range(8):
            if e_idx not in valid_experts: continue
            
            expert_name = self.private_expert_names[e_idx]
            mask = (dominant_experts == e_idx)
            count = np.sum(mask)
            color = self.custom_colors[e_idx % len(self.custom_colors)]
            
            grid_mask = (Z_cleaned == e_idx)
            if np.any(grid_mask):
                labeled_array, num_features = scipy.ndimage.label(grid_mask)
                if num_features > 0:
                    sizes = scipy.ndimage.sum(grid_mask, labeled_array, range(1, num_features + 1))
                    max_label = np.argmax(sizes) + 1
                    largest_mask = (labeled_array == max_label)
                    
                    xs = xx[largest_mask]
                    ys = yy[largest_mask]
                    cx, cy = np.mean(xs), np.mean(ys)
                    dists = (xs - cx)**2 + (ys - cy)**2
                    best_idx = np.argmin(dists)
                    peak_x, peak_y = xs[best_idx], ys[best_idx]
                    
                    ax_regions.text(
                        peak_x, peak_y, expert_name, 
                        fontsize=13, fontweight='bold', color='black',
                        ha='center', va='center', zorder=5,
                        bbox=dict(facecolor='white', alpha=0.85, edgecolor=color, linewidth=1.5, boxstyle='round,pad=0.4')
                    )
            
            expert_rel_v = rel_v[mask]
            expert_gap = gap[mask]
            if count > max_samples_per_expert:
                sampled_indices = np.random.choice(count, max_samples_per_expert, replace=False)
                expert_rel_v = expert_rel_v[sampled_indices]
                expert_gap = expert_gap[sampled_indices]
                
            ax_scatter.scatter(expert_rel_v, expert_gap, c=[color], alpha=0.8, s=15, edgecolors='white', linewidths=0.2, label=expert_name, zorder=2)
            
        legend_patches = []
        for e_idx in range(8):
            if e_idx in valid_experts:
                color = self.custom_colors[e_idx % len(self.custom_colors)]
                expert_name = self.private_expert_names[e_idx]
                legend_patches.append(mpatches.Patch(color=color, alpha=0.35, label=expert_name))
                
        if np.any(Z_cleaned == transition_label):
            legend_patches.append(mpatches.Patch(color='#E8E8E8', alpha=0.8, label="Transition Region", edgecolor='gray'))

        for ax in [ax_regions, ax_scatter]:
            ax.set_xlabel(r'Difference of Velocity ($\Delta v = v_{lead} - v_{ego}$, m/s)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Front to Rear Distance ($gap$, m)', fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.6, zorder=1) 
            
            ax.set_xlim(-8, 8) 
            ax.set_ylim(0, 32)
            
        ax_regions.set_title('Non-linear Phase Space Partition (Regions Only)', fontsize=18, fontweight='bold', pad=20)
        ax_regions.legend(handles=legend_patches, title='Dominant Expert', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12, title_fontsize=14)

        ax_scatter.set_title('Non-linear Phase Space Partition with Scatter Points', fontsize=18, fontweight='bold', pad=20)
        ax_scatter.legend(title='Dominant Expert', loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12, title_fontsize=14)
        
        fig_regions.tight_layout()
        fig_scatter.tight_layout()
        
        save_regions = "expert_wiedemann_phase_space_regions.png"
        save_scatter = "expert_wiedemann_phase_space_scatter.png"
        fig_regions.savefig(save_regions, bbox_inches='tight', dpi=300)
        fig_scatter.savefig(save_scatter, bbox_inches='tight', dpi=300)
        
        print(f"   >> ✅ 成功保存抗锯齿区域划分图 (无散点): {save_regions}")
        print(f"   >> ✅ 成功保存抗锯齿相空间分工图 (带散点): {save_scatter}")
        
        fig_regions.show()
        fig_scatter.show()

    def plot_scene_raincloud(self):
        raw_data = self._collect_rollout_frames()
        data = self._filter_valid_data(raw_data)
        phys_feats_np = data['phys_feats']
        scene_ids = data['scene_ids']
        
        feature_names = ['Avg Speed', 'Speed Std', 'Avg Acc', 'Inv TTC', 'Avg Jerk', 'Time Headway', 'Delta Inv TTC', 'Jerk Std', 'Acc Std']
        scale_factors = np.array([30.0, 5.0, 3.0, 1.0, 5.0, 2.0, 0.5, 5.0, 3.0])
        
        unique_scenes = np.unique(scene_ids)
        scene_names_map = {
            s_id: self.scene_names.get(int(s_id), f'Scene {int(s_id) + 1}')
            for s_id in unique_scenes
        }

        print(f"\n⏳ 准备生成 {len(unique_scenes)} 个场景的【以场景为中心】九维物理特征云雨图...")
        max_samples = 600 
        num_features = len(feature_names)
        
        for s_idx, s_id in enumerate(unique_scenes):
            mask = (scene_ids == s_id)
            scene_feats = phys_feats_np[mask]
            
            if len(scene_feats) == 0: continue
            
            scene_feats = scene_feats * scale_factors
            if len(scene_feats) > max_samples:
                idx_sampled = np.random.choice(len(scene_feats), max_samples, replace=False)
                scene_feats = scene_feats[idx_sampled]
                
            fig, axes = plt.subplots(1, num_features, figsize=(20, 8))
            fig.canvas.manager.set_window_title(f'Adaptive Raincloud Plot - {scene_names_map[s_id]}')
            
            box_width = 0.15
            violin_width = 0.45
            box_pos = -0.05
            violin_pos = 0.05
            scatter_pos = -0.15
            
            for i, (category, ax) in enumerate(zip(feature_names, axes)):
                data_points = scene_feats[:, i].copy() 
                
                color = self.custom_colors[i % len(self.custom_colors)]
                
                ax.boxplot(
                    data_points, positions=[box_pos], widths=box_width, patch_artist=True,
                    showfliers=False, notch=True,
                    medianprops={'color': 'black', 'linewidth': 3},
                    boxprops={'facecolor': color, 'edgecolor': color, 'linewidth': 3},
                    whiskerprops={'color': color, 'linewidth': 3},
                    capprops={'color': color, 'linewidth': 3}
                )
                
                if np.std(data_points) > 1e-5:
                    violin = ax.violinplot(
                        data_points, positions=[violin_pos], widths=violin_width,
                        showmeans=False, showmedians=False, showextrema=False
                    )
                    for pc in violin['bodies']:
                        pc.set_facecolor(color)
                        pc.set_edgecolor(color)
                        pc.set_alpha(0.35)
                        vertices = pc.get_paths()[0].vertices
                        vertices[:, 0] = np.where(vertices[:, 0] > violin_pos, vertices[:, 0], violin_pos)
                
                ax.scatter(
                    np.random.normal(scatter_pos, 0.02, len(data_points)), data_points,
                    color=color, alpha=0.7, s=30, edgecolor='white', linewidth=0.5, zorder=3
                )
                
                ax.set_xticks([0]) 
                ax.set_xticklabels([category], fontsize=13, rotation=25, ha='right')
                ax.tick_params(axis='y', labelsize=12)
                
                if i == 0:
                    ax.set_ylabel('Physical Value', fontsize=18)
                
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_color('black')
                ax.spines['left'].set_color('black')
                
                ax.set_xlim(-0.35, 0.35)
                
            scene_label = scene_names_map[s_id]
            plt.suptitle(f'Adaptive Raincloud Plots of 9D Features - {scene_label}', y=1.02, fontsize=22, fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.35)
            
            save_filename = f"scene_raincloud_{s_idx+1}_adaptive.png"
            fig.savefig(save_filename, bbox_inches='tight', dpi=300)
            print(f"   >> ✅ 成功渲染并保存: {save_filename}")
            
            fig.show()

    def plot_feature_raincloud_across_scenes(self):
        raw_data = self._collect_rollout_frames()
        data = self._filter_valid_data(raw_data)
        phys_feats_np = data['phys_feats']
        scene_ids = data['scene_ids']
        
        feature_names = ['Avg Speed', 'Speed Std', 'Avg Acc', 'Inv TTC', 'Avg Jerk', 'Time Headway', 'Delta Inv TTC', 'Jerk Std', 'Acc Std']
        scale_factors = np.array([30.0, 5.0, 3.0, 1.0, 5.0, 2.0, 0.5, 5.0, 3.0])
        
        unique_scenes = np.unique(scene_ids)
        scene_names_map = {
            s_id: self.scene_names.get(int(s_id), f'Scene {int(s_id) + 1}')
            for s_id in unique_scenes
        }

        print(f"\n⏳ 准备按物理特征维度生成【跨场景】的云雨图 (总共 {len(feature_names)} 张)...")
        max_samples = 800 
        
        for feat_idx, category in enumerate(feature_names):
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.canvas.manager.set_window_title(f'Feature Across Scenes - {category}')
            
            x_labels = []
            x_positions = np.arange(len(unique_scenes))
            
            box_width = 0.15
            violin_width = 0.45
            
            for i, s_id in enumerate(unique_scenes):
                mask = (scene_ids == s_id)
                scene_feats = phys_feats_np[mask, feat_idx] * scale_factors[feat_idx]
                
                if len(scene_feats) == 0: continue
                
                if len(scene_feats) > max_samples:
                    scene_feats = np.random.choice(scene_feats, max_samples, replace=False)
                    
                color = self.scene_colors[i % len(self.scene_colors)]
                scene_label = scene_names_map[s_id]
                x_labels.append(scene_label)
                
                base_pos = x_positions[i]
                box_pos = base_pos - 0.05
                violin_pos = base_pos + 0.05
                scatter_pos = base_pos - 0.15
                
                ax.boxplot(
                    scene_feats, positions=[box_pos], widths=box_width, patch_artist=True,
                    showfliers=False, notch=True,
                    medianprops={'color': 'black', 'linewidth': 2},
                    boxprops={'facecolor': color, 'edgecolor': color, 'linewidth': 2},
                    whiskerprops={'color': color, 'linewidth': 2},
                    capprops={'color': color, 'linewidth': 2}
                )
                
                if np.std(scene_feats) > 1e-5:
                    violin = ax.violinplot(
                        scene_feats, positions=[violin_pos], widths=violin_width,
                        showmeans=False, showmedians=False, showextrema=False
                    )
                    for pc in violin['bodies']:
                        pc.set_facecolor(color)
                        pc.set_edgecolor(color)
                        pc.set_alpha(0.4)
                        # 将小提琴左半边切掉
                        vertices = pc.get_paths()[0].vertices
                        vertices[:, 0] = np.where(vertices[:, 0] > violin_pos, vertices[:, 0], violin_pos)
                
                ax.scatter(
                    np.random.normal(scatter_pos, 0.03, len(scene_feats)), scene_feats,
                    color=color, alpha=0.6, s=15, edgecolor='white', linewidth=0.3, zorder=3
                )
                
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=14, fontweight='bold')
            ax.tick_params(axis='y', labelsize=12)
            
            unit = ""
            if "Speed" in category: unit = " (m/s)"
            elif "Acc" in category: unit = " (m/s²)"
            elif "Jerk" in category: unit = " (m/s³)"
            elif "Time Headway" in category: unit = " (s)"
            elif "Inv TTC" in category: unit = " (s⁻¹)"
            
            ax.set_ylabel(f'{category}{unit}', fontsize=16, fontweight='bold')
            ax.set_title(f'Distribution of {category} Across Identified Scenes', fontsize=18, fontweight='bold', pad=15)
            
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            
            plt.tight_layout()
            
            safe_category_name = category.replace(" ", "_")
            save_filename = f"feature_raincloud_{feat_idx+1}_{safe_category_name}.png"
            fig.savefig(save_filename, bbox_inches='tight', dpi=300)
            print(f"   >> ✅ 成功渲染并保存跨场景对比图: {save_filename}")
            
            fig.show()

    def _interactive_loop(self):
        while True:
            try:
                print("\n" + "="*65)
                print("【论文图表生成与可视化控制台】")
                print(f"  [1] 输入车辆编号 (0 到 {self.total_samples-1}) -> 特定车辆轨迹多专家对比图")
                print(f"  [2] 输入 'scene'                     -> 绘制 9维场景聚类PCA 3D边界图")
                print(f"  [3] 输入 'expert'                    -> 绘制 8大私有专家散点划分图及输出定量表格")
                print(f"  [4] 输入 'kde'                       -> 🌟绘制双子瀑布图与防遮挡合并曲面图")
                print(f"  [5] 输入 'scene_adv'                 -> 绘制场景高级分析图 (Rank-Based 雷达图)")
                print(f"  [6] 输入 'raincloud'                 -> 绘制场景中心的九维特征云雨图")
                print(f"  [7] 输入 'feat_raincloud'            -> 绘制 9 大物理特征的跨场景云雨图")
                print(f"  [8] 输入 'expert_scatter'            -> 绘制抗锯齿/去碎片的【Wiedemann 式相空间图】")
                print(f"  [9] 输入 'kde and expert_scatter'    -> 🔥同步执行 [4] 和 [8] (保证拟合样本点绝对一致)")
                print(f"  [10] 输入 'q'                        -> 退出可视化程序")
                user_input = input("👉 请输入您的指令 (例如 'kde and expert_scatter'): ").strip().lower()
                
                if user_input == 'q':
                    print("🚪 退出可视化工具...")
                    plt.ioff()
                    plt.close('all')
                    break
                elif user_input == 'scene':
                    self.plot_scene_3d()
                    plt.pause(0.1)
                    continue
                elif user_input == 'expert':
                    self.plot_expert_domain_3d()
                    plt.pause(0.1)
                    continue
                elif user_input == 'kde':
                    self.plot_expert_kde_3d()
                    plt.pause(0.1)
                    continue
                elif user_input == 'scene_adv':
                    self.plot_scene_advanced_analysis()
                    plt.pause(0.1)
                    continue
                elif user_input == 'raincloud':
                    self.plot_scene_raincloud()
                    plt.pause(0.1)
                    continue
                elif user_input == 'feat_raincloud':
                    self.plot_feature_raincloud_across_scenes()
                    plt.pause(0.1)
                    continue
                elif user_input == 'expert_scatter':
                    self.plot_expert_scatter_2d()
                    plt.pause(0.1)
                    continue
                elif user_input == 'kde and expert_scatter':
                    print("\n⏳ [同步模式] 正在统一收集数据并进行同步抽样锁定，保证 3D曲面 和 2D散点图 样本绝对对应...")
                    raw_data = self._collect_rollout_frames()
                    data = self._filter_valid_data(raw_data)
                    sync_data = self._get_sync_sampled_data(data, max_pts=2500) # 将样本池直接锁定并分配
                    self.plot_expert_kde_3d(provided_data=sync_data)
                    self.plot_expert_scatter_2d(provided_data=sync_data)
                    plt.pause(0.1)
                    continue
                    
                if user_input.isdigit():
                    idx = int(user_input)
                    if 0 <= idx < self.total_samples:
                        print(f"⏳ 正在分离计算各基准专家以及 MoE 的长程闭环轨迹...")
                        self.plot_sample(idx)
                        plt.pause(0.1) 
                    else:
                        print("❌ 编号越界。")
            except Exception as e:
                traceback.print_exc()
                print(f"❌ 发生未知异常: {e}")


