// PMSF-V1 主应用
const PMSFApp = {
  currentTab: 'dashboard',

  async init() {
    this.bindTabs();
    this.bindSearch();
    window.addEventListener('resize', () => PMSFCharts.resizeAll());

    try {
      await PMSFData.loadAll();
      this.renderAll();
    } catch (e) {
      console.error('数据加载失败:', e);
      this.showError();
    }
  },

  bindTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        this.switchTab(tab);
      });
    });
  },

  switchTab(tab) {
    this.currentTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === 'tab-' + tab));
    setTimeout(() => PMSFCharts.resizeAll(), 100);
  },

  bindSearch() {
    const input = document.getElementById('searchIssue');
    if (input) {
      input.addEventListener('input', () => this.filterHistory(input.value));
    }
  },

  renderAll() {
    this.renderHeader();
    this.renderDashboard();
    this.renderHistory();
    this.renderPrediction();
    this.renderModel();
    this.renderReport();
    this.renderRuntime();
  },

  renderHeader() {
    const d = PMSFData;
    const updateTime = d.runtime ? d.runtime.last_update : new Date().toLocaleString('zh-CN');
    document.getElementById('updateTime').textContent = '数据更新: ' + updateTime;
  },

  renderDashboard() {
    const d = PMSFData;
    const s = d.stats || {};

    document.getElementById('statTotalPeriods').textContent = s.total_periods || d.history.length || '--';
    document.getElementById('statDateRange').textContent = s.date_range || '--';

    if (d.latest) {
      const cs = d.latest.current_state || {};
      document.getElementById('statCurrentState').textContent = cs.state_name || cs.state || '--';
      document.getElementById('statStateDuration').textContent = cs.state ? `状态 ${cs.state}` : '';
      document.getElementById('statLatestIssue').textContent = d.latest.target_issue || '--';
    }

    // 中奖统计
    const prizes = d.predictions || [];
    const hitCount = prizes.filter(p => (p.front_hit || 0) + (p.back_hit || 0) > 0).length;
    document.getElementById('statTotalPrize').textContent = hitCount;
    document.getElementById('statHitRate').textContent = prizes.length ? `命中率 ${(hitCount / prizes.length * 100).toFixed(1)}%` : '暂无数据';

    // 图表
    if (s.front_freq) PMSFCharts.renderHotNumbers(s.front_freq);
    if (d.history) PMSFCharts.renderSumTrend(d.history);
    if (s.odd_even_dist) PMSFCharts.renderOddEven(s.odd_even_dist);
    if (d.latest && d.latest.current_state) PMSFCharts.renderStateProb(d.latest.current_state.probabilities);

    // 最新4组
    this.renderLatestPredictions();
  },

  renderLatestPredictions() {
    const container = document.getElementById('latestPredictions');
    if (!container || !PMSFData.latest || !PMSFData.latest.groups) return;

    const tags = { A: 'tag-a', B: 'tag-b', C: 'tag-c', D: 'tag-d' };
    container.innerHTML = PMSFData.latest.groups.map(g => {
      const struct = g.structure || {};
      return `
        <div class="prediction-group">
          <div class="group-header">
            <div class="group-label">
              <span class="tag ${tags[g.label] || ''}">${g.label}组</span>
              ${g.name}
            </div>
          </div>
          <div class="group-balls">
            ${PMSFData.formatBalls(g.front, 'front')}
            <div class="divider"></div>
            ${PMSFData.formatBalls(g.back, 'back')}
          </div>
          <div class="group-meta">
            <span class="meta-item">${struct.odd_even || ''}</span>
            <span class="meta-item">${struct.big_small || ''}</span>
            <span class="meta-item">和值 ${struct.sum || ''}</span>
            <span class="meta-item">跨度 ${struct.span || ''}</span>
          </div>
        </div>
      `;
    }).join('');
  },

  renderHistory() {
    const d = PMSFData;
    const s = d.stats || {};

    if (s.front_freq) PMSFCharts.renderFrontFreq(s.front_freq);
    if (s.back_freq) PMSFCharts.renderBackFreq(s.back_freq);
    if (s.sum_dist) PMSFCharts.renderSumDist(s.sum_dist);
    if (d.history) PMSFCharts.renderSpanDist(d.history);
    if (s.big_small_dist) PMSFCharts.renderBigSmall(s.big_small_dist);
    if (s.zone_dist_top10) PMSFCharts.renderZone(s.zone_dist_top10);

    // 表格
    const tbody = document.querySelector('#historyTable tbody');
    if (tbody && d.history) {
      const sorted = [...d.history].reverse();
      document.getElementById('historyCount').textContent = `共 ${sorted.length} 期`;
      tbody.innerHTML = sorted.slice(0, 200).map(row => `
        <tr>
          <td><strong>${row.issue}</strong></td>
          <td>${row.date || ''}</td>
          <td>${PMSFData.formatBalls([row.front01, row.front02, row.front03, row.front04, row.front05], 'front')}</td>
          <td>${PMSFData.formatBalls([row.back01, row.back02], 'back')}</td>
          <td>${row.sum_front || ''}</td>
          <td>${row.span_front || ''}</td>
          <td>${row.odd_even || ''}</td>
          <td>${row.big_small || ''}</td>
          <td>${row.zone || ''}</td>
        </tr>
      `).join('');
      this._allHistory = sorted;
    }
  },

  filterHistory(keyword) {
    const tbody = document.querySelector('#historyTable tbody');
    if (!tbody || !this._allHistory) return;
    const filtered = keyword ? this._allHistory.filter(r => r.issue.includes(keyword)) : this._allHistory;
    tbody.innerHTML = filtered.slice(0, 200).map(row => `
      <tr>
        <td><strong>${row.issue}</strong></td>
        <td>${row.date || ''}</td>
        <td>${PMSFData.formatBalls([row.front01, row.front02, row.front03, row.front04, row.front05], 'front')}</td>
        <td>${PMSFData.formatBalls([row.back01, row.back02], 'back')}</td>
        <td>${row.sum_front || ''}</td>
        <td>${row.span_front || ''}</td>
        <td>${row.odd_even || ''}</td>
        <td>${row.big_small || ''}</td>
        <td>${row.zone || ''}</td>
      </tr>
    `).join('');
  },

  renderPrediction() {
    const d = PMSFData;

    // 最新预测详情
    const detail = document.getElementById('predictionDetail');
    if (detail && d.latest) {
      document.getElementById('predIssueLabel').textContent = `目标期号: ${d.latest.target_issue}`;
      const cs = d.latest.current_state || {};
      detail.innerHTML = `
        <div style="margin-bottom:20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
          <span class="state-badge state-${cs.state || 'B'}">${cs.state_name || ''} (${cs.state || ''})</span>
          <span style="color:var(--text-secondary);font-size:13px;">
            A: ${((cs.probabilities || {}).A || 0) * 100 | 0}% | 
            B: ${((cs.probabilities || {}).B || 0) * 100 | 0}% | 
            C: ${((cs.probabilities || {}).C || 0) * 100 | 0}%
          </span>
        </div>
        <div class="grid grid-4">
          ${(d.latest.groups || []).map(g => this._groupCard(g)).join('')}
        </div>
      `;
    }

    // 中奖台账
    const tbody = document.querySelector('#prizeTable tbody');
    if (tbody && d.predictions) {
      const sorted = [...d.predictions].reverse();
      tbody.innerHTML = sorted.map(p => {
        const prize = PMSFData.calcPrize(p.front_hit || 0, p.back_hit || 0);
        return `
          <tr>
            <td><strong>${p.issue}</strong></td>
            <td><span class="tag tag-${p.group ? p.group.toLowerCase() : ''}" style="padding:2px 8px;border-radius:4px;font-size:11px;">${p.group}组</span></td>
            <td>${PMSFData.formatBalls(p.predicted_front || [], 'front')}</td>
            <td>${PMSFData.formatBalls(p.predicted_back || [], 'back')}</td>
            <td>${PMSFData.formatBalls(p.actual_front || [], 'front')}</td>
            <td>${PMSFData.formatBalls(p.actual_back || [], 'back')}</td>
            <td><strong style="color:var(--accent-green);">${p.front_hit || 0}/5</strong></td>
            <td><strong style="color:var(--accent-cyan);">${p.back_hit || 0}/2</strong></td>
            <td><span class="prize-tag prize-${prize.level}">${prize.name}</span></td>
          </tr>
        `;
      }).join('');
    }

    // 图表
    PMSFCharts.renderPrizeDist(d.predictions);
    PMSFCharts.renderGroupHit(d.predictions);
  },

  _groupCard(g) {
    const tags = { A: 'tag-a', B: 'tag-b', C: 'tag-c', D: 'tag-d' };
    const struct = g.structure || {};
    return `
      <div class="prediction-group">
        <div class="group-header">
          <div class="group-label">
            <span class="tag ${tags[g.label] || ''}">${g.label}组</span>
            ${g.name}
          </div>
        </div>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:10px;">${g.description || ''}</p>
        <div class="group-balls">
          ${PMSFData.formatBalls(g.front, 'front')}
          <div class="divider"></div>
          ${PMSFData.formatBalls(g.back, 'back')}
        </div>
        <div class="group-meta">
          <span class="meta-item">${struct.odd_even || ''}</span>
          <span class="meta-item">和值 ${struct.sum || ''}</span>
          <span class="meta-item">跨度 ${struct.span || ''}</span>
        </div>
      </div>
    `;
  },

  renderModel() {
    const d = PMSFData;
    const s = d.stats || {};

    // 架构图
    const arch = document.getElementById('architecture');
    if (arch) {
      const layers = [
        { num: 5, name: '组合优化层', modules: 'Monte Carlo · Genetic Algorithm · Risk Control · Structure Filter', color: '#ef4444' },
        { num: 4, name: '概率融合层', modules: 'XGBoost · CatBoost · TFT · GNN · Copula · Bayesian Fusion', color: '#f59e0b' },
        { num: 3, name: '状态识别层', modules: 'HMM · HSMM (主模型) · Markov Switching', color: '#8b5cf6' },
        { num: 2, name: '彭湃规则层', modules: '双线系统 · 号码关系网络 · 三态系统 (纠缠/终止/拓展)', color: '#06b6d4' },
        { num: 1, name: '数据基础层', modules: 'SQLite数据库 · 网络抓取 · 23维特征工程', color: '#10b981' }
      ];
      arch.innerHTML = layers.map((l, i) => `
        ${i > 0 ? '<div class="arch-arrow">▼</div>' : ''}
        <div class="arch-layer">
          <div class="layer-num" style="background:${l.color};">${l.num}</div>
          <div class="layer-info">
            <div class="layer-name">${l.name}</div>
            <div class="layer-modules">${l.modules}</div>
          </div>
        </div>
      `).join('');
    }

    // 融合权重
    if (s.fusion_weights) PMSFCharts.renderFusionWeight(s.fusion_weights);

    // 关系网络
    if (s.top_relations) PMSFCharts.renderRelation(s.top_relations);

    // 模型详解
    const details = document.getElementById('modelDetails');
    if (details) {
      const models = [
        { name: 'XGBoost', type: '梯度提升树', desc: '号码概率排序主模型，输入20维数值特征（遗漏、冷热、尾数、区域、趋势等），输出号码出现概率。优势：结构化数据、非线性建模、特征重要性可解释。' },
        { name: 'CatBoost', type: '梯度提升树', desc: '处理类别变量的梯度提升模型，自动处理区域、尾数、状态等类别特征，无需手动one-hot编码。与XGBoost形成互补。' },
        { name: 'TFT (Temporal Fusion Transformer)', type: '时序Transformer', desc: '时间序列融合Transformer，通过变量选择网络、门控残差网络(GRN)和注意力机制捕捉长期依赖关系，自动调整5/10/30期不同窗口的权重。' },
        { name: 'GNN (图神经网络)', type: '图卷积网络', desc: '在35节点的号码关系网络上学习节点Embedding，通过图卷积聚合邻居信息，学习号码间的共现、邻号、尾数、纠缠关系，输出号码关系评分。' },
        { name: 'Copula', type: '相关性模型', desc: '研究号码之间的非线性依赖关系，通过Gaussian Copula建模联合分布，计算给定已选号码条件下其他号码的条件概率，用于蒙特卡洛采样时的概率修正。' },
        { name: 'HSMM (隐半马尔可夫模型)', type: '状态模型 (PMSF主状态模型)', desc: '在HMM基础上增加状态持续时间建模，学习"纠缠态通常持续5期"等规律。输入冷热、遗漏、配对、区域、尾数、跨度等6维状态指标，输出A/B/C三态概率及持续期数估计。' },
        { name: 'HMM (隐藏马尔可夫模型)', type: '状态模型', desc: '寻找状态变化路径，通过转移矩阵判断下一状态概率。只考虑下一状态，不考虑持续时间，作为HSMM的辅助模型。' },
        { name: 'Markov Switching', type: '状态切换模型', desc: '处理不同状态下的不同号码分布，纠缠态偏向热号延续，拓展态偏向冷号恢复。用于将状态概率条件化到号码概率上。' }
      ];
      details.innerHTML = `<div class="grid grid-2">${models.map(m => `
        <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:8px;padding:16px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <strong style="color:var(--accent-cyan);">${m.name}</strong>
            <span style="font-size:11px;color:var(--text-muted);background:var(--bg-card);padding:2px 8px;border-radius:4px;">${m.type}</span>
          </div>
          <p style="font-size:13px;color:var(--text-secondary);line-height:1.6;">${m.desc}</p>
        </div>
      `).join('')}</div>`;
    }
  },

  renderReport() {
    const el = document.getElementById('reportContent');
    if (!el) return;
    if (PMSFData.report && PMSFData.report.content) {
      el.innerHTML = PMSFData.report.content;
    } else {
      el.innerHTML = '<div class="loading"><div class="spinner"></div>报告数据加载中...</div>';
    }
  },

  renderRuntime() {
    const d = PMSFData;

    // 运行流程
    const flow = document.getElementById('runtimeFlow');
    if (flow) {
      const steps = [
        { name: '数据抓取', desc: '从500彩票网获取最新开奖数据', icon: '📥' },
        { name: '特征工程', desc: '计算23维号码特征（遗漏、冷热、趋势等）', icon: '🔧' },
        { name: '状态识别', desc: 'HSMM/HMM/Markov Switching判定当前状态', icon: '🎯' },
        { name: '概率建模', desc: 'XGBoost/CatBoost/TFT/GNN/Copula多模型预测', icon: '🤖' },
        { name: '概率融合', desc: '加权融合 + 贝叶斯修正 + 彭湃规则偏置', icon: '⚖️' },
        { name: '组合优化', desc: '蒙特卡洛采样 + 遗传算法 + 结构过滤', icon: '🧬' },
        { name: '输出生成', desc: '4组差异化推荐 + 风控验证', icon: '📤' }
      ];
      flow.innerHTML = `<div style="display:flex;gap:8px;overflow-x:auto;padding-bottom:12px;">
        ${steps.map((s, i) => `
          <div style="flex:1;min-width:120px;background:var(--bg-secondary);border:1px solid var(--border);border-radius:8px;padding:14px;text-align:center;position:relative;">
            <div style="font-size:24px;margin-bottom:6px;">${s.icon}</div>
            <div style="font-size:13px;font-weight:600;margin-bottom:4px;">${i + 1}. ${s.name}</div>
            <div style="font-size:11px;color:var(--text-muted);">${s.desc}</div>
          </div>
          ${i < steps.length - 1 ? '<div style="display:flex;align-items:center;color:var(--text-muted);">→</div>' : ''}
        `).join('')}
      </div>`;
    }

    // 复盘分析
    const review = document.getElementById('reviewAnalysis');
    if (review && d.predictions && d.predictions.length > 0) {
      const latest = [...d.predictions].reverse()[0];
      const prize = PMSFData.calcPrize(latest.front_hit || 0, latest.back_hit || 0);
      review.innerHTML = `
        <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">
          <div>
            <div style="font-size:12px;color:var(--text-muted);">复盘期号</div>
            <div style="font-size:24px;font-weight:700;color:var(--accent-cyan);">${latest.issue}</div>
          </div>
          <div>
            <div style="font-size:12px;color:var(--text-muted);">推荐组别</div>
            <div style="font-size:18px;font-weight:600;">${latest.group}组</div>
          </div>
          <div>
            <div style="font-size:12px;color:var(--text-muted);">前区命中</div>
            <div style="font-size:18px;font-weight:600;color:var(--accent-green);">${latest.front_hit || 0}/5</div>
          </div>
          <div>
            <div style="font-size:12px;color:var(--text-muted);">后区命中</div>
            <div style="font-size:18px;font-weight:600;color:var(--accent-blue);">${latest.back_hit || 0}/2</div>
          </div>
          <div>
            <div style="font-size:12px;color:var(--text-muted);">获得奖项</div>
            <div><span class="prize-tag prize-${prize.level}">${prize.name}</span></div>
          </div>
        </div>
        <div style="display:flex;gap:20px;flex-wrap:wrap;">
          <div>
            <div style="font-size:12px;color:var(--text-muted);margin-bottom:6px;">预测号码</div>
            <div>${PMSFData.formatBalls(latest.predicted_front || [], 'front')} <span style="margin:0 6px;">|</span> ${PMSFData.formatBalls(latest.predicted_back || [], 'back')}</div>
          </div>
          <div>
            <div style="font-size:12px;color:var(--text-muted);margin-bottom:6px;">实际开奖</div>
            <div>${PMSFData.formatBalls(latest.actual_front || [], 'front')} <span style="margin:0 6px;">|</span> ${PMSFData.formatBalls(latest.actual_back || [], 'back')}</div>
          </div>
        </div>
      `;
    } else if (review) {
      review.innerHTML = '<div style="color:var(--text-muted);padding:20px;text-align:center;">暂无复盘数据</div>';
    }

    // 运行日志
    const logBody = document.querySelector('#runtimeLog tbody');
    if (logBody && d.runtime && d.runtime.logs) {
      logBody.innerHTML = d.runtime.logs.map(l => `
        <tr>
          <td style="font-size:12px;">${l.time}</td>
          <td>${l.action}</td>
          <td style="font-size:12px;color:var(--text-secondary);">${l.detail}</td>
          <td><span class="prize-tag prize-${l.status === 'success' ? 6 : l.status === 'running' ? 5 : 0}">${l.status}</span></td>
        </tr>
      `).join('');
    }
  },

  showError() {
    document.querySelectorAll('.tab-content').forEach(c => c.innerHTML =
      '<div class="loading"><div class="spinner"></div>数据加载失败，请检查 data/ 目录下的JSON文件</div>'
    );
  }
};

// 启动
document.addEventListener('DOMContentLoaded', () => PMSFApp.init());
