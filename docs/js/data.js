// PMSF-V1 数据加载模块
const PMSFData = {
  history: [],
  predictions: [],
  latest: null,
  stats: null,
  report: null,
  runtime: null,

  async loadAll() {
    const results = await Promise.allSettled([
      this.loadJSON('data/history.json'),
      this.loadJSON('data/predictions.json'),
      this.loadJSON('data/latest_prediction.json'),
      this.loadJSON('data/stats.json'),
      this.loadJSON('data/report.json'),
      this.loadJSON('data/runtime.json')
    ]);

    this.history = results[0].status === 'fulfilled' ? results[0].value : [];
    this.predictions = results[1].status === 'fulfilled' ? results[1].value : [];
    this.latest = results[2].status === 'fulfilled' ? results[2].value : null;
    this.stats = results[3].status === 'fulfilled' ? results[3].value : null;
    this.report = results[4].status === 'fulfilled' ? results[4].value : null;
    this.runtime = results[5].status === 'fulfilled' ? results[5].value : null;

    return this;
  },

  async loadJSON(path) {
    const resp = await fetch(path + '?t=' + Date.now());
    if (!resp.ok) throw new Error('Failed to load ' + path);
    return resp.json();
  },

  // 计算奖项等级
  calcPrize(frontHit, backHit) {
    if (frontHit === 5 && backHit === 2) return { level: 1, name: '一等奖' };
    if (frontHit === 5 && backHit === 1) return { level: 2, name: '二等奖' };
    if (frontHit === 5 && backHit === 0) return { level: 3, name: '三等奖' };
    if (frontHit === 4 && backHit === 2) return { level: 4, name: '四等奖' };
    if (frontHit === 4 && backHit === 1) return { level: 5, name: '五等奖' };
    if (frontHit === 3 && backHit === 2) return { level: 6, name: '六等奖' };
    if (frontHit === 4 && backHit === 0) return { level: 7, name: '七等奖' };
    if (frontHit === 3 && backHit === 1) return { level: 8, name: '八等奖' };
    if (frontHit === 2 && backHit === 2) return { level: 9, name: '九等奖' };
    return { level: 0, name: '未中奖' };
  },

  // 获取状态名称
  getStateName(state) {
    const names = { A: '纠缠热态', B: '终止冷态', C: '拓展回补态' };
    return names[state] || state;
  },

  // 格式化号码
  formatBall(num) {
    return String(num).padStart(2, '0');
  },

  // 格式化号码列表
  formatBalls(nums, type = 'front') {
    return nums.map(n => `<span class="ball ball-${type}">${this.formatBall(n)}</span>`).join('');
  }
};
