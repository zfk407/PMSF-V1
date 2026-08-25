// PMSF-V1 图表模块
const PMSFCharts = {
  instances: {},

  init(id) {
    if (this.instances[id]) {
      this.instances[id].dispose();
    }
    const el = document.getElementById(id);
    if (!el) return null;
    const chart = echarts.init(el, 'dark');
    this.instances[id] = chart;
    return chart;
  },

  resizeAll() {
    Object.values(this.instances).forEach(c => c && c.resize());
  },

  baseOption() {
    return {
      backgroundColor: 'transparent',
      textStyle: { color: '#94a3b8', fontFamily: 'inherit' },
      grid: { left: '3%', right: '4%', bottom: '8%', top: '15%', containLabel: true },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(17,24,39,0.95)',
        borderColor: '#2a3654',
        textStyle: { color: '#e8ecf4' }
      }
    };
  },

  // 前区号码热度柱状图
  renderHotNumbers(data) {
    const chart = this.init('chartHotNumbers');
    if (!chart || !data) return;
    const nums = Object.keys(data).map(Number).sort((a, b) => a - b);
    const values = nums.map(n => +(data[n] * 100).toFixed(2));
    const option = {
      ...this.baseOption(),
      xAxis: { type: 'category', data: nums.map(n => String(n).padStart(2, '0')), axisLabel: { color: '#94a3b8', fontSize: 10 } },
      yAxis: { type: 'value', name: '出现频率(%)', axisLabel: { color: '#94a3b8' } },
      series: [{
        type: 'bar',
        data: values,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#3b82f6' }, { offset: 1, color: '#1e3a5f' }
          ]),
          borderRadius: [4, 4, 0, 0]
        },
        markLine: { data: [{ type: 'average', name: '平均' }], lineStyle: { color: '#f59e0b' } }
      }]
    };
    chart.setOption(option);
  },

  // 和值走势
  renderSumTrend(data) {
    const chart = this.init('chartSumTrend');
    if (!chart || !data || !data.length) return;
    const recent = data.slice(-100);
    const option = {
      ...this.baseOption(),
      xAxis: { type: 'category', data: recent.map(d => d.issue), axisLabel: { color: '#94a3b8', fontSize: 9, interval: 9 } },
      yAxis: { type: 'value', name: '和值', axisLabel: { color: '#94a3b8' } },
      series: [{
        type: 'line',
        data: recent.map(d => d.sum_front),
        smooth: true,
        symbol: 'none',
        lineStyle: { color: '#06b6d4', width: 2 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(6,182,212,0.3)' }, { offset: 1, color: 'rgba(6,182,212,0)' }
          ])
        },
        markLine: { data: [{ type: 'average', name: '均值' }], lineStyle: { color: '#f59e0b' } }
      }]
    };
    chart.setOption(option);
  },

  // 奇偶结构饼图
  renderOddEven(data) {
    const chart = this.init('chartOddEven');
    if (!chart || !data) return;
    const option = {
      ...this.baseOption(),
      tooltip: { trigger: 'item', formatter: '{b}: {c}% ({d}%)' },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        data: Object.entries(data).map(([k, v]) => ({ name: k, value: +(v * 100).toFixed(1) })),
        itemStyle: { borderRadius: 6, borderColor: '#1a2236', borderWidth: 2 },
        label: { color: '#e8ecf4' }
      }]
    };
    chart.setOption(option);
  },

  // 状态概率
  renderStateProb(data) {
    const chart = this.init('chartStateProb');
    if (!chart || !data) return;
    const states = ['A', 'B', 'C'];
    const names = ['纠缠热态', '终止冷态', '拓展回补态'];
    const colors = ['#ef4444', '#3b82f6', '#10b981'];
    const option = {
      ...this.baseOption(),
      tooltip: { trigger: 'item', formatter: '{b}: {c}%' },
      series: [{
        type: 'pie',
        radius: ['50%', '75%'],
        data: states.map((s, i) => ({ name: names[i], value: +((data[s] || 0) * 100).toFixed(1), itemStyle: { color: colors[i] } })),
        itemStyle: { borderRadius: 8, borderColor: '#1a2236', borderWidth: 3 },
        label: { color: '#e8ecf4', formatter: '{b}\n{c}%' }
      }]
    };
    chart.setOption(option);
  },

  // 前区频率
  renderFrontFreq(data) {
    const chart = this.init('chartFrontFreq');
    if (!chart || !data) return;
    const nums = Object.keys(data).map(Number).sort((a, b) => a - b);
    const option = {
      ...this.baseOption(),
      title: { text: '前区号码频率(01-35)', left: 'center', textStyle: { color: '#e8ecf4', fontSize: 14 } },
      xAxis: { type: 'category', data: nums.map(n => String(n).padStart(2, '0')), axisLabel: { color: '#94a3b8', fontSize: 10 } },
      yAxis: { type: 'value', axisLabel: { color: '#94a3b8' } },
      series: [{
        type: 'bar',
        data: nums.map(n => +(data[n] * 100).toFixed(2)),
        itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#ef4444' }, { offset: 1, color: '#7f1d1d' }]), borderRadius: [3, 3, 0, 0] }
      }]
    };
    chart.setOption(option);
  },

  // 后区频率
  renderBackFreq(data) {
    const chart = this.init('chartBackFreq');
    if (!chart || !data) return;
    const nums = Object.keys(data).map(Number).sort((a, b) => a - b);
    const option = {
      ...this.baseOption(),
      title: { text: '后区号码频率(01-12)', left: 'center', textStyle: { color: '#e8ecf4', fontSize: 14 } },
      xAxis: { type: 'category', data: nums.map(n => String(n).padStart(2, '0')), axisLabel: { color: '#94a3b8' } },
      yAxis: { type: 'value', axisLabel: { color: '#94a3b8' } },
      series: [{
        type: 'bar',
        data: nums.map(n => +(data[n] * 100).toFixed(2)),
        itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#3b82f6' }, { offset: 1, color: '#1e3a5f' }]), borderRadius: [3, 3, 0, 0] }
      }]
    };
    chart.setOption(option);
  },

  // 和值分布
  renderSumDist(data) {
    const chart = this.init('chartSumDist');
    if (!chart || !data) return;
    const option = {
      ...this.baseOption(),
      xAxis: { type: 'category', data: Object.keys(data), axisLabel: { color: '#94a3b8', fontSize: 11 } },
      yAxis: { type: 'value', name: '频率(%)', axisLabel: { color: '#94a3b8' } },
      series: [{
        type: 'bar',
        data: Object.values(data).map(v => +(v * 100).toFixed(1)),
        itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#8b5cf6' }, { offset: 1, color: '#4c1d95' }]), borderRadius: [4, 4, 0, 0] }
      }]
    };
    chart.setOption(option);
  },

  // 跨度分布
  renderSpanDist(data) {
    const chart = this.init('chartSpanDist');
    if (!chart || !data) return;
    const counts = {};
    data.forEach(d => {
      const span = d.span_front;
      counts[span] = (counts[span] || 0) + 1;
    });
    const keys = Object.keys(counts).map(Number).sort((a, b) => a - b);
    const total = data.length;
    const option = {
      ...this.baseOption(),
      xAxis: { type: 'category', data: keys, axisLabel: { color: '#94a3b8' } },
      yAxis: { type: 'value', name: '频率(%)', axisLabel: { color: '#94a3b8' } },
      series: [{
        type: 'bar',
        data: keys.map(k => +((counts[k] / total) * 100).toFixed(1)),
        itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#06b6d4' }, { offset: 1, color: '#164e63' }]), borderRadius: [3, 3, 0, 0] }
      }]
    };
    chart.setOption(option);
  },

  // 大小结构
  renderBigSmall(data) {
    const chart = this.init('chartBigSmall');
    if (!chart || !data) return;
    const option = {
      ...this.baseOption(),
      tooltip: { trigger: 'item', formatter: '{b}: {c}%' },
      series: [{
        type: 'pie',
        radius: ['35%', '65%'],
        roseType: 'radius',
        data: Object.entries(data).map(([k, v]) => ({ name: k, value: +(v * 100).toFixed(1) })),
        itemStyle: { borderRadius: 6, borderColor: '#1a2236', borderWidth: 2 },
        label: { color: '#e8ecf4' }
      }]
    };
    chart.setOption(option);
  },

  // 四区分布
  renderZone(data) {
    const chart = this.init('chartZone');
    if (!chart || !data) return;
    const option = {
      ...this.baseOption(),
      tooltip: { trigger: 'item', formatter: '{b}: {c}%' },
      series: [{
        type: 'pie',
        radius: ['35%', '65%'],
        data: Object.entries(data).slice(0, 10).map(([k, v]) => ({ name: k, value: +(v * 100).toFixed(1) })),
        itemStyle: { borderRadius: 6, borderColor: '#1a2236', borderWidth: 2 },
        label: { color: '#e8ecf4', fontSize: 11 }
      }]
    };
    chart.setOption(option);
  },

  // 融合权重
  renderFusionWeight(data) {
    const chart = this.init('chartFusionWeight');
    if (!chart || !data) return;
    const option = {
      ...this.baseOption(),
      tooltip: { trigger: 'item', formatter: '{b}: {c}%' },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        data: Object.entries(data).map(([k, v]) => ({ name: k, value: +(v * 100).toFixed(1) })),
        itemStyle: { borderRadius: 6, borderColor: '#1a2236', borderWidth: 2 },
        label: { color: '#e8ecf4', formatter: '{b}\n{c}%' }
      }]
    };
    chart.setOption(option);
  },

  // 关系网络（共现Top20，用横向柱状图）
  renderRelation(data) {
    const chart = this.init('chartRelation');
    if (!chart || !data || !data.length) return;
    const top = data.slice(0, 20);
    const option = {
      ...this.baseOption(),
      grid: { left: '3%', right: '8%', bottom: '3%', top: '5%', containLabel: true },
      xAxis: { type: 'value', axisLabel: { color: '#94a3b8' } },
      yAxis: { type: 'category', data: top.map(d => `${d[0]}-${d[1]}`).reverse(), axisLabel: { color: '#94a3b8', fontSize: 10 } },
      series: [{
        type: 'bar',
        data: top.map(d => +(d[2] * d[3] * 100).toFixed(2)).reverse(),
        itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: '#8b5cf6' }, { offset: 1, color: '#ec4899' }]), borderRadius: [0, 4, 4, 0] }
      }]
    };
    chart.setOption(option);
  },

  // 中奖等级分布
  renderPrizeDist(data) {
    const chart = this.init('chartPrizeDist');
    if (!chart) return;
    const counts = {};
    (data || []).forEach(p => {
      const level = p.prize_level || 0;
      counts[level] = (counts[level] || 0) + 1;
    });
    const names = { 0: '未中奖', 1: '一等奖', 2: '二等奖', 3: '三等奖', 4: '四等奖', 5: '五等奖', 6: '六等奖', 7: '七等奖', 8: '八等奖', 9: '九等奖' };
    const keys = Object.keys(counts).map(Number).sort((a, b) => b - a);
    if (keys.length === 0) {
      chart.setOption({ ...this.baseOption(), title: { text: '暂无中奖记录', left: 'center', top: 'center', textStyle: { color: '#64748b', fontSize: 14 } } });
      return;
    }
    const option = {
      ...this.baseOption(),
      tooltip: { trigger: 'item', formatter: '{b}: {c}次' },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        data: keys.map(k => ({ name: names[k] || k, value: counts[k] })),
        itemStyle: { borderRadius: 6, borderColor: '#1a2236', borderWidth: 2 },
        label: { color: '#e8ecf4' }
      }]
    };
    chart.setOption(option);
  },

  // 各组命中率
  renderGroupHit(data) {
    const chart = this.init('chartGroupHit');
    if (!chart) return;
    const groups = { A: [], B: [], C: [], D: [] };
    (data || []).forEach(p => {
      if (groups[p.group]) {
        groups[p.group].push(p.front_hit || 0);
      }
    });
    const avgHit = {};
    Object.entries(groups).forEach(([g, arr]) => {
      avgHit[g] = arr.length ? +(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2) : 0;
    });
    const hasData = Object.values(avgHit).some(v => v > 0);
    if (!hasData) {
      chart.setOption({ ...this.baseOption(), title: { text: '暂无预测记录', left: 'center', top: 'center', textStyle: { color: '#64748b', fontSize: 14 } } });
      return;
    }
    const option = {
      ...this.baseOption(),
      xAxis: { type: 'category', data: ['A组', 'B组', 'C组', 'D组'], axisLabel: { color: '#94a3b8' } },
      yAxis: { type: 'value', name: '平均命中数', max: 5, axisLabel: { color: '#94a3b8' } },
      series: [{
        type: 'bar',
        data: ['A', 'B', 'C', 'D'].map((g, i) => ({
          value: avgHit[g],
          itemStyle: { color: ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'][i], borderRadius: [6, 6, 0, 0] }
        })),
        label: { show: true, position: 'top', color: '#e8ecf4' }
      }]
    };
    chart.setOption(option);
  // ========== 双色球图表 ==========

  // 九转连环图（彭湃双色球核心可视化）
  renderJiuzhuan(ssq) {
    const chart = this.init('chartJiuzhuan');
    if (!chart || !ssq) return;

    const rl = ssq.rules || {};
    const pairs = rl.pair_groups || [];
    const mainTrans = rl.main_transition;
    const hotGroups = new Set(rl.hot_groups || []);
    const state = (ssq.current_state || {}).state || 'B';
    const stateColor = { A: '#f59e0b', B: '#ef4444', C: '#10b981' }[state] || '#06b6d4';

    // 节点：33个红球（16组 x 2 + 1独立号）+ 中心过渡号
    const nodes = [];
    const edges = [];

    // 外环：16组配对号
    pairs.forEach((pair, gi) => {
      const isHot = hotGroups.has(gi);
      const color = isHot ? '#ef4444' : '#3b82f6';
      const size = isHot ? 34 : 24;
      [0, 1].forEach((k) => {
        nodes.push({
          id: 'n' + pair[k],
          name: pair[k],
          symbolSize: size,
          x: null, y: null,
          value: gi,
          itemStyle: { color, borderColor: '#0f172a', borderWidth: 2 },
          label: { show: true, fontSize: 11, color: '#e8ecf4' }
        });
      });
      // 组内配对连线（恒值34）
      edges.push({
        source: 'n' + pair[0],
        target: 'n' + pair[1],
        lineStyle: { color: isHot ? 'rgba(239,68,68,0.7)' : 'rgba(59,130,246,0.4)', width: isHot ? 2 : 1 }
      });
    });

    // 独立过渡号33
    nodes.push({
      id: 'n33', name: '33', symbolSize: 26,
      value: 16,
      itemStyle: { color: '#8b5cf6', borderColor: '#0f172a', borderWidth: 2 },
      label: { show: true, fontSize: 11, color: '#e8ecf4' }
    });

    // 中心主过渡号
    nodes.push({
      id: 'center', name: String(mainTrans).padStart(2, '0'),
      symbolSize: 46, x: 0, y: 0,
      itemStyle: { color: stateColor, borderColor: '#fff', borderWidth: 2, shadowBlur: 20, shadowColor: stateColor },
      label: { show: true, fontSize: 14, fontWeight: 'bold', color: '#fff' }
    });

    // 中心到各节点连线：主过渡号与其配对/邻号的关系
    if (mainTrans != null) {
      const transStr = String(mainTrans).padStart(2, '0');
      nodes.forEach(n => {
        if (n.id === 'center') return;
        const num = parseInt(n.id.replace('n', ''));
        // 主过渡号与配对号强关联
        if (num === (34 - mainTrans)) {
          edges.push({ source: 'center', target: n.id, lineStyle: { color: stateColor, width: 3, curveness: 0.2, type: 'dashed' } });
        }
        // 主过渡号与邻号（左右甩）关联
        if (Math.abs(num - mainTrans) === 1) {
          edges.push({ source: 'center', target: n.id, lineStyle: { color: 'rgba(139,92,246,0.5)', width: 1.5, curveness: 0.2 } });
        }
      });
    }

    // 环形布局：手动计算外环坐标
    const outerNodes = nodes.filter(n => n.id !== 'center');
    const R = 150;
    outerNodes.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / outerNodes.length - Math.PI / 2;
      n.x = Math.cos(angle) * R;
      n.y = Math.sin(angle) * R;
    });

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        backgroundColor: 'rgba(17,24,39,0.95)',
        borderColor: '#2a3654',
        textStyle: { color: '#e8ecf4' },
        formatter: (p) => {
          if (p.dataType === 'node') {
            const num = parseInt((p.data.id || '').replace('n', ''));
            const isMain = p.data.id === 'center';
            if (isMain) return `主过渡号 ${p.data.name}（${rl.line === 'single' ? '单期线固定' : '双期线固定'}）`;
            return `红球 ${String(num).padStart(2, '0')} · 组${p.data.value}`;
          }
          return '';
        }
      },
      series: [{
        type: 'graph',
        layout: 'none',
        data: nodes,
        links: edges,
        roam: true,
        draggable: true,
        scaleLimit: { min: 0.6, max: 2 },
        label: { position: 'inside', show: true },
        lineStyle: { opacity: 0.6 }
      }],
      title: {
        text: `${rl.line === 'single' ? '单期线' : '双期线'} · 主过渡号 ${String(mainTrans).padStart(2, '0')}`,
        left: 'center',
        top: 0,
        textStyle: { color: '#e8ecf4', fontSize: 13 }
      }
    };
    chart.setOption(option);
  },

  // 双色球红球概率Top10
  renderSsqRedTop(data) {
    const chart = this.init('chartSsqRedTop');
    if (!chart || !data || !data.length) {
      if (chart) chart.setOption({ ...this.baseOption(), title: { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#64748b', fontSize: 13 } } });
      return;
    }
    const sorted = [...data].sort((a, b) => b.probability - a.probability).slice(0, 10);
    const option = {
      ...this.baseOption(),
      grid: { left: '5%', right: '10%', bottom: '5%', top: '10%', containLabel: true },
      xAxis: { type: 'value', name: '概率', axisLabel: { color: '#94a3b8' } },
      yAxis: {
        type: 'category',
        data: sorted.map(d => String(d.number).padStart(2, '0')).reverse(),
        axisLabel: { color: '#94a3b8', fontSize: 12 }
      },
      series: [{
        type: 'bar',
        data: sorted.map(d => d.probability).reverse(),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#06b6d4' }, { offset: 1, color: '#3b82f6' }
          ]),
          borderRadius: [0, 6, 6, 0]
        },
        label: { show: true, position: 'right', color: '#e8ecf4', fontSize: 11, formatter: p => p.value.toFixed(4) }
      }]
    };
    chart.setOption(option);
  },

  // 双色球蓝球概率Top5
  renderSsqBlueTop(data) {
    const chart = this.init('chartSsqBlueTop');
    if (!chart || !data || !data.length) {
      if (chart) chart.setOption({ ...this.baseOption(), title: { text: '暂无数据', left: 'center', top: 'center', textStyle: { color: '#64748b', fontSize: 13 } } });
      return;
    }
    const sorted = [...data].sort((a, b) => b.probability - a.probability).slice(0, 5);
    const option = {
      ...this.baseOption(),
      grid: { left: '5%', right: '10%', bottom: '5%', top: '10%', containLabel: true },
      xAxis: { type: 'value', name: '概率', axisLabel: { color: '#94a3b8' } },
      yAxis: {
        type: 'category',
        data: sorted.map(d => String(d.number).padStart(2, '0')).reverse(),
        axisLabel: { color: '#94a3b8', fontSize: 12 }
      },
      series: [{
        type: 'bar',
        data: sorted.map(d => d.probability).reverse(),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#8b5cf6' }, { offset: 1, color: '#a855f7' }
          ]),
          borderRadius: [0, 6, 6, 0]
        },
        label: { show: true, position: 'right', color: '#e8ecf4', fontSize: 11, formatter: p => p.value.toFixed(4) }
      }]
    };
    chart.setOption(option);
  }
};
