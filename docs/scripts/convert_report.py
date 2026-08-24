"""将Markdown报告转换为HTML并保存为report.json
动态部分（最新运行结果、推荐号码）使用占位符，由前端实时注入
"""
import os
import re
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(DOCS_DIR)

# 输入：项目根目录 results/ 下最新的 md 报告
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_PATH = os.path.join(DOCS_DIR, "data", "report.json")


def find_latest_report():
    """查找最新的报告文件"""
    if not os.path.exists(RESULTS_DIR):
        return None
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.md') and '报告' in f]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(RESULTS_DIR, files[0])


def md_to_html(md_text):
    """简易Markdown转HTML"""
    lines = md_text.split('\n')
    html = []
    in_table = False
    table_rows = []
    in_code = False
    code_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip().startswith('```'):
            if in_code:
                html.append('<pre><code>' + '\n'.join(code_lines) + '</code></pre>')
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if all(re.match(r'^[-:]+$', c) for c in cells):
                i += 1
                continue
            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            if table_rows:
                header = table_rows[0]
                body = table_rows[1:]
                html.append('<table>')
                html.append('<thead><tr>' + ''.join(f'<th>{c}</th>' for c in header) + '</tr></thead>')
                html.append('<tbody>')
                for row in body:
                    html.append('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>')
                html.append('</tbody></table>')
            in_table = False
            table_rows = []

        if line.startswith('# '):
            html.append(f'<h1>{line[2:].strip()}</h1>')
        elif line.startswith('## '):
            html.append(f'<h2>{line[3:].strip()}</h2>')
        elif line.startswith('### '):
            html.append(f'<h3>{line[4:].strip()}</h3>')
        elif line.startswith('#### '):
            html.append(f'<h4>{line[5:].strip()}</h4>')
        elif line.startswith('> '):
            html.append(f'<blockquote>{line[2:].strip()}</blockquote>')
        elif line.strip() == '---':
            html.append('<hr>')
        elif re.match(r'^\d+\.\s', line):
            html.append(f'<p>{line.strip()}</p>')
        elif line.startswith('- '):
            html.append(f'<p>• {line[2:].strip()}</p>')
        elif line.strip() == '':
            pass
        else:
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            html.append(f'<p>{text}</p>')

        i += 1

    if in_table and table_rows:
        header = table_rows[0]
        body = table_rows[1:]
        html.append('<table>')
        html.append('<thead><tr>' + ''.join(f'<th>{c}</th>' for c in header) + '</tr></thead>')
        html.append('<tbody>')
        for row in body:
            html.append('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>')
        html.append('</tbody></table>')

    return '\n'.join(html)


def inject_placeholder(html_content):
    """将'最新运行结果'部分替换为动态占位符"""
    # 找到 6.3 最新运行结果 的 h3 标签
    pattern = r'<h3>6\.3\s*最新运行结果</h3>.*?(?=<h2>|$)'
    replacement = '<h3>6.3 最新运行结果</h3>\n<div id="report-latest-prediction" class="report-dynamic">正在加载最新预测数据...</div>\n'
    return re.sub(pattern, replacement, html_content, flags=re.DOTALL)


def main():
    report_path = find_latest_report()
    if not report_path:
        print("未找到报告文件")
        return

    print(f"读取报告: {report_path}")
    with open(report_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    html_content = md_to_html(md_text)
    html_content = inject_placeholder(html_content)

    report_data = {
        "title": "ZHIHUI-DLT 大乐透多尺度状态融合系统 — 详细报告",
        "generated_at": os.path.basename(report_path).replace('PMSF-V1详细报告_', '').replace('.md', ''),
        "version": "ZHIHUI-DLT v1.0",
        "content": html_content
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"报告已转换并保存: {OUTPUT_PATH}")
    print(f"HTML内容长度: {len(html_content)} 字符")
    print(f"包含动态占位符: {'report-latest-prediction' in html_content}")


if __name__ == '__main__':
    main()
