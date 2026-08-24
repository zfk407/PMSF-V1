"""将Markdown报告转换为HTML并保存为report.json"""
import os
import re
import json

REPORT_PATH = r"E:\PMSF-V1\results\PMSF-V1详细报告_20260824_130158.md"
OUTPUT_PATH = r"E:\PMSF-V1\web\data\report.json"

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

        # 代码块
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

        # 表格
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            # 跳过分隔行
            if all(re.match(r'^[-:]+$', c) for c in cells):
                i += 1
                continue
            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            # 结束表格
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

        # 标题
        if line.startswith('# '):
            html.append(f'<h1>{line[2:].strip()}</h1>')
        elif line.startswith('## '):
            html.append(f'<h2>{line[3:].strip()}</h2>')
        elif line.startswith('### '):
            html.append(f'<h3>{line[4:].strip()}</h3>')
        elif line.startswith('#### '):
            html.append(f'<h4>{line[5:].strip()}</h4>')
        # 引用
        elif line.startswith('> '):
            html.append(f'<blockquote>{line[2:].strip()}</blockquote>')
        # 水平线
        elif line.strip() == '---':
            html.append('<hr>')
        # 列表
        elif re.match(r'^\d+\.\s', line):
            html.append(f'<p>{line.strip()}</p>')
        elif line.startswith('- '):
            html.append(f'<p>• {line[2:].strip()}</p>')
        # 空行
        elif line.strip() == '':
            pass
        # 普通段落
        else:
            # 处理粗体
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            html.append(f'<p>{text}</p>')

        i += 1

    # 处理末尾表格
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


def main():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        md_text = f.read()

    html_content = md_to_html(md_text)

    report_data = {
        "title": "PMSF-V1 彭湃大乐透多尺度状态融合系统 — 详细报告",
        "generated_at": "2026-08-24 13:01:58",
        "version": "PMSF-V1.0",
        "content": html_content
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"报告已转换并保存: {OUTPUT_PATH}")
    print(f"HTML内容长度: {len(html_content)} 字符")


if __name__ == '__main__':
    main()
