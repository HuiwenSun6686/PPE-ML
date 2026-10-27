from pathlib import Path
import json, os, subprocess, pandas as pd, shutil

def excel(path, sheets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from openpyxl.styles import Font, PatternFill, Alignment
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, rows in sheets.items():
            q = pd.DataFrame(rows)
            q.to_excel(writer, sheet_name=name[:31], index=False)
            s = writer.sheets[name[:31]]
            s.freeze_panes = 'B2'
            s.auto_filter.ref = s.dimensions
            s.sheet_view.showGridLines = False
            for c in s[1]:
                c.font = Font(name='Microsoft YaHei', bold=True, color='FFFFFF')
                c.fill = PatternFill('solid', fgColor='244969')
            for col in s.columns:
                width = min(45, max(14, len(str(col[0].value)) * 1.2 + 2))
                s.column_dimensions[col[0].column_letter].width = width
            s.row_dimensions[1].height = 42
            for c in s[1]:
                c.alignment = Alignment(wrap_text=True, vertical='center')
            for col in s.columns:
                if str(col[0].value) in ['result_interpretation', 'figure_interpretation', 'description']:
                    s.column_dimensions[col[0].column_letter].width = 90
                    for c in col[1:]:
                        c.alignment = Alignment(wrap_text=True, vertical='center')
                        s.row_dimensions[c.row].height = 108

def json_out(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')

def csv(p, x):
    p = Path(p)
    temporary = p.with_name(p.name + '.temporary')
    frame = pd.DataFrame(x)
    with temporary.open('w', encoding='utf-8-sig', newline='') as fh:
        frame.to_csv(fh, index=False)
        fh.flush()
        os.fsync(fh.fileno())
    temporary.replace(p)
    if len(frame.columns):
        assert len(pd.read_csv(p)) == len(frame), 'Exported CSV row count differs from the source frame'

def copy_inputs(pairs, dst):
    Path(dst).mkdir(parents=True, exist_ok=True)
    for src, name in pairs:
        dest = Path(dst) / name
        if Path(src).resolve() != dest.resolve():
            shutil.copy2(src, dest)
