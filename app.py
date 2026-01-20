import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
import xml.etree.ElementTree as ET

st.set_page_config(page_title="Hockey Advanced Analytics", layout="wide")

# Print-friendly layout for A4 portrait PDF
st.markdown(
    """
<style>
@page {
  size: A4 portrait;
  margin: 12mm;
}
@media print {
  html, body {
    width: 210mm;
    height: auto;
  }
  .main .block-container {
    max-width: 190mm;
    padding-left: 6mm;
    padding-right: 6mm;
  }
  div[data-testid="stColumns"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 6mm !important;
  }
  div[data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
  }
  h1, h2, h3, h4 {
    break-after: avoid;
    page-break-after: avoid;
  }
  .element-container, .stPlotlyChart, .stDataFrame, .stMarkdown, .stTable {
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .stPlotlyChart, .stDataFrame, .stTable {
    margin-bottom: 6mm;
  }
  .js-plotly-plot, .plotly, .plotly-graph-div {
    max-width: 100% !important;
  }
  .js-plotly-plot .svg-container {
    max-width: 100% !important;
  }
}
</style>
    """,
    unsafe_allow_html=True
)
# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리 함수
# -----------------------------------------------------------------------------

def extract_team(row_str):
    """Row 문자열에서 팀명을 추출"""
    if not isinstance(row_str, str):
        return "Unknown"
    # 예: "인도 A25 START" -> "인도"
    # 공백으로 분리 후 첫 단어가 팀명이라고 가정 (데이터셋 패턴 기반)
    # 필요 시 예외 처리 추가 가능
    first = row_str.split(' ')[0]
    # 분석 대상에서 제외할 사용자 정의 태그
    ignore_tags = {"한국빌드업", "한국프레스", "코치님"}
    if first in ignore_tags:
        return "Unknown"
    return first

TEAM_COLOR_MAP = {
    '일본': '#d62728',
    'Japan': '#d62728',
    '인도': '#1f77b4',
    'India': '#1f77b4',
}


def get_team_color(team: str, default_primary="#1f77b4", default_alt="#d62728"):
    """팀 이름 기반 색상 매핑 (일본=빨강, 인도=파랑). 매핑 없으면 기본 팔레트 순환."""
    if team in TEAM_COLOR_MAP:
        return TEAM_COLOR_MAP[team]
    return default_primary if default_primary else default_alt


def alpha_color(hex_color: str, alpha: float) -> str:
    """#rrggbb -> rgba(r,g,b,alpha)"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def order_teams(teams):
    """일본/인도 순서를 고정하고 나머지는 입력 순서를 유지"""
    preferred = ["일본", "Japan", "인도", "India"]
    unique = []
    seen = set()
    # 선호 순서 먼저
    for p in preferred:
        if p in teams and p not in seen:
            unique.append(p); seen.add(p)
    # 나머지 입력 순서대로
    for t in teams:
        if t not in seen:
            unique.append(t); seen.add(t)
    return unique

def check_location(loc_str, targets):
    """
    loc_str에 targets 리스트(예: ['75', '100'])에 포함된 숫자가 있는지 확인
    """
    if not isinstance(loc_str, str):
        return False
    for t in targets:
        if t in loc_str:
            return True
    return False


def count_events(dframe: pd.DataFrame, row_keyword: str, loc_targets):
    """Row keyword와 지역 조건을 동시에 만족하는 이벤트 수"""
    if dframe.empty:
        return 0
    cond1 = dframe['Row'].str.contains(row_keyword, na=False)
    cond2 = dframe['지역'].apply(lambda x: check_location(x, loc_targets))
    return int(dframe[cond1 & cond2].shape[0])


def compute_press_metrics(rows_us: pd.DataFrame, rows_opp: pd.DataFrame, team_time: float, att_time: float):
    """Press Attempts / Success / SPP / Allowed SPP 계산"""
    us_to_75_100 = count_events(rows_us, '턴오버', ['75', '100'])
    us_foul_75_100 = count_events(rows_us, '파울', ['75', '100'])
    opp_foul_25_50 = count_events(rows_opp, '파울', ['25', '50'])

    press_attempts = us_to_75_100 + us_foul_75_100 + opp_foul_25_50
    press_success = us_to_75_100 + us_foul_75_100

    opp_to_75_100 = count_events(rows_opp, '턴오버', ['75', '100'])
    opp_foul_75_100 = count_events(rows_opp, '파울', ['75', '100'])
    us_foul_25_50 = count_events(rows_us, '파울', ['25', '50'])

    allowed_denom = opp_to_75_100 + opp_foul_75_100 + us_foul_25_50
    build_up_time = team_time - att_time

    spp = build_up_time / press_attempts if press_attempts > 0 else 0
    allowed_spp = build_up_time / allowed_denom if allowed_denom > 0 else 0

    return {
        'press_attempts': press_attempts,
        'press_success': press_success,
        'spp': spp,
        'allowed_spp': allowed_spp
    }

def _common_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Row', '지역', '결과', 'Ungrouped']:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            df[col] = ''
    if 'Duration' not in df.columns:
        df['Duration'] = 0
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce').fillna(0)
    df['Team'] = df['Row'].apply(extract_team)
    return df


@st.cache_data
def process_data(file):
    file.seek(0)
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding='cp949')
    except pd.errors.EmptyDataError:
        file.seek(0)
        df = pd.read_csv(file, encoding='cp949')

    return _common_preprocessing(df)


@st.cache_data
def process_xml_data(file):
    file.seek(0)
    try:
        tree = ET.parse(file)
        root = tree.getroot()
    except Exception as e:
        st.error(f"XML 파일을 읽는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

    parsed_rows = []
    for inst in root.findall('.//instance'):
        try:
            start_elem = inst.find('start')
            end_elem = inst.find('end')
            code_elem = inst.find('code')
            if start_elem is None or end_elem is None:
                continue
            start = float(start_elem.text or 0.0)
            end = float(end_elem.text or 0.0)
            code = (code_elem.text or '').strip() if code_elem is not None else ''

            regions = []
            results = []
            ungrouped = []

            for label in inst.findall('label'):
                group_elem = label.find('group')
                text_elem = label.find('text')
                if text_elem is None or not text_elem.text:
                    continue
                text = str(text_elem.text).strip()
                group_name = str(group_elem.text).strip() if group_elem is not None and group_elem.text else ""

                if "지역" in group_name or any(x in group_name for x in ['Location', 'Zone']):
                    regions.append(text)
                elif "결과" in group_name or any(x in group_name for x in ['Result', 'Outcome']):
                    results.append(text)
                elif any(x in text for x in ['좌_', '우_', '중_', 'LEFT', 'RIGHT']):
                    regions.append(text)
                elif any(x in text.lower() for x in ['entry', 'goal', 'shoot', '슈팅', '득점', 'save']):
                    results.append(text)
                else:
                    ungrouped.append(text)

            parsed_rows.append({
                'Start': start,
                'End': end,
                'Row': code,
                'Duration': end - start,
                '지역': ", ".join(regions),
                '결과': ", ".join(results),
                'Ungrouped': ", ".join(ungrouped)
            })
        except Exception:
            continue

    df = pd.DataFrame(parsed_rows)
    return _common_preprocessing(df)

# -----------------------------------------------------------------------------
# 1-1. 쿼터 파생/시각화에 필요한 보조 함수 
# -----------------------------------------------------------------------------

QUARTER_COLUMN_CANDIDATES = [
    'Quarter', 'quarter', 'QUARTER',
    'Period', 'PERIOD', 'period',
    '쿼터', '쿼터구분', 'Ungrouped'
]
QUARTER_TOKEN_REGEX = re.compile(
    r'(?:q\s*([1-4])|([1-4])\s*q|quarter\s*([1-4])|([1-4])\s*quarter|period\s*([1-4])|([1-4])\s*period|쿼터\s*([1-4])|([1-4])\s*쿼터|([1-4])\s*분기?)',
    re.IGNORECASE
)

QUARTER_METRIC_PATTERNS = {
    'Circle Entries': [
        r'서클\s*진입', r'슈팅\s*서클', r'circle\s*entry', r'attack\s*circle', r'att\s*circle'
    ],
    'Shots': [
        r'^\s*\S+\s*슈팅(?:\s+\d+)?\s*$',
        r'^\s*shot(?:\s+\d+)?\s*$'
    ],
    'PCs': [
        r'^\s*\S+\s*페널티\s*코너\s*$',
        r'^\s*\S+\s*페널티코너\s*$',
        r'\bpenalty\s*corner\b'
    ],
    'Goals': [
        r'^\s*\S+\s*득점\s*$',
        r'\bgoal\b'
    ]
}



def _extract_quarter_label(text: str):
    """Row/Ungrouped 문자열에서 쿼터 정보를 찾아 Q1~Q4 형태로 반환"""
    if not isinstance(text, str):
        return None
    match = QUARTER_TOKEN_REGEX.search(text)
    if match:
        for group in match.groups():
            if group:
                return f"Q{group.strip()}"
    compact = re.search(r'([1-4])Q', text, re.IGNORECASE)
    if compact:
        return f"Q{compact.group(1)}"
    leading = re.search(r'Q([1-4])', text, re.IGNORECASE)
    if leading:
        return f"Q{leading.group(1)}"
    return None


def _normalize_quarter_value(x: str):
    text = str(x)
    segs = text.split(',')
    for seg in reversed(segs):
        lbl = _extract_quarter_label(seg)
        if lbl:
            return lbl
    lbl = _extract_quarter_label(text)
    if lbl:
        return lbl
    m = re.search(r'([1-4])\s*쿼터', text, re.IGNORECASE)
    if m:
        return f"Q{m.group(1)}"
    m = re.search(r'q\s*([1-4])', text, re.IGNORECASE)
    if m:
        return f"Q{m.group(1)}"
    if len(segs) > 1:
        return segs[-1].strip()
    return text.strip()

def _detect_quarter_column(df: pd.DataFrame):
    """CSV에 존재하는 쿼터 컬럼을 찾거나 Row 기반으로 파생"""
    for cand in QUARTER_COLUMN_CANDIDATES:
        if cand in df.columns:
            return cand
    if 'DetectedQuarter' in df.columns and df['DetectedQuarter'].notna().any():
        return 'DetectedQuarter'

    derived = df['Row'].apply(_extract_quarter_label)
    if derived.notna().any():
        df['DetectedQuarter'] = derived
        return 'DetectedQuarter'

    if 'Ungrouped' in df.columns:
        derived = df['Ungrouped'].apply(_extract_quarter_label)
        if derived.notna().any():
            df['DetectedQuarter'] = derived
            return 'DetectedQuarter'
    return None


def _count_pattern(series: pd.Series, patterns):
    """여러 패턴 중 하나라도 포함된 Row 개수를 반환"""
    if series.empty:
        return 0
    mask = pd.Series(False, index=series.index)
    for pat in patterns:
        mask = mask | series.str.contains(pat, case=False, regex=True, na=False)
    return int(mask.sum())


def _quarter_sort_key(value):
    """시각화 시 Q1~Q4 순으로 정렬하기 위한 키"""
    text = str(value)
    match = re.search(r'([1-4])', text)
    if match:
        return int(match.group(1))
    return 99


def build_quarter_summary(df: pd.DataFrame):
    """??? ?? ?? / ?? / PC / ?? / ?? / SPP ? ??"""
    quarter_col = _detect_quarter_column(df)
    if not quarter_col:
        return pd.DataFrame()

    quarter_df = df.copy()
    quarter_df['Duration'] = pd.to_numeric(quarter_df.get('Duration', 0), errors='coerce').fillna(0)
    quarter_df[quarter_col] = quarter_df[quarter_col].fillna('').astype(str)
    quarter_df = quarter_df[quarter_col] != ''
    quarter_df = df.loc[quarter_df].copy()
    if quarter_df.empty:
        return pd.DataFrame()

    quarter_df['QuarterNorm'] = quarter_df[quarter_col].apply(_normalize_quarter_value)
    quarter_df = quarter_df[quarter_df['QuarterNorm'].isin([f"Q{i}" for i in range(1, 5)])]
    if quarter_df.empty:
        return pd.DataFrame()

    summary_rows = []
    for quarter, group in quarter_df.groupby('QuarterNorm'):
        q_label = str(quarter).strip()
        teams = [t for t in group['Team'].unique() if t not in ['START', 'Unknown', 'givepc', 'getpc', '??', 'YOO', '한국빌드업', '한국프레스', '코치님']]
        for team in teams:
            rows_us = group[group['Team'] == team]
            rows_opp = group[group['Team'] != team]
            if rows_us.empty:
                continue

            row_series = rows_us['Row'].fillna('')

            team_seq_df = rows_us[rows_us['Row'].str.contains('TEAM')]
            att_seq_df = rows_us[rows_us['Row'].str.contains('ATT')]
            team_time = team_seq_df['Duration'].sum()
            att_time = att_seq_df['Duration'].sum()
            opp_team_time = rows_opp[rows_opp['Row'].str.contains('TEAM')]['Duration'].sum()
            opp_att_time = rows_opp[rows_opp['Row'].str.contains('ATT')]['Duration'].sum()
            overall_possession = (team_time / (team_time + opp_team_time) * 100) if (team_time + opp_team_time) > 0 else 0
            att_share = (att_time / (att_time + opp_att_time) * 100) if (att_time + opp_att_time) > 0 else 0

            press_data = compute_press_metrics(rows_us, rows_opp, team_time, att_time)

            build_rows = rows_us[rows_us['Row'].str.contains('DM START|D25 START', regex=True, na=False)]
            build25_attempts = len(build_rows)
            build25_success = len(build_rows[build_rows['결과'].str.contains('25Y entry', na=False)])
            build25_ratio = (build25_success / build25_attempts * 100) if build25_attempts > 0 else 0

            entry25_rows = rows_us[rows_us['Row'].str.contains('A25 START', na=False)]
            entry25_count = len(entry25_rows)
            ce_count = _count_pattern(row_series, QUARTER_METRIC_PATTERNS['Circle Entries'])
            sec_per_ce = att_time / ce_count if ce_count > 0 else 0

            summary_rows.append({
                'Quarter': q_label,
                'Team': team,
                'Circle Entries': _count_pattern(row_series, QUARTER_METRIC_PATTERNS['Circle Entries']),
                'Shots': _count_pattern(row_series, QUARTER_METRIC_PATTERNS['Shots']),
                'PCs': _count_pattern(row_series, QUARTER_METRIC_PATTERNS['PCs']),
                'Goals': _count_pattern(row_series, QUARTER_METRIC_PATTERNS['Goals']),
                'Build25 Attempts': build25_attempts,
                'Build25 Success': build25_success,
                'Build25 Ratio (%)': round(build25_ratio, 1),
                '25y Entries': entry25_count,
                'Sec per CE': round(sec_per_ce, 2),
                'Press Attempts': press_data['press_attempts'],
                'Press Success': press_data['press_success'],
                'Duration (s)': rows_us['Duration'].fillna(0).sum(),
                'Possession (%)': round(overall_possession, 1),
                'ATT Possession (%)': round(att_share, 1),
                'SPP (초/회)': round(press_data['spp'], 2),
                'Allowed SPP (초/회)': round(press_data['allowed_spp'], 2)
            })

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary

    summary['QuarterSort'] = summary['Quarter'].apply(_quarter_sort_key)
    summary = summary.sort_values(['QuarterSort', 'Team']).drop(columns='QuarterSort')
    return summary.reset_index(drop=True)


def _overlap_sum(df: pd.DataFrame, start: float, end: float) -> float:
    """구간 [start, end) 와 행별 시간대(start_sec~end_sec)의 겹치는 길이 합산"""
    if df.empty:
        return 0.0
    overlap = np.minimum(df['end_sec'], end) - np.maximum(df['start_sec'], start)
    return float(overlap.clip(lower=0).sum())


def build_possession_timeline(df: pd.DataFrame, teams, window_seconds: int = 60):
    """
    Duration 누적 시간을 이용해 1분 단위 점유율(TEAM/ATT)을 계산
    팀 리스트는 metrics_df에서 받은 순서를 그대로 사용
    """
    if df.empty or 'Duration' not in df.columns or not teams:
        return pd.DataFrame()

    data = df.copy()
    data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce').fillna(0)
    data = data[data['Duration'] > 0].reset_index(drop=True)
    if data.empty:
        return pd.DataFrame()

    data['start_sec'] = data['Duration'].cumsum().shift(fill_value=0)
    data['end_sec'] = data['start_sec'] + data['Duration']
    max_time = data['end_sec'].max()
    if max_time <= 0:
        return pd.DataFrame()

    windows = np.arange(0, max_time + window_seconds, window_seconds)
    records = []
    for i in range(len(windows) - 1):
        start, end = float(windows[i]), float(windows[i + 1])
        window_rows = data[(data['end_sec'] > start) & (data['start_sec'] < end)]
        if window_rows.empty:
            continue

        team_rows = window_rows[window_rows['Row'].str.contains('TEAM', na=False)]
        att_rows = window_rows[window_rows['Row'].str.contains('ATT', na=False)]

        for team in teams:
            us_team_time = _overlap_sum(team_rows[team_rows['Team'] == team], start, end)
            opp_team_time = _overlap_sum(team_rows[team_rows['Team'] != team], start, end)
            us_att_time = _overlap_sum(att_rows[att_rows['Team'] == team], start, end)
            opp_att_time = _overlap_sum(att_rows[att_rows['Team'] != team], start, end)

            poss = (us_team_time / (us_team_time + opp_team_time) * 100) if (us_team_time + opp_team_time) > 0 else 0
            att_poss = (us_att_time / (us_att_time + opp_att_time) * 100) if (us_att_time + opp_att_time) > 0 else 0

            records.append({
                'Minute': int(start // 60) + 1,
                'Team': team,
                'Possession (%)': round(poss, 1),
                'ATT Possession (%)': round(att_poss, 1)
            })

    return pd.DataFrame(records)


def build_possession_3min_by_quarter(df: pd.DataFrame, teams, quarter_col: str, window_seconds: int = 180):
    """
    쿼터별로 3분 단위 ATT 점유율 평균을 계산
    - quarter_col: 원본 DF의 쿼터 컬럼명 (감지 실패 시 None)
    """
    if df.empty or not teams or 'Duration' not in df.columns or not quarter_col:
        return pd.DataFrame()

    data = df.copy()
    data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce').fillna(0)
    data = data[data['Duration'] > 0]
    if data.empty:
        return pd.DataFrame()

    # 쿼터 라벨 정규화
    data['QuarterNorm'] = data[quarter_col].apply(_normalize_quarter_value)
    data = data[data['QuarterNorm'].isin([f"Q{i}" for i in range(1, 5)])]
    if data.empty:
        return pd.DataFrame()

    records = []
    for q in sorted(data['QuarterNorm'].unique(), key=_quarter_sort_key):
        q_rows = data[data['QuarterNorm'] == q].copy()
        q_rows['start_sec'] = q_rows['Duration'].cumsum().shift(fill_value=0)
        q_rows['end_sec'] = q_rows['start_sec'] + q_rows['Duration']
        max_time = q_rows['end_sec'].max()
        if max_time <= 0:
            continue

        windows = np.arange(0, max_time + window_seconds, window_seconds)
        for i in range(len(windows) - 1):
            start, end = float(windows[i]), float(windows[i + 1])
            window_rows = q_rows[(q_rows['end_sec'] > start) & (q_rows['start_sec'] < end)]
            if window_rows.empty:
                continue

            att_rows = window_rows[window_rows['Row'].str.contains('ATT', na=False)]

            for team in teams[:2]:
                us_att = _overlap_sum(att_rows[att_rows['Team'] == team], start, end)
                opp_att = _overlap_sum(att_rows[att_rows['Team'] != team], start, end)
                att_poss = (us_att / (us_att + opp_att) * 100) if (us_att + opp_att) > 0 else 0
                records.append({
                    'Quarter': q,
                    'WindowIdx': i + 1,
                    'WindowStartMin': round(start / 60.0, 1),
                    'Team': team,
                    'ATT Possession (%)': round(att_poss, 1)
                })
    return pd.DataFrame(records)


def _add_advantage_bands(fig, x_vals, series_a, series_b, color_a, color_b):
    """구간별 우위 팀 색으로 배경 음영 추가 (y 전체 영역)"""
    if not x_vals or len(x_vals) < 2:
        return
    for i in range(len(x_vals) - 1):
        start_x = x_vals[i]
        end_x = x_vals[i + 1]
        a_val = series_a[i]
        b_val = series_b[i]
        if a_val == b_val:
            continue
        win_color = color_a if a_val > b_val else color_b
        fig.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=start_x, x1=end_x, y0=0, y1=1,
            fillcolor=win_color,
            opacity=0.12,
            line=dict(width=0)
        )


def _insert_cross_points(x_vals, y_a, y_b):
    """교차 지점을 선형 보간으로 삽입해 촘촘한 라인을 만든다."""
    if not x_vals:
        return [], [], []

    dense_x = [x_vals[0]]
    dense_a = [y_a[0]]
    dense_b = [y_b[0]]

    for i in range(len(x_vals) - 1):
        x0, x1 = x_vals[i], x_vals[i + 1]
        a0, a1 = y_a[i], y_a[i + 1]
        b0, b1 = y_b[i], y_b[i + 1]
        diff0 = a0 - b0
        diff1 = a1 - b1
        if diff0 * diff1 < 0:
            t = diff0 / (diff0 - diff1)
            x_cross = x0 + t * (x1 - x0)
            a_cross = a0 + (a1 - a0) * t
            b_cross = b0 + (b1 - b0) * t
            dense_x.append(x_cross)
            dense_a.append(a_cross)
            dense_b.append(b_cross)
        dense_x.append(x1)
        dense_a.append(a1)
        dense_b.append(b1)
    return dense_x, dense_a, dense_b


def _add_segment_fill(fig, xs, top_vals, bottom_vals, color):
    fig.add_trace(go.Scatter(
        x=xs,
        y=bottom_vals,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=xs,
        y=top_vals,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=color,
        hoverinfo='skip',
        showlegend=False
    ))


def _apply_dynamic_shading(fig, dense_x, dense_a, dense_b, color_a, color_b, prefer_lower=True):
    """육안상 위쪽 라인에 해당하는 영역에 색상을 칠한다."""
    if len(dense_x) < 2:
        return

    for i in range(len(dense_x) - 1):
        xs = [dense_x[i], dense_x[i + 1]]
        a_seg = [dense_a[i], dense_a[i + 1]]
        b_seg = [dense_b[i], dense_b[i + 1]]
        if any(val is None for val in a_seg + b_seg):
            continue

        avg_a = (a_seg[0] + a_seg[1]) / 2
        avg_b = (b_seg[0] + b_seg[1]) / 2
        if abs(avg_a - avg_b) < 1e-9:
            continue

        if prefer_lower:
            prefer_a = avg_a < avg_b
        else:
            prefer_a = avg_a > avg_b

        if prefer_a:
            top, bottom, color = a_seg, b_seg, color_a
        else:
            top, bottom, color = b_seg, a_seg, color_b

        _add_segment_fill(fig, xs, top, bottom, color)


# -----------------------------------------------------------------------------
# 1-2. 턴오버 위치 시각화를 위한 함수 
# -----------------------------------------------------------------------------

TURNOVER_BANDS = ['Circle', 'F50', 'B50', 'D25']
TURNOVER_LANES = ['Left', 'Center', 'Right']

LANE_KEYWORDS = {
    'Left': ['LEFT', '왼', 'L '],
    'Right': ['RIGHT', '오', 'R ']
}

BAND_KEYWORDS = {
    'Circle': ['CIRCLE', '서클', 'ATTACK CIRCLE', 'A25', 'ATT25'],
    'F50': ['F50', 'A50', '75', '100'],
    'B50': ['B50', 'MID', '50'],
    'D25': ['D25', 'DEF', '25', '0']
}


def _map_turnover_zone(loc_str):
    if not isinstance(loc_str, str):
        return 'Unknown', 'Center'
    text = loc_str.strip()
    upper_text = text.upper()

    prefix_map = {
        '좌': 'Left',
        '중': 'Center',
        '우': 'Right',
        'LEFT': 'Left',
        'CENTER': 'Center',
        'RIGHT': 'Right',
        'L': 'Left',
        'C': 'Center',
        'R': 'Right'
    }
    band_code_map = {
        '25': 'Circle',
        '50': 'F50',
        '75': 'B50',
        '100': 'D25'
    }

    lane = 'Center'
    band = 'D25'

    if '_' in text:
        prefix, suffix = text.split('_', 1)
        prefix = prefix.strip()
        suffix = suffix.strip()
        lane = prefix_map.get(prefix, lane)
        band = band_code_map.get(suffix, band)
    else:
        for key, keywords in LANE_KEYWORDS.items():
            if any(k in upper_text for k in keywords):
                lane = key
                break
        for key, keywords in BAND_KEYWORDS.items():
            if any(k in upper_text for k in keywords):
                band = key
                break
        if band == 'B50' and 'F50' in upper_text:
            band = 'F50'

    return band if band in TURNOVER_BANDS else 'D25', lane


def build_turnover_counts(df, team):
    turnovers = df[(df['Team'] == team) & (df['Row'].str.contains('턴오버', na=False))]
    if turnovers.empty:
        return pd.DataFrame(0, index=TURNOVER_BANDS, columns=TURNOVER_LANES)

    counts = pd.DataFrame(0, index=TURNOVER_BANDS, columns=TURNOVER_LANES)
    for _, row in turnovers.iterrows():
        band, lane = _map_turnover_zone(row.get('지역', ''))
        if band in counts.index and lane in counts.columns:
            counts.loc[band, lane] += 1
    return counts


def build_zone_counts(df):
    """팀 필터/키워드 없이 지역별 빈도만 세기 (압박 지도용)"""
    if df is None or df.empty:
        return pd.DataFrame(0, index=TURNOVER_BANDS, columns=TURNOVER_LANES)
    counts = pd.DataFrame(0, index=TURNOVER_BANDS, columns=TURNOVER_LANES)
    for _, row in df.iterrows():
        band, lane = _map_turnover_zone(row.get('지역', ''))
        if band in counts.index and lane in counts.columns:
            counts.loc[band, lane] += 1
    return counts


def press_attempt_events_for_team(df_all: pd.DataFrame, team: str):
    """
    SPP에서 사용한 press_attempt 정의와 동일한 이벤트를 반환:
    - 우리 팀 턴오버(지역 75/100)
    - 우리 팀 파울(지역 75/100)
    - 상대 팀 파울(지역 25/50)
    """
    df = df_all.copy()
    # 위치 마스크
    mask_high = df['지역'].astype(str).str.contains(r'75|100', na=False)
    mask_low = df['지역'].astype(str).str.contains(r'25|50', na=False)
    # 우리 이벤트: 턴오버/파울 & 지역 75/100
    ours = df[(df['Team'] == team) &
              mask_high &
              df['Row'].str.contains(r'턴오버|파울', case=False, na=False, regex=True)]
    # 상대 이벤트: 파울 & 지역 25/50
    opps = df[(df['Team'] != team) &
              mask_low &
              df['Row'].str.contains(r'파울', case=False, na=False, regex=True)]
    return pd.concat([ours, opps], ignore_index=True)


def build_turnover_field_figure(counts_df, max_ref=None, show_scale=False, view_half="defensive"):
    length = 91.4
    width = 55.0
    half_length = length / 2
    is_attacking = view_half == "attacking"
    lane_width = width / len(TURNOVER_LANES)
    band_ranges = (
        {
            'F50': (half_length, 68.5),
            'Circle': (68.5, length)
        }
        if is_attacking else
        {
            'D25': (0.0, 22.9),
            'B50': (22.9, half_length)
        }
    )

    max_count = counts_df.values.max() if counts_df is not None else 0
    if max_ref is not None:
        max_count = max(max_ref, 1)
    else:
        max_count = max(max_count, 1)

    fig = go.Figure()

    # shading per zone (rotated: x=lane, y=band)
    for band, (y0, y1) in band_ranges.items():
        for lane_idx, lane in enumerate(TURNOVER_LANES):
            count = counts_df.loc[band, lane] if band in counts_df.index else 0
            intensity = 0.25 + 0.5 * (count / max_count)
            fill = f'rgba(214, 39, 40, {intensity})' if count > 0 else 'rgba(0,0,0,0)'
            x0 = lane_idx * lane_width
            x1 = x0 + lane_width
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                          line=dict(color="rgba(0,0,0,0)"), fillcolor=fill, layer="below")

    # pitch boundary (rotated: x=0..width, y=half)
    y0, y1 = (half_length, length) if is_attacking else (0.0, half_length)
    fig.add_shape(type="rect", x0=0, x1=width, y0=y0, y1=y1,
                  line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)")

    # center line and 23m line (horizontal)
    fig.add_shape(type="line", x0=0, x1=width, y0=half_length, y1=half_length,
                  line=dict(color="black", width=2))
    line_23m = (length - 22.9) if is_attacking else 22.9
    fig.add_shape(type="line", x0=0, x1=width, y0=line_23m, y1=line_23m,
                  line=dict(color="black", width=2))

    cx = width / 2

    def add_arc(yc, theta_start, theta_end, radius, dashed=False, upward=True):
        theta = np.radians(np.linspace(theta_start, theta_end, 200))
        x = cx + radius * np.cos(theta)
        sign = 1 if upward else -1
        y = yc + sign * radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color="black", width=2, dash='dash' if dashed else 'solid'),
            hoverinfo='skip',
            showlegend=False
        ))

    # arcs near goal line (only the selected half)
    if is_attacking:
        add_arc(length, 180, 360, 20.0, dashed=True, upward=False)
        add_arc(length, 180, 360, 14.63, dashed=False, upward=False)
    else:
        add_arc(0, 0, 180, 20.0, dashed=True, upward=True)
        add_arc(0, 0, 180, 14.63, dashed=False, upward=True)

    # penalty spots
    spot_y = (length - 6.475) if is_attacking else 6.475
    fig.add_shape(type="circle",
                  x0=cx - 0.3, x1=cx + 0.3,
                  y0=spot_y - 0.3, y1=spot_y + 0.3,
                  fillcolor="black", line=dict(color="black"))

    # goals (simplified)
    goal_depth = 0.6
    post_off = 1.83
    goal_y0 = (length - goal_depth) if is_attacking else 0
    fig.add_shape(type="rect",
                  x0=cx - post_off, x1=cx + post_off,
                  y0=goal_y0, y1=goal_y0 + goal_depth,
                  line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)")

    # text annotations (counts) - keep 위치 동일
    text_x = []
    text_y = []
    text_val = []
    for band, (y0, y1) in band_ranges.items():
        for lane_idx, lane in enumerate(TURNOVER_LANES):
            count = counts_df.loc[band, lane] if band in counts_df.index else 0
            x0 = lane_idx * lane_width
            text_x.append((x0 + x0 + lane_width) / 2)
            text_y.append((y0 + y1) / 2)
            text_val.append(str(int(count)))

    fig.add_trace(go.Scatter(
        x=text_x, y=text_y,
        mode='text',
        text=text_val,
        textfont=dict(color='black', size=16, family="Arial Black"),
        hoverinfo='skip'
    ))

    # attack direction marker (outside pitch)
    arrow_x = width + 6
    y_min, y_max = (half_length, length) if is_attacking else (0.0, half_length)
    arrow_y = y_min + (y_max - y_min) * 0.5
    fig.add_annotation(
        x=arrow_x, y=arrow_y + 5,
        ax=arrow_x, ay=arrow_y - 5,
        xref="x", yref="y", axref="x", ayref="y",
        text="공격방향",
        showarrow=True,
        arrowhead=2, arrowsize=1.4, arrowwidth=3,
        arrowcolor="black",
        font=dict(color="black", size=12),
        xshift=10, xanchor="left"
    )

    # 공격 방향 텍스트 제거 (숫자만 표시)

    aspect_ratio = width / length
    base_width = 900
    fig.update_layout(
        xaxis=dict(visible=False, range=[-2, length + 2]),
        # 스케일 고정(가로:세로 1:1)으로 짜부 방지
        yaxis=dict(visible=False, range=[-2, width + 2], scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        width=base_width,
        height=int(base_width * aspect_ratio),
        margin=dict(l=10, r=10, t=80, b=60)
    )

    if show_scale:
        # 절대 기준 색상표 (0 ~ max_count)
        fig.add_trace(go.Heatmap(
            z=[[0, max_count]],
            colorscale=[[0, "#ffffff"], [1, "#d62728"]],
            showscale=True,
            colorbar=dict(title="압박 횟수", len=0.7, thickness=12),
            x=[length + 5, length + 6], y=[-5, -4],  # 화면 밖에 배치
            hoverinfo='skip'
        ))
    return fig


def render_entry_analysis(df: pd.DataFrame):
    """
    서클 진입 위치별 성과 분석 (하프필드 도식 + 좌/중/우 Entries·Success·효율)
    """
    st.markdown("### 🗺️ 서클 진입 위치 & 성과 분석")
    if df.empty:
        st.warning("데이터가 없습니다.")
        return

    # 서클 진입 이벤트 탐지: "국가명 슈팅서클 진입" 패턴이 있으면 그것만 사용
    mask_country_shoot = df['Row'].str.contains(r'\S+\s*슈팅서클\s*진입', case=False, na=False, regex=True)
    if mask_country_shoot.any():
        entry_mask = mask_country_shoot
    else:
        entry_mask = df['Row'].str.contains(
            r'서클\s*진입|Circle Entry|A25',
            case=False, na=False, regex=True
        )
    entry_df = df[entry_mask].copy()
    if entry_df.empty:
        st.info("'서클 진입' 관련 데이터를 찾을 수 없습니다.")
        return

    # 팀/쿼터 필터 (컬럼명을 느슨하게 매칭)
    def _find_col(df_in, candidates):
        def norm_name(name: str):
            # 공백/언더스코어/하이픈 제거 + 소문자
            return re.sub(r'[^0-9a-zA-Z\u3131-\u318e\uac00-\ud7a3]', '', str(name).lower())

        norm_map = {norm_name(c): c for c in df_in.columns}
        for cand in candidates:
            key = norm_name(cand)
            if key in norm_map:
                return norm_map[key]
        # 부분일치(예: qtr, quarter_no 등)
        for k, orig in norm_map.items():
            if any(key in k for key in map(norm_name, candidates)):
                return orig
        return None

    # 필터 UI 컬럼
    c_team, c_q = st.columns(2)

    team_col = _find_col(entry_df, ['Team', '팀', 'HomeTeam', 'AwayTeam'])
    quarter_col = _find_col(entry_df, ['Quarter', '쿼터', 'Q', 'Period', 'Qtr', 'QuarterNo', 'Ungrouped'])
    # 값 기반 쿼터 컬럼 추론 (예: 1쿼터, 2쿼터, 전반/후반 등)
    if quarter_col is None:
        regex_q = r'(?:^|\\s)([1-4])\\s*쿼터|\\bQ[1-4]\\b|\\b[1-4]Q\\b|전반|후반'
        best_col = None
        best_score = 0
        for col in entry_df.columns:
            series = entry_df[col].dropna().astype(str)
            if series.empty:
                continue
            score = series.str.contains(regex_q, case=False, regex=True, na=False).mean()
            if score > best_score:
                best_score = score
                best_col = col
        manual_opts = ['선택 안함'] + entry_df.columns.tolist()
        default_idx = manual_opts.index(best_col) if best_col in manual_opts else 0
        picked = c_q.selectbox("쿼터 컬럼을 선택하세요 (없으면 '선택 안함')", manual_opts, index=default_idx, key="entry_q_pick")
        quarter_col = None if picked == '선택 안함' else picked
    if team_col:
        team_opts = ['전체'] + sorted(entry_df[team_col].dropna().unique().tolist())
        sel_team = c_team.selectbox("팀 선택", team_opts, index=0)
        if sel_team != '전체':
            entry_df = entry_df[entry_df[team_col] == sel_team]
        else:
            sel_team = '전체'
            c_team.caption("Team 컬럼이 없어 팀 필터를 생략합니다.")
    if quarter_col:
        # 쿼터 숫자 추출
        def _qnum(v):
            if pd.isna(v):
                return None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    iv = int(v)
                    return iv if 1 <= iv <= 4 else None
                except Exception:
                    pass
            s = str(v)
            if '전반' in s:
                return 1
            if '후반' in s:
                return 3
            # 우선 '4쿼터' 'Q4' '4Q' 같은 패턴에서 숫자 추출
            for pattern in [r'([1-4])\s*쿼터', r'[Qq]\s*([1-4])', r'([1-4])\s*[Qq]']:
                m = re.search(pattern, s)
                if m:
                    return int(m.group(1))
            # 점수 숫자와 섞여 있을 때는 마지막 1~4를 사용
            nums = re.findall(r'([1-4])', s)
            return int(nums[-1]) if nums else None

        entry_df['__qnum'] = entry_df[quarter_col].apply(_qnum)
        q_vals = sorted([q for q in entry_df['__qnum'].dropna().unique().tolist() if 1 <= q <= 4])
        all_qs = [1, 2, 3, 4]  # 직접선택 옵션에 항상 1~4 노출
        q_label_map = {q: f"{q}쿼터" for q in all_qs}

        mode = c_q.radio("쿼터/전후반 선택", ["전체", "전반", "후반", "직접선택"], horizontal=True, key="entry_q_mode")
        if mode == "전체":
            sel_qs = q_vals
        elif mode == "전반":
            sel_qs = [q for q in q_vals if q <= 2]
        elif mode == "후반":
            sel_qs = [q for q in q_vals if q >= 3]
        else:
            opts_labels = [q_label_map[q] for q in all_qs]
            default_labels = [q_label_map[q] for q in (q_vals if q_vals else all_qs)]
            sel_labels = c_q.multiselect("쿼터 선택", opts_labels, default=default_labels, key="entry_q_multi")
            sel_qs = [q for q in all_qs if q_label_map[q] in sel_labels]

        if sel_qs:
            entry_df = entry_df[entry_df['__qnum'].isin(sel_qs)]
    else:
        sel_qs = []

    if entry_df.empty:
        st.info("선택한 필터에 해당하는 데이터가 없습니다.")
        return

    def classify_zone(text):
        if not isinstance(text, str):
            return 'Unknown'
        # 좌_25, 중_25, 우_25 등의 패턴을 깨끗이 정리
        cleaned = text.replace('_', ' ').upper()
        cleaned = ''.join(ch for ch in cleaned if not ch.isdigit())
        if any(x in cleaned for x in ['LEFT', '좌', 'L ']):
            return 'Left'
        if any(x in cleaned for x in ['RIGHT', '우', 'R ']):
            return 'Right'
        if any(x in cleaned for x in ['CENTER', '중', 'C ']):
            return 'Center'
        return 'Unknown'

    entry_df['Zone'] = entry_df.apply(lambda x: classify_zone(x.get('지역', '')) or classify_zone(x.get('Row', '')), axis=1)

    def check_success(text):
        if not isinstance(text, str):
            return 0
        t = text.upper()
        if any(x in t for x in ['슈팅', 'SHOT', 'PC', 'CORNER', '득점', 'GOAL']):
            return 1
        return 0
    def check_shot(text):
        if not isinstance(text, str):
            return 0
        t = text.upper()
        return 1 if any(x in t for x in ['SHOT', '슈팅']) else 0
    def check_goal(text):
        if not isinstance(text, str):
            return 0
        t = text.upper()
        return 1 if any(x in t for x in ['GOAL', '득점']) else 0
    def check_pc(text):
        if not isinstance(text, str):
            return 0
        t = text.upper()
        return 1 if any(x in t for x in ['PC', 'P.C', 'CORNER']) else 0

    entry_df['Is_Success'] = entry_df['결과'].apply(check_success)
    entry_df['Is_Shot'] = entry_df['결과'].apply(check_shot)
    entry_df['Is_Goal'] = entry_df['결과'].apply(check_goal)
    entry_df['Is_PC'] = entry_df['결과'].apply(check_pc)

    stats = entry_df.groupby('Zone').agg(
        Entries=('Row', 'count'),
        Shots=('Is_Shot', 'sum'),
        Goals=('Is_Goal', 'sum'),
        PCs=('Is_PC', 'sum'),
        Success=('Is_Success', 'sum')
    ).reset_index()

    zone_order = ['Left', 'Center', 'Right']
    stats = stats.set_index('Zone').reindex(zone_order).fillna(0).reset_index()

    # 좌표 확인용 안내점 토글
    show_coords = st.checkbox("마우스 좌표 보기 (디버그용)", value=True, key="show_coords_entry")

    fig = go.Figure()
    # 필드/규격 기반 도식 (단위 m, 필드 너비 55m)
    field_w = 55.0
    field_h = 25.0
    cx = field_w / 2
    goal_w = 3.66
    goal_depth = 1.2
    goal_left = cx - goal_w / 2
    goal_right = cx + goal_w / 2
    back_y = field_h  # 백라인을 상단으로

    fig.add_shape(type="rect", x0=0, y0=0, x1=field_w, y1=field_h, line=dict(color="black", width=3), fillcolor="rgba(0,0,0,0)")
    fig.add_shape(type="rect", x0=goal_left, x1=goal_right, y0=back_y - goal_depth, y1=back_y, line=dict(color="black", width=3), fillcolor="rgba(0,0,0,0)")
    for dx in [-6, -3, 3, 6]:  # 중앙 눈금 제거
        fig.add_shape(type="line", x0=cx+dx, x1=cx+dx, y0=back_y, y1=back_y - goal_depth*0.8, line=dict(color="black", width=2))

    line_y = back_y - 14.63
    fig.add_shape(type="line", x0=goal_left, x1=goal_right, y0=line_y, y1=line_y, line=dict(color="black", width=4))

    r_main = 14.63
    r_broken = r_main + 5.0
    theta = np.linspace(0, np.pi/2, 80)

    left_arc_x = goal_left - r_main * np.sin(theta)
    left_arc_y = back_y - r_main * np.cos(theta)
    right_arc_x = goal_right + r_main * np.sin(theta)
    right_arc_y = back_y - r_main * np.cos(theta)
    fig.add_trace(go.Scatter(x=left_arc_x, y=left_arc_y, mode='lines', line=dict(color='black', width=4), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=right_arc_x, y=right_arc_y, mode='lines', line=dict(color='black', width=4), hoverinfo='skip', showlegend=False))

    left_arc_b_x = goal_left - r_broken * np.sin(theta)
    left_arc_b_y = back_y - r_broken * np.cos(theta)
    right_arc_b_x = goal_right + r_broken * np.sin(theta)
    right_arc_b_y = back_y - r_broken * np.cos(theta)
    fig.add_trace(go.Scatter(x=left_arc_b_x, y=left_arc_b_y, mode='lines', line=dict(color='black', width=3, dash='dash'), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=right_arc_b_x, y=right_arc_b_y, mode='lines', line=dict(color='black', width=3, dash='dash'), hoverinfo='skip', showlegend=False))
    broken_y = line_y - 5.0
    fig.add_shape(type="line", x0=goal_left, x1=goal_right, y0=broken_y, y1=broken_y, line=dict(color="black", width=3, dash="dash"))

    fig.add_trace(go.Scatter(x=[cx], y=[back_y - 5], mode='markers', marker=dict(color='black', size=8), showlegend=False))
    # 화살표: 필드 방향(위쪽 골대)으로 향하도록 상하 반전 + y축 5 내려 배치
    arrows = [
        dict(x=11.0,  y=21.0, ax=2.75,  ay=13.0),    # 왼쪽 대각 (머리 위쪽)
        dict(x=27.5,  y=11.0, ax=27.5,  ay=0.0),     # 중앙 꼬리 y축 -3 (길이 +3)
        dict(x=44.0,  y=21.0, ax=52.25, ay=13.0)     # 오른쪽 대각
    ]
    for arr in arrows:
        fig.add_annotation(
            x=arr['x'], y=arr['y'],
            ax=arr['ax'], ay=arr['ay'],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.8, arrowwidth=6,
            arrowcolor="rgba(70, 130, 180, 0.7)",
            opacity=0.9
        )

    # 구역별 텍스트 (Entries / Success / 효율)
    text_positions = {
        'Left':   {'x': 5,  'y': 7},   # Left/Right만 Y -10 이동
        'Center': {'x': 30, 'y': 17},
        'Right':  {'x': 50, 'y': 7}
    }
    for _, row in stats.iterrows():
        z = row['Zone']
        if z not in text_positions:
            continue
        entries = int(row['Entries'])
        success = int(row['Success'])
        rate = (success / entries * 100) if entries > 0 else 0
        fig.add_annotation(
            x=text_positions[z]['x'],
            y=text_positions[z]['y'],
            text=(f"<b>{z}</b><br>"
                  f"Entries: <b>{entries}</b>회<br>"
                  f"Success(슈팅/PC/득점): <b>{success}</b>회<br>"
                  f"효율: <b>{rate:.0f}%</b>"),
            showarrow=False,
            font=dict(color="#222", size=14),
            align="center",
            opacity=0.95,
        )

    if show_coords:
        grid_x = []
        grid_y = []
        for x in range(0, 101, 5):
            for y in range(0, 101, 5):
                grid_x.append(x)
                grid_y.append(y)
        fig.add_trace(go.Scatter(
            x=grid_x, y=grid_y,
            mode='markers',
            marker=dict(color='rgba(0,0,0,0.2)', size=4),
            hovertemplate="x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, field_w + 2]),
        # 정상 방향 + 종횡비 유지
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-2, field_h + 5], scaleanchor="x", scaleratio=1
        ),
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        plot_bgcolor="white",
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### 📋 좌·중·우 서클 진입 성과")
    stat_view = stats.copy()
    # 합계 행 추가
    total_row = {
        'Zone': '전체',
        'Entries': stat_view['Entries'].sum(),
        'Shots': stat_view['Shots'].sum(),
        'Goals': stat_view['Goals'].sum(),
        'PCs': stat_view['PCs'].sum(),
        'Success': stat_view['Success'].sum()
    }
    stat_view = pd.concat([stat_view, pd.DataFrame([total_row])], ignore_index=True)

    stat_view['슈팅(득점/시도)'] = stat_view.apply(lambda r: f"{int(r['Goals'])}/{int(r['Shots'])}", axis=1)
    stat_view['효율 (%)'] = stat_view.apply(lambda r: round((r['Success']/r['Entries']*100) if r['Entries']>0 else 0, 1), axis=1)
    display_cols = ['Zone', 'Entries', '슈팅(득점/시도)', 'PCs', 'Success', '효율 (%)']
    st.dataframe(stat_view[display_cols].set_index('Zone'), use_container_width=True)

TIME_COLUMN_CANDIDATES = ['시간', 'Time', 'Clock', 'Match Time', 'Elapsed', 'Game Clock', '경과시간', 'Ungrouped']


def _parse_time_to_seconds(value):
    if not isinstance(value, str):
        value = str(value) if pd.notna(value) else ''
    text = value.strip()
    if not text:
        return None
    match = re.search(r'(\d{1,2})[:;](\d{2})(?:[:;](\d{2}))?', text)
    if match:
        h = int(match.group(3)) if match.group(3) else 0
        m = int(match.group(1))
        s = int(match.group(2))
        if h > 0:
            return h * 3600 + m * 60 + s
        return m * 60 + s
    match = re.search(r'(\d+)\s*\'', text)
    if match:
        return int(match.group(1)) * 60
    try:
        return float(text)
    except ValueError:
        return None


def build_shot_pc_events(df):
    if 'Row' not in df.columns or 'Team' not in df.columns:
        return pd.DataFrame()
    mask = df['Row'].str.contains('슈팅|페널티|PC', case=False, na=False)
    events = df[mask].copy()
    if events.empty:
        return events

    def extract_seconds(row):
        for col in TIME_COLUMN_CANDIDATES:
            if col in row and pd.notna(row[col]):
                secs = _parse_time_to_seconds(row[col])
                if secs is not None:
                    return secs
        return None

    events['event_seconds'] = events.apply(extract_seconds, axis=1)
    if events['event_seconds'].notna().sum() == 0:
        events['event_seconds'] = np.arange(len(events)) * 60
    else:
        events['event_seconds'] = events['event_seconds'].astype(float)
        events['event_seconds'] = events['event_seconds'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both'))

    events['event_order'] = np.arange(len(events))
    return events[['Team', 'event_seconds', 'event_order']]


def _build_cumulative_step(series, match_end_seconds):
    if series.empty:
        return [0], [0]
    secs = series.sort_values().tolist()
    times = [0]
    values = [0]
    current = 0
    for sec in secs:
        times.extend([sec, sec])
        values.extend([current, current + 1])
        current += 1
    final_time = match_end_seconds if match_end_seconds else secs[-1]
    times.append(final_time)
    values.append(current)
    return np.array(times) / 60.0, values

# -----------------------------------------------------------------------------
# 2. 지표 계산 로직 (User Formula 반영)
# -----------------------------------------------------------------------------

def calculate_metrics(df):
    teams = [t for t in df['Team'].unique() if t not in ['START', 'Unknown', 'givepc', 'getpc', '코치님', 'YOO', '한국빌드업', '한국프레스']]
    # 주요 2개 팀만 선정 (데이터량이 많은 순)
    if len(teams) > 2:
        teams = df['Team'].value_counts().index.tolist()[:2]
    
    metrics_list = []
    
    for team in teams:
        opp_team = [t for t in teams if t != team][0]
        
        # --- Data Filtering ---
        # 우리 팀 Rows
        rows_us = df[df['Team'] == team]
        # 상대 팀 Rows
        rows_opp = df[df['Team'] == opp_team]
        
        # --- 1. 시간 및 시퀀스 계산 ---
        # TEAM time: Row가 "<팀> TEAM" 으로 끝나는 것의 Duration 합
        # ATT time: Row가 "<팀> ATT" 로 끝나는 것의 Duration 합
        # (문자열 포함 여부로 판단)
        
        team_seq_df = rows_us[rows_us['Row'].str.contains('TEAM')]
        att_seq_df = rows_us[rows_us['Row'].str.contains('ATT')]
        
        team_time = team_seq_df['Duration'].sum()
        att_time = att_seq_df['Duration'].sum()
        
        team_seq_count = len(team_seq_df)
        att_seq_count = len(att_seq_df)
        
        team_avg_seq_time = team_time / team_seq_count if team_seq_count > 0 else 0
        att_avg_seq_time = att_time / att_seq_count if att_seq_count > 0 else 0
        
        # 상대방 시간 (점유율 계산용)
        opp_team_time = rows_opp[rows_opp['Row'].str.contains('TEAM')]['Duration'].sum()
        opp_att_time = rows_opp[rows_opp['Row'].str.contains('ATT')]['Duration'].sum()
        
        # --- 2. 압박 (Pressing) ---
        press_data = compute_press_metrics(rows_us, rows_opp, team_time, att_time)
        press_attempts = press_data['press_attempts']
        press_success = press_data['press_success']
        press_index = press_data['spp']
        allowed_press_index = press_data['allowed_spp']

        # --- 3. Build25 ---
        # Attempts: Row in {DM START, D25 START}
        build_rows = rows_us[rows_us['Row'].str.contains('DM START|D25 START', regex=True)]
        build25_attempts = len(build_rows)
        # Success: 결과에 "25Y entry" 포함
        build25_success = len(build_rows[build_rows['결과'].str.contains('25Y entry', na=False)])
        build25_ratio = (build25_success / build25_attempts * 100) if build25_attempts > 0 else 0
        
        # --- 4. Event Counts (CE, 25Y, Shot, PC, Goal) ---
        # CE: 서클 진입 관련 패턴 (공백/대소문자 변형 대응)
        ce_rows = rows_us[rows_us['Row'].str.contains(
            r'서클\s*진입|슈팅\s*서클|circle\s*entry|attack\s*circle|att\s*circle',
            case=False, regex=True
        )]
        ce_count = len(ce_rows)
        
        # 25Y: Row = "<팀> A25 START"
        entry25_rows = rows_us[rows_us['Row'].str.contains('A25 START')]
        entry25_count = len(entry25_rows)
        
        # Shot: Row가 정확히 "<팀> 슈팅" 형태만 카운트 (뒤에 다른 단어 붙은 경우 제외)
        shot_pattern_exact = rf'^\s*{re.escape(team)}\s*슈팅(?:\s+\d+)?\s*$'
        shot_rows = df[df['Row'].str.contains(shot_pattern_exact, case=False, regex=True, na=False)]
        shot_count = len(shot_rows)
        
        # PC: Row = "<팀> 페널티코너"
        pc_rows = rows_us[rows_us['Row'].str.contains(
            r'페널티\s*코너|penalty\s*corner|\bPCs?\b|\bPC\b',
            case=False, regex=True
        )]
        pc_count = len(pc_rows)
        
        # Goals
        # PC Goal: PC Row 중 결과에 GOAL 포함
        pc_goals = len(pc_rows[pc_rows['결과'].str.contains('GOAL|득점', regex=True, na=False)])
        # Total Goal: Row에 "득점" 포함 (데이터셋 패턴: '인도 득점')
        total_goal_rows = rows_us[rows_us['Row'].str.contains('득점')]
        total_goals = len(total_goal_rows)
        # Field Goal
        field_goals = total_goals - pc_goals
        if field_goals < 0: field_goals = 0 # 예외처리
        
        # --- 5. Possession & Sec/CE ---
        overall_possession = (team_time / (team_time + opp_team_time) * 100) if (team_time + opp_team_time) > 0 else 0
        att_share = (att_time / (att_time + opp_att_time) * 100) if (att_time + opp_att_time) > 0 else 0
        
        sec_per_ce = att_time / ce_count if ce_count > 0 else 0
        
        # 결과 저장
        metrics_list.append({
            'Team': team,
            'Goals (Total)': total_goals,
            'Field Goals': field_goals,
            'PC Goals': pc_goals,
            'Shots': shot_count,
            'Circle Entries (CE)': ce_count,
            '25y Entries': entry25_count,
            'PCs': pc_count,
            'Possession (%)': round(overall_possession, 1),
            'ATT Possession (%)': round(att_share, 1),
            'Sec per CE': round(sec_per_ce, 1),
            'Build25 Ratio (%)': round(build25_ratio, 1),
            'Press Attempts': press_attempts,
            'Press Success': press_success,
            'SPP (초/회)': round(press_index, 2),
            'Allowed SPP (초/회)': round(allowed_press_index, 2),
            'Avg Seq Time (TEAM)': round(team_avg_seq_time, 1),
            'Avg Seq Time (ATT)': round(att_avg_seq_time, 1)
        })
        
    return pd.DataFrame(metrics_list)

# -----------------------------------------------------------------------------
# 3. 메인 앱 UI
# -----------------------------------------------------------------------------

# (GPS용 보조 함수 - 간단 로더/표시)
@st.cache_data
def process_gps_data(file):
    """GPS CSV 파일 로드 및 전처리 (인코딩 자동 재시도)"""
    file.seek(0)
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        try:
            df = pd.read_csv(file, encoding='cp949')
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1')
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, encoding='cp949')
    df.columns = df.columns.str.strip()
    return df


def render_gps_dashboard(gps_df: pd.DataFrame):
    """GPS dashboard with radar / workload map / quarter heatmap / parallel coordinates."""
    if gps_df.empty:
        st.warning("데이터가 없습니다.")
        return

    hsr_cols = [c for c in gps_df.columns if 'Velocity Band' in c and 'Total Distance' in c and any(x in c for x in ['5', '6', '7', '8'])]
    gps_df = gps_df.copy()
    gps_df['HSR Distance'] = gps_df[hsr_cols].sum(axis=1) if hsr_cols else 0

    st.markdown("### 🏃‍♂️ GPS 피지컬 분석")
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "레이더 차트(프로필 비교)",
        "활동량 맵",
        "쿼터별 히트맵",
        "성과 비교 (Parallel)"
    ])

    # Session 데이터 추출
    if 'Period Name' in gps_df.columns:
        session_data = gps_df[gps_df['Period Name'] == 'Session'].copy()
        if session_data.empty:
            session_data = gps_df.copy()
    else:
        session_data = gps_df.copy()

    # 1) 레이더 차트
    with sub_tab1:
        st.markdown("#### 선수 프로필 & 평균 비교")
        st.caption("선수/지표/평균(팀·포지션)을 선택해 비교하세요.")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            players = sorted(session_data['Player Name'].dropna().unique()) if 'Player Name' in session_data.columns else []
            default_players = session_data.nlargest(2, 'Total Distance')['Player Name'].tolist() if 'Total Distance' in session_data.columns and not session_data.empty else []
            radar_players = st.multiselect("비교할 선수", options=players, default=default_players, key="radar_players_final")
        with c2:
            show_team_avg = st.checkbox("팀 평균 보기", value=True, key="radar_team_avg_final")
            pos_opts = sorted(session_data['Position Name'].dropna().unique()) if 'Position Name' in session_data.columns else []
            radar_pos_avgs = st.multiselect("포지션 평균 추가", options=pos_opts, key="radar_pos_avg_final")
        with c3:
            num_cols = session_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            exclude_keywords = ['Unix', 'Time', 'Number', 'Band', '%', 'Effort']
            metric_candidates = [c for c in num_cols if not any(k in c for k in exclude_keywords)]
            if 'HSR Distance' in session_data.columns and 'HSR Distance' not in metric_candidates:
                metric_candidates.append('HSR Distance')
            default_metrics = [m for m in ['Total Distance', 'Meterage Per Minute', 'Maximum Velocity', 'HSR Distance', 'Total Player Load'] if m in session_data.columns]
            radar_metrics = st.multiselect("차트 축 지표", options=metric_candidates, default=default_metrics, key="radar_metrics_final")

        if not radar_metrics:
            st.info("지표를 한 개 이상 선택하세요.")
        elif not radar_players and not show_team_avg and not radar_pos_avgs:
            st.info("선수 또는 평균을 선택하세요.")
        else:
            max_vals = {m: session_data[m].max() if m in session_data.columns and session_data[m].max() > 0 else 1 for m in radar_metrics}
            fig_radar = go.Figure()

            palette = px.colors.qualitative.Plotly

            def add_trace(name, series, color=None, line=None, fill='toself', opacity=0.75):
                vals = [(series[m] / max_vals[m] * 100) if max_vals[m] > 0 else 0 for m in radar_metrics]
                vals += [vals[0]]
                theta = radar_metrics + [radar_metrics[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=theta, name=name,
                    fill=fill, line=line or dict(color=color), opacity=opacity,
                    marker=dict(color=color)
                ))

            # 선수들
            for p in radar_players:
                row = session_data[session_data['Player Name'] == p]
                if not row.empty:
                    idx = radar_players.index(p)
                    add_trace(p, row.iloc[0], color=palette[idx % len(palette)], opacity=0.6)

            if show_team_avg:
                add_trace("Team Average", session_data[radar_metrics].mean(), line=dict(color='black', dash='dot', width=2), fill='none', opacity=0.9)

            for pos in radar_pos_avgs:
                pos_df = session_data[session_data['Position Name'] == pos]
                if not pos_df.empty:
                    add_trace(f"AVG {pos}", pos_df[radar_metrics].mean(), line=dict(color='#ff7f0e', dash='dash', width=2), fill='none', opacity=0.8)

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
                showlegend=True,
                height=520,
                title="선수 프로필 레이더 (100% = 팀 내 최고 기록)",
                margin=dict(t=50, b=40, l=40, r=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with sub_tab2:
        st.markdown("#### 1. 활동량 맵 (Workload Map)")
        st.caption("[선택 구간] 선수별 활동량 분포 및 포지션 평균(★)")
        filt_col1, filt_col2 = st.columns(2)
        filtered_map_df = gps_df.copy()
        with filt_col1:
            if 'Period Name' in gps_df.columns:
                map_periods = sorted(gps_df['Period Name'].dropna().unique().tolist())
                if 'Session' in map_periods:
                    map_periods.remove('Session')
                    map_periods.insert(0, 'Session')
                sel_map_period = st.selectbox("구간 선택", options=map_periods, index=0, key="gps_map_period")
                filtered_map_df = gps_df[gps_df['Period Name'] == sel_map_period].copy()
        with filt_col2:
            if 'Position Name' in filtered_map_df.columns:
                map_positions = sorted(filtered_map_df['Position Name'].dropna().unique())
                sel_map_positions = st.multiselect("포지션 선택", options=map_positions, default=map_positions, key="gps_map_positions")
                filtered_map_df = filtered_map_df[filtered_map_df['Position Name'].isin(sel_map_positions)].copy()
        if filtered_map_df.empty:
            st.warning("조건에 맞는 데이터가 없습니다.")
        elif {'Total Distance', 'Meterage Per Minute'}.issubset(filtered_map_df.columns):
            pos_avg = None
            if 'Position Name' in filtered_map_df.columns and not filtered_map_df['Position Name'].dropna().empty:
                pos_avg = filtered_map_df.groupby('Position Name')[['Total Distance', 'Meterage Per Minute']].mean().reset_index()
            fig_vol = px.scatter(
                filtered_map_df,
                x='Total Distance',
                y='Meterage Per Minute',
                color='Position Name' if 'Position Name' in filtered_map_df.columns else None,
                text='Player Name' if 'Player Name' in filtered_map_df.columns else None,
                hover_data=['Maximum Velocity'] if 'Maximum Velocity' in filtered_map_df.columns else None,
                opacity=0.75
            )
            fig_vol.update_traces(textposition='top center', textfont=dict(size=9), marker=dict(size=10))
            if pos_avg is not None and not pos_avg.empty:
                fig_vol.add_trace(go.Scatter(
                    x=pos_avg['Total Distance'],
                    y=pos_avg['Meterage Per Minute'],
                    mode='markers+text',
                    text=pos_avg['Position Name'],
                    textposition='bottom center',
                    marker=dict(symbol='star', size=18, color='black', line=dict(width=1, color='white')),
                    name='포지션 평균'
                ))
            fig_vol.update_layout(
                xaxis_title="총 이동 거리 (m)",
                yaxis_title="분당 이동 거리 (m/min)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=520
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Total Distance 또는 Meterage Per Minute 컬럼이 없습니다.")

        if 'Maximum Velocity' in filtered_map_df.columns:
            top_speed_df = filtered_map_df.sort_values('Maximum Velocity', ascending=True)
            fig_speed = px.bar(
                top_speed_df,
                x='Maximum Velocity',
                y='Player Name' if 'Player Name' in top_speed_df.columns else None,
                orientation='h',
                text='Maximum Velocity',
                color='Position Name' if 'Position Name' in top_speed_df.columns else None,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_speed.update_traces(texttemplate='%{text:.1f}', textposition='inside')
            fig_speed.update_layout(xaxis_title="km/h", yaxis_title="", showlegend=True, height=520)
            st.plotly_chart(fig_speed, use_container_width=True)

    with sub_tab3:
        st.markdown("#### 쿼터별 강도 흐름 (Heatmap)")
        st.caption("붉은색일수록 분당 활동량(m/min)이 높은 상태, 파란색은 낮은 상태입니다.")
        if 'Period Name' in gps_df.columns:
            q_mask = gps_df['Period Name'].astype(str).str.contains(r'(Q[1-4]|Period|Half|쿼터)', case=False, regex=True) & (gps_df['Period Name'] != 'Session')
            q_df = gps_df[q_mask].copy()
            if not q_df.empty and 'Player Name' in q_df.columns and 'Meterage Per Minute' in q_df.columns:
                heat_data = q_df.pivot_table(index='Player Name', columns='Period Name', values='Meterage Per Minute', aggfunc='mean').fillna(0)
                heat_data = heat_data[sorted(heat_data.columns)]
                fig_heat = px.imshow(
                    heat_data,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    text_auto=".0f",
                    labels=dict(x="쿼터", y="선수", color="m/min")
                )
                fig_heat.update_layout(height=max(500, len(heat_data) * 30))
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("쿼터별 데이터를 만들 수 없습니다.")
        else:
            st.info("'Period Name' 컬럼이 없습니다.")

    with sub_tab4:
        st.markdown("#### 📈 성과 비교 (Parallel Coordinates 스타일)")
        st.caption("선택 선수는 진하게, 나머지는 연하게 표시됩니다.")
        pc_c1, pc_c2 = st.columns(2)
        with pc_c1:
            all_players = sorted(session_data['Player Name'].dropna().unique()) if 'Player Name' in session_data.columns else []
            default_pc_players = session_data.nlargest(3, 'Total Distance')['Player Name'].tolist() if 'Total Distance' in session_data.columns and not session_data.empty else []
            pc_players = st.multiselect("선수 선택 (강조)", options=all_players, default=default_pc_players, key="pc_players_sel")
        with pc_c2:
            num_cols = session_data.select_dtypes(include=['number']).columns.tolist()
            exclude = ['Unix', 'Time', 'Number', 'Band', '%', 'Effort']
            rec_metrics = [c for c in num_cols if not any(x in c for x in exclude)]
            if 'HSR Distance' not in rec_metrics and 'HSR Distance' in session_data.columns:
                rec_metrics.append('HSR Distance')
            def_pc_metrics = [m for m in ['Total Distance', 'Meterage Per Minute', 'Maximum Velocity', 'HSR Distance', 'Total Player Load'] if m in session_data.columns]
            pc_metrics = st.multiselect("지표 선택", options=rec_metrics, default=def_pc_metrics, key="pc_metrics_sel")

        if pc_metrics and 'Player Name' in session_data.columns:
            max_map = {m: (session_data[m].max() if m in session_data.columns and session_data[m].max() > 0 else 1) for m in pc_metrics}
            fig_pc = go.Figure()
            palette = px.colors.qualitative.Plotly

            for idx, (_, row) in enumerate(session_data.iterrows()):
                player = row['Player Name']
                color = palette[idx % len(palette)] if player in pc_players else 'lightgray'
                width = 3 if player in pc_players else 1
                opacity = 1.0 if player in pc_players else 0.25
                y_vals = [(row[m] / max_map[m] * 100) if m in row else 0 for m in pc_metrics]
                fig_pc.add_trace(go.Scatter(
                    x=pc_metrics,
                    y=y_vals,
                    mode='markers',
                    name=str(player),
                    marker=dict(color=color, size=10, line=dict(color=color, width=0)),
                    opacity=opacity,
                    hovertemplate="<b>%{text}</b><br>%{x}: %{customdata:.1f} (score %{y:.0f})<extra></extra>",
                    text=[player] * len(pc_metrics),
                    customdata=[row[m] if m in row else 0 for m in pc_metrics]
                ))

            fig_pc.update_layout(
                yaxis_title="Score (0-100, 팀 내 최대값 기준)",
                xaxis_title="지표",
                yaxis=dict(range=[0, 110]),
                legend_title_text="선수",
                height=520
            )
            st.plotly_chart(fig_pc, use_container_width=True)
        else:
            st.info("비교할 선수/지표를 선택하세요.")

st.title("🏑 Field Hockey Match Report & GPS Analysis")

st.markdown("### 📂 데이터 파일 업로드")
upload_container = st.container()
with upload_container:
    col_old, col_new = st.columns(2)
    with col_old:
        st.info("기존 SportsCode CSV (Legacy)")
        file_old = st.file_uploader("SportsCode CSV 파일 업로드", type=['csv'], key="sc_uploader")
    with col_new:
        st.info("📑 **버전 B: 신규 SportsCode (CSV / XML)**")
        st.markdown("- 태그가 많은 CSV 또는 **XML(강력 추천)**")
        file_new = st.file_uploader("신규 형식 파일 업로드", type=['csv', 'xml'], key="uploader_new")
st.divider()


tab_summary = st.container()
tab1 = st.container()
tab2 = st.container()
tab3 = st.container()

raw_df = None
if file_new is not None:
    if file_new.name.lower().endswith('.xml'):
        raw_df = process_xml_data(file_new)
        st.toast(f"✅ XML 포맷 분석 성공! (가장 정확함): {file_new.name}")
    else:
        raw_df = process_data(file_new)
        st.toast(f"✅ 신규 CSV 포맷 분석 시작: {file_new.name}")
elif file_old is not None:
    raw_df = process_data(file_old)
    st.toast(f"✅ 기존 포맷 분석 시작: {file_old.name}")

if raw_df is not None:
    # 1. 데이터 및 전처리 (raw_df는 업로드 블록에서 생성됨)
    
    # 2. 지표 계산
    metrics_df = calculate_metrics(raw_df)
    quarter_summary = build_quarter_summary(raw_df)

    def _normalize_columns(df: pd.DataFrame):
        if df.empty:
            return df
        rename_map = {}
        for c in df.columns:
            if 'Circle Entries (CE)' in c:
                rename_map[c] = 'Circle Entries'
            if 'SPP (' in c and '초/회' not in c:
                rename_map[c] = 'SPP (초/회)'
            if 'Allowed SPP' in c and '초/회' not in c:
                rename_map[c] = 'Allowed SPP (초/회)'
        df = df.rename(columns=rename_map)
        for col in ['Circle Entries', '25y Entries', 'Sec per CE', 'Press Attempts', 'Press Success',
                    'SPP (초/회)', 'Allowed SPP (초/회)', 'Build25 Ratio (%)']:
            if col not in df.columns:
                df[col] = 0
        return df

    metrics_df = _normalize_columns(metrics_df)
    quarter_summary = _normalize_columns(quarter_summary)

    if 'Quarter' in quarter_summary.columns:
        quarter_summary['Quarter'] = quarter_summary['Quarter'].apply(_normalize_quarter_value)
        # 유효한 Q1~Q4만 유지
        quarter_summary = quarter_summary[quarter_summary['Quarter'].isin([f"Q{i}" for i in range(1, 5)])].reset_index(drop=True)
        # 쿼터/팀 중복 레코드 제거 (Q1~Q4만 1회씩 보이도록)
        quarter_summary = quarter_summary.drop_duplicates(subset=['Quarter', 'Team'], keep='first')

    if not quarter_summary.empty:
        rename_map = {}
        for col in quarter_summary.columns:
            if 'SPP (' in col and '초/회' not in col:
                rename_map[col] = 'SPP (초/회)'
            if 'Allowed SPP' in col and '초/회' not in col:
                rename_map[col] = 'Allowed SPP (초/회)'
        quarter_summary = quarter_summary.rename(columns=rename_map)
        for col in ['25y Entries', 'Sec per CE', 'Press Attempts', 'Press Success', 'SPP (초/회)', 'Allowed SPP (초/회)']:
            if col not in quarter_summary.columns:
                quarter_summary[col] = 0

    # 3. 시각화 (탭 구성) + 경기 요약 탭
    with tab_summary:
        st.markdown("## 경기 요약")
        # 구간 선택
        period_options = [
            ("경기 전체", None),
            ("Q1", ["Q1"]),
            ("Q2", ["Q2"]),
            ("Q3", ["Q3"]),
            ("Q4", ["Q4"]),
            ("전반 (Q1+Q2)", ["Q1", "Q2"]),
            ("후반 (Q3+Q4)", ["Q3", "Q4"]),
        ]
        period_label, sel_quarters = st.selectbox("구간 선택", period_options, format_func=lambda x: x[0], key="summary_period_final")

        def aggregate_period(sel_quarters):
            if sel_quarters is None:
                return metrics_df.copy()
            if quarter_summary.empty:
                return pd.DataFrame()
            qdf = quarter_summary.copy()
            def _normalize_q(text):
                s = str(text)
                label = _extract_quarter_label(s)
                if label:
                    return label
                digit = re.search(r'([1-4])', s)
                return f"Q{digit.group(1)}" if digit else s.strip()
            qdf['QuarterNorm'] = qdf['Quarter'].apply(_normalize_q)
            qdf = qdf[qdf['QuarterNorm'].isin(sel_quarters)]
            if qdf.empty:
                return pd.DataFrame()
            g = qdf.groupby('Team')
            base_cols = ['Circle Entries', 'Shots', 'PCs', 'Goals', '25y Entries', 'Duration (s)',
                         'Press Attempts', 'Press Success', 'Build25 Attempts', 'Build25 Success']
            agg_cols = [c for c in base_cols if c in qdf.columns]
            data = g[agg_cols].sum(min_count=1).reset_index()
            # 누락 컬럼 기본값 0
            for c in base_cols:
                if c not in data.columns:
                    data[c] = 0
            data = data.fillna(0)
            data['Field Goals'] = (data['Goals'] - data['PCs']).clip(lower=0)
            data['PC Goals'] = 0
            # 쿼터 집계 시 Goals 컬럼을 Goals (Total)로도 노출해 득점이 '-'로 표시되지 않도록 처리
            if 'Goals' in data.columns and 'Goals (Total)' not in data.columns:
                data['Goals (Total)'] = data['Goals']
            data['Build25 Ratio (%)'] = (data['Build25 Success'] / data['Build25 Attempts'] * 100).replace([np.inf, np.nan], 0).round(1)
            w_pos = []; w_att = []; w_spp = []; w_allowed = []; w_secce = []
            for team, gg in g:
                dur = gg['Duration (s)'].sum() if 'Duration (s)' in gg.columns else 0
                if dur > 0:
                    w_pos.append((team, (gg['Possession (%)'] * gg['Duration (s)']).sum() / dur if 'Possession (%)' in gg.columns else 0))
                    w_att.append((team, (gg['ATT Possession (%)'] * gg['Duration (s)']).sum() / dur if 'ATT Possession (%)' in gg.columns else 0))
                    w_spp.append((team, (gg['SPP (초/회)'] * gg['Duration (s)']).sum() / dur if 'SPP (초/회)' in gg.columns else 0))
                    w_allowed.append((team, (gg['Allowed SPP (초/회)'] * gg['Duration (s)']).sum() / dur if 'Allowed SPP (초/회)' in gg.columns else 0))
                    w_secce.append((team, (gg['Sec per CE'] * gg['Duration (s)']).sum() / dur if 'Sec per CE' in gg.columns else 0))
                else:
                    w_pos.append((team, 0)); w_att.append((team, 0)); w_spp.append((team, 0)); w_allowed.append((team, 0)); w_secce.append((team, 0))
            w_pos = dict(w_pos); w_att = dict(w_att); w_spp = dict(w_spp); w_allowed = dict(w_allowed); w_secce = dict(w_secce)
            data['Possession (%)'] = data['Team'].map(w_pos).fillna(0).round(1)
            data['ATT Possession (%)'] = data['Team'].map(w_att).fillna(0).round(1)
            data['SPP (초/회)'] = data['Team'].map(w_spp).fillna(0).round(2)
            data['Allowed SPP (초/회)'] = data['Team'].map(w_allowed).fillna(0).round(2)
            data['Sec per CE'] = data['Team'].map(w_secce).fillna(0).round(2)
            return data

        summary_df = aggregate_period(sel_quarters)
        if summary_df.empty:
            st.info("선택한 구간에 대한 통계를 표시할 수 없습니다.")
            st.stop()
        summary_df = summary_df.fillna(0)

        # 팀 선택
        team_options = summary_df['Team'].tolist()
        left_team = st.selectbox("왼쪽 팀", team_options, index=0, key="summary_left_team_final")
        right_candidates = [t for t in team_options if t != left_team] or team_options
        right_team = st.selectbox("오른쪽 팀", right_candidates, index=0, key="summary_right_team_final")

        mdict = summary_df.set_index('Team')

        def val(team, key, decimals=False, suffix=""):
            if team not in mdict.index or key not in mdict.columns:
                return "-"
            try:
                num = float(mdict.at[team, key])
                if pd.isna(num):
                    return "-"
                if decimals:
                    return f"{num:.1f}{suffix}"
                return f"{num:.0f}{suffix}"
            except Exception:
                return str(mdict.at[team, key])

        # 스타일 + 프린트 색 유지
        summary_styles = """
        <style>
        .summary-card {background:#fff;border-radius:18px;padding:18px 22px;box-shadow:0 6px 18px rgba(0,0,0,0.04);margin-bottom:14px;}
        .summary-title {text-align:center;font-weight:700;font-size:20px;margin-bottom:14px;color:#111;}
        .summary-row {display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f1f1f5;}
        .summary-row:last-child {border-bottom:none;}
        .label {flex:1;text-align:center;font-weight:600;color:#111;}
        .pill {min-width:44px;display:inline-flex;align-items:center;justify-content:center;padding:6px 12px;border-radius:14px;font-weight:700;font-size:14px;}
        .pill-left {background:#0f66c2;color:#fff;}
        .pill-right {background:#d62728;color:#fff;}
        @media print {
            * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
        }
        </style>
        """

        def render_section(title, rows, height=240):
            html = [summary_styles, f'<div class="summary-card"><div class="summary-title">{title}</div>']
            for l, lbl, r in rows:
                html.append(
                    f'<div class="summary-row">'
                    f'<div class="pill pill-left">{l}</div>'
                    f'<div class="label">{lbl}</div>'
                    f'<div class="pill pill-right">{r}</div>'
                    f'</div>'
                )
            html.append('</div>')
            st.components.v1.html("\n".join(html), height=height, scrolling=False)

        # 점유율 바 + 파이
        posA = float(mdict.at[left_team, 'Possession (%)']) if left_team in mdict.index and 'Possession (%)' in mdict.columns else 0.0
        posB = float(mdict.at[right_team, 'Possession (%)']) if right_team in mdict.index and 'Possession (%)' in mdict.columns else 0.0
        attA = float(mdict.at[left_team, 'ATT Possession (%)']) if left_team in mdict.index and 'ATT Possession (%)' in mdict.columns else 0.0
        attB = float(mdict.at[right_team, 'ATT Possession (%)']) if right_team in mdict.index and 'ATT Possession (%)' in mdict.columns else 0.0

        def pie(values, labels, title):
            fig = go.Figure(go.Pie(
                values=values, labels=labels, hole=0.55,
                marker=dict(colors=['#0f66c2', '#d62728']),
                textinfo='label+percent', sort=False, direction='clockwise', rotation=180,
                textfont=dict(color="#fff")
            ))
            fig.update_layout(title=title, showlegend=False, margin=dict(l=10,r=10,t=40,b=10), height=260)
            return fig

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(pie([posA, posB],[left_team,right_team],"전체 점유율"), use_container_width=True)
        with c2:
            st.plotly_chart(pie([attA, attB],[left_team,right_team],"공격 점유율"), use_container_width=True)

        render_section("공격 지표", [
            (val(left_team,'Goals (Total)'), "득점", val(right_team,'Goals (Total)')),
            (val(left_team,'Shots'), "슈팅", val(right_team,'Shots')),
            (val(left_team,'PCs'), "페널티코너", val(right_team,'PCs')),
            (val(left_team,'Circle Entries'), "서클 진입", val(right_team,'Circle Entries')),
            (val(left_team,'Field Goals'), "필드골", val(right_team,'Field Goals')),
            (val(left_team,'PC Goals'), "PC 골", val(right_team,'PC Goals')),
        ], height=320)
        render_section("빌드업 & 효율", [
            (val(left_team,'25y Entries'), "A25 진입", val(right_team,'25y Entries')),
            (val(left_team,'Build25 Ratio (%)', decimals=True, suffix="%"), "Build25 성공률", val(right_team,'Build25 Ratio (%)', decimals=True, suffix="%")),
            (val(left_team,'Sec per CE', decimals=True), "Sec/CE", val(right_team,'Sec per CE', decimals=True)),
        ], height=260)
        render_section("압박 지표", [
            (val(left_team,'Press Attempts'), "프레스 시도", val(right_team,'Press Attempts')),
            (val(left_team,'Press Success'), "프레스 성공", val(right_team,'Press Success')),
            (val(left_team,'SPP (초/회)', decimals=True), "SPP", val(right_team,'SPP (초/회)', decimals=True)),
            (val(left_team,'Allowed SPP (초/회)', decimals=True), "Allowed SPP", val(right_team,'Allowed SPP (초/회)', decimals=True)),
        ], height=320)
    
    colors = {'Team A': '#1f77b4', 'Team B': '#ff7f0e'} # 기본 색상
    
    with tab1:
        st.markdown("## 공격 효율 (Attack)")
        col1, col2 = st.columns(2)
        with col1:
            # 슈팅 및 골 차트
            fig_atk = px.bar(metrics_df, x='Team', y=['Field Goals', 'PC Goals', 'Shots'], 
                             title="득점 및 슈팅 (Goals & Shots)", barmode='group')
            st.plotly_chart(fig_atk, use_container_width=True)
        with col2:
            # 25y 진입 대비 서클 진입 효율 (쿼터별 오버랩 막대 + 전환율 라인)
            if quarter_summary.empty or 'Quarter' not in quarter_summary.columns:
                st.info("쿼터 정보가 없어 25y→서클 진입 효율을 표시할 수 없습니다.")
            else:
                ce_col = 'Circle Entries' if 'Circle Entries' in quarter_summary.columns else 'Circle Entries (CE)'
                if ce_col not in quarter_summary.columns:
                    st.info("서클 진입 수치가 없습니다.")
                else:
                    team_opts = metrics_df['Team'].tolist()
                    sel_team = st.selectbox("팀 선택 (25y→서클 진입 효율)", team_opts, key="ce_eff_team")

                    # raw_df에서 직접 쿼터별 25y/CE 재집계 (라벨이 섞여 있어도 Q1~Q4로 정규화)
                    if 'Quarter' in raw_df.columns:
                        q_col = 'Quarter'
                    elif 'Ungrouped' in raw_df.columns:
                        q_col = 'Ungrouped'
                    else:
                        q_col = None

                    if q_col is None:
                        st.info("쿼터 정보를 찾을 수 없습니다.")
                    else:
                        df_team = raw_df[raw_df['Team'] == sel_team].copy()
                        df_team['QuarterNorm'] = df_team[q_col].apply(_normalize_quarter_value)
                        df_team = df_team[df_team['QuarterNorm'].isin([f"Q{i}" for i in range(1, 5)])]

                        ce_mask_raw = df_team['Row'].str.contains(
                            r'서클\s*진입|슈팅\s*서클|circle\s*entry|attack\s*circle|att\s*circle',
                            case=False, regex=True, na=False
                        )
                        ce_counts = df_team[ce_mask_raw].groupby('QuarterNorm').size()
                        entry25_counts = df_team[df_team['Row'].str.contains('A25 START', na=False)].groupby('QuarterNorm').size()

                        q_order = [f"Q{i}" for i in range(1, 5)]
                        qdf = pd.DataFrame({'Quarter': q_order})
                        qdf['25y Entries'] = qdf['Quarter'].map(entry25_counts).fillna(0).astype(int)
                        qdf['Circle Entries'] = qdf['Quarter'].map(ce_counts).fillna(0).astype(int)
                        qdf['CE/25y (%)'] = qdf.apply(lambda r: (r['Circle Entries'] / r['25y Entries'] * 100) if r['25y Entries'] else 0, axis=1)

                        # y축 정규화: 모든 팀/쿼터 기준 최대값으로 범위 통일
                        ce_counts_all = raw_df[raw_df[q_col].notna()].copy()
                        ce_counts_all['QuarterNorm'] = ce_counts_all[q_col].apply(_normalize_quarter_value)
                        ce_counts_all = ce_counts_all[ce_counts_all['QuarterNorm'].isin(q_order)]
                        ce_all = ce_counts_all[ce_counts_all['Row'].str.contains(
                            r'서클\s*진입|슈팅\s*서클|circle\s*entry|attack\s*circle|att\s*circle',
                            case=False, regex=True, na=False
                        )].groupby(['Team', 'QuarterNorm']).size()
                        entry25_all = ce_counts_all[ce_counts_all['Row'].str.contains('A25 START', na=False)].groupby(['Team', 'QuarterNorm']).size()
                        all_count_max = max(ce_all.max() if not ce_all.empty else 0, entry25_all.max() if not entry25_all.empty else 0)
                        all_rate_max = 0
                        if not entry25_all.empty:
                            # 전환율 최대
                            rates = []
                            for (t, q), ce_v in ce_all.items():
                                ent_v = entry25_all.get((t, q), 0)
                                if ent_v:
                                    rates.append(ce_v / ent_v * 100)
                            all_rate_max = max(rates) if rates else 0

                        fig_eff = go.Figure()
                        fig_eff.add_bar(
                            name="25y Entries", x=qdf['Quarter'], y=qdf['25y Entries'],
                            marker_color="#94bdf2", opacity=0.65,
                            text=qdf['25y Entries'], textposition="inside",
                            texttemplate="<b>%{text}</b>",
                            textfont=dict(size=15)
                        )
                        fig_eff.add_bar(
                            name="Circle Entries", x=qdf['Quarter'], y=qdf['Circle Entries'],
                            marker_color="#1f77b4", opacity=0.9,
                            text=qdf['Circle Entries'], textposition="inside",
                            texttemplate="<b>%{text}</b>",
                            textfont=dict(size=15)
                        )
                        fig_eff.add_trace(go.Scatter(
                            name="CE/25y (%)", x=qdf['Quarter'], y=qdf['CE/25y (%)'],
                            mode="lines+markers", yaxis="y2", line=dict(color="#d62728", width=3), marker=dict(size=8)
                        ))
                        fig_eff.update_layout(
                            title="25y 진입 대비 서클 진입 효율 (쿼터별)",
                            barmode="overlay",
                            yaxis=dict(title="횟수", rangemode="tozero",
                                       range=[0, all_count_max * 1.1 if all_count_max else None]),
                            yaxis2=dict(title="전환율 (%)", overlaying="y", side="right", rangemode="tozero",
                                        range=[0, all_rate_max * 1.1 if all_rate_max else None]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=420,
                            margin=dict(l=10, r=10, t=50, b=10)
                        )
                        st.plotly_chart(fig_eff, use_container_width=True)
        st.markdown("#### 쿼터별 공격 위협 (슈팅 + PC)")
        if quarter_summary.empty:
            st.info("쿼터 정보가 없어 공격 위협 추세를 표시할 수 없습니다.")
        else:
            attack_team = st.selectbox("공격 위협 그래프를 볼 팀을 선택하세요", quarter_summary['Team'].unique().tolist(), key="tab_attack_team")
            opponent_candidates = [t for t in quarter_summary['Team'].unique() if t != attack_team]
            if not opponent_candidates:
                st.info("상대 팀 정보가 부족해 공격 위협 비교를 표시할 수 없습니다.")
            else:
                attack_opponent = st.selectbox("비교할 상대 팀을 선택하세요", opponent_candidates, key="tab_attack_opponent")
                attack_df = quarter_summary.copy()
                attack_df['Attack Threat'] = attack_df['Shots'].fillna(0) + attack_df['PCs'].fillna(0)

                team_series = attack_df[attack_df['Team'] == attack_team].set_index('Quarter')['Attack Threat']
                opp_series = attack_df[attack_df['Team'] == attack_opponent].set_index('Quarter')['Attack Threat']
                quarters_all = sorted(set(team_series.index).union(opp_series.index), key=_quarter_sort_key)

                if not quarters_all:
                    st.info("선택한 팀의 쿼터별 공격 위협 데이터를 찾을 수 없습니다.")
                else:
                    def _safe_val(series, q):
                        if series.empty:
                            return 0.0
                        return float(series[series.index == q].sum())

                    team_values = [_safe_val(team_series, q) for q in quarters_all]
                    opp_values = [_safe_val(opp_series, q) for q in quarters_all]
                    x_numeric = list(range(len(quarters_all)))
                    dense_x, dense_team, dense_opp = _insert_cross_points(x_numeric, team_values, opp_values)

                    team_color = 'rgba(214, 39, 40, 1)'
                    team_fill = 'rgba(214, 39, 40, 0.18)'
                    opp_color = 'rgba(31, 119, 180, 1)'
                    opp_fill = 'rgba(31, 119, 180, 0.18)'

                    fig_attack = go.Figure()
                    _apply_dynamic_shading(fig_attack, dense_x, dense_team, dense_opp, team_fill, opp_fill, prefer_lower=False)
                    fig_attack.add_trace(go.Scatter(
                        x=x_numeric, y=team_values, mode='lines+markers',
                        name=f"{attack_team} 공격 위협(슈팅+PC)",
                        line=dict(color=team_color, width=3), marker=dict(size=8)
                    ))
                    fig_attack.add_trace(go.Scatter(
                        x=x_numeric, y=opp_values, mode='lines+markers',
                        name=f"{attack_opponent} 공격 위협(슈팅+PC)",
                        line=dict(color=opp_color, width=3), marker=dict(size=8)
                    ))
                    fig_attack.update_layout(
                        xaxis=dict(
                            title="쿼터",
                            tickmode='array',
                            tickvals=x_numeric,
                            ticktext=quarters_all
                        ),
                        yaxis_title="횟수 (슈팅+PC)",
                        title=f"{attack_team} vs {attack_opponent} — 쿼터별 공격 위협",
                        hovermode="x unified",
                        height=450
                    )
                    st.plotly_chart(fig_attack, use_container_width=True)

        st.markdown("#### 시간대별 슈팅+PC 누적 그래프")
        events_df = build_shot_pc_events(raw_df)
        if events_df.empty or events_df['Team'].nunique() < 2:
            st.info("슈팅 또는 PC 이벤트가 충분하지 않아 누적 그래프를 표시할 수 없습니다.")
        else:
            team_options = events_df['Team'].unique().tolist()
            # 요청에 따라 시간대별 누적 슈팅+PC 그래프 제거
        # 서클 진입 위치 & 성과 분석 도식
        render_entry_analysis(raw_df)

    with tab2:
        st.markdown("## 빌드업 & 압박 (Build & Press)")
        col3, col4 = st.columns(2)
        with col3:
            # 압박 지표 시각화
            # Press Index가 낮을수록 좋음? 아니면 높을수록 좋음?
            # 수식: Build Time / Press Events. 높을수록 빌드업을 오래 버텼다는 뜻 (압박 덜 당함 or 탈압박 잘함)
            fig_press = go.Figure()
            fig_press.add_trace(go.Bar(x=metrics_df['Team'], y=metrics_df['Allowed SPP (초/회)'], name='가한 압박 수치(초/회'))
            fig_press.add_trace(go.Bar(x=metrics_df['Team'], y=metrics_df['SPP (초/회)'], name='당한 압박 수치(초/회'))
            fig_press.update_layout(title="압박 지표 (SPP)", barmode='group')
            st.plotly_chart(fig_press, use_container_width=True)
            st.info("**SPP**(Seconds Per Press): 빌드업 시간 ÷ 당한 압박 이벤트. 값이 낮을수록 상대 압박에 빠르게 대응했다는 뜻입니다.")
            
        with col4:
            # Build25 성공률
            fig_build = px.bar(metrics_df, x='Team', y='Build25 Ratio (%)', 
                               title="Build25 성공률 (Defense Start -> 25y Entry)", 
                               text='Build25 Ratio (%)', color='Team')
            st.plotly_chart(fig_build, use_container_width=True)

        st.markdown("#### 쿼터별 SPP vs Allowed SPP")
        if quarter_summary.empty:
            st.info("쿼터 정보가 없어 SPP 추이를 표시할 수 없습니다.")
        else:
            spp_team_options = quarter_summary['Team'].unique().tolist()
            selected_team = st.selectbox("SPP 그래프를 볼 팀을 선택하세요", spp_team_options, key="tab_spp_team")
            team_quarter_df = quarter_summary[quarter_summary['Team'] == selected_team].copy()
            for c in ['SPP (초/회)', 'Allowed SPP (초/회)']:
                if c not in team_quarter_df.columns:
                    team_quarter_df[c] = 0
            if team_quarter_df.empty:
                st.info("선택한 팀의 쿼터 데이터가 충분하지 않습니다.")
            else:
                team_quarter_df['QuarterSort'] = team_quarter_df['Quarter'].apply(_quarter_sort_key)
                team_quarter_df = team_quarter_df.sort_values('QuarterSort')
                quarters = team_quarter_df['Quarter'].tolist()
                spp_values = team_quarter_df['SPP (초/회)'].astype(float).tolist()
                allowed_values = team_quarter_df['Allowed SPP (초/회)'].astype(float).tolist()

                x_numeric = list(range(len(quarters)))
                dense_x, dense_allowed, dense_spp = _insert_cross_points(x_numeric, allowed_values, spp_values)

                press_color = 'rgba(214, 39, 40, 1)'
                press_fill = 'rgba(214, 39, 40, 0.18)'
                allowed_color = 'rgba(31, 119, 180, 1)'
                allowed_fill = 'rgba(31, 119, 180, 0.18)'

                fig_spp = go.Figure()
                _apply_dynamic_shading(fig_spp, dense_x, dense_allowed, dense_spp, allowed_fill, press_fill, prefer_lower=True)

                fig_spp.add_trace(go.Scatter(
                    x=x_numeric, y=allowed_values, mode='lines+markers', name='가한 압박 수치(초/회',
                    line=dict(color=allowed_color, width=3), marker=dict(size=8)
                ))
                fig_spp.add_trace(go.Scatter(
                    x=x_numeric, y=spp_values, mode='lines+markers', name='당한 압박 수치(초/회',
                    line=dict(color=press_color, width=3), marker=dict(size=8)
                ))

                fig_spp.update_layout(
                    xaxis=dict(
                        title="쿼터",
                        tickmode='array',
                        tickvals=x_numeric,
                        ticktext=quarters
                    ),
                    yaxis_title="압박 수치 (초/이벤트) — 낮을수록 우위(역축)",
                    title=f"{selected_team} — 쿼터별 압박지수(SPP) vs 탈압박지수(초/회",
                    hovermode="x unified",
                    height=450
                )
                fig_spp.update_yaxes(autorange="reversed")

                st.plotly_chart(fig_spp, use_container_width=True)

        st.markdown("#### 압박 지도 (빌드업 시 당한 압박 위치)")
        # 압박 시도 정의 (SPP와 동일): 우리팀 턴오버(75/100) + 우리팀 파울(75/100) + 상대 파울(25/50)
        def _press_attempt_events(df_in: pd.DataFrame, team: str):
            df = df_in.copy()
            lane_mask_high = df['지역'].astype(str).str.contains(r'75|100', na=False)
            lane_mask_low = df['지역'].astype(str).str.contains(r'25|50', na=False)
            # 우리 팀 이벤트
            ours = df[(df['Team'] == team) & lane_mask_high &
                      df['Row'].str.contains(r'턴오버|파울', case=False, na=False, regex=True)]
            # 상대 파울 (우리 진영 25/50)
            opps = df[(df['Team'] != team) & lane_mask_low &
                      df['Row'].str.contains(r'파울', case=False, na=False, regex=True)]
            combined = pd.concat([ours, opps], ignore_index=True)
            return combined

        show_attacking_half = st.checkbox("공격 진영 보기", value=False, key="press_heat_attacking_half")
        view_half = "attacking" if show_attacking_half else "defensive"

        team_opts = metrics_df['Team'].tolist()
        sel_pair = st.multiselect("왼쪽/오른쪽에 배치할 두 팀 선택", team_opts, default=team_opts[:2], key="press_heat_pair")
        if len(sel_pair) != 2:
            st.info("두 팀을 선택하면 압박 지도를 표시합니다.")
        else:
            left_team, right_team = sel_pair
            def _build_counts(team):
                ev = press_attempt_events_for_team(raw_df, team)
                return build_zone_counts(ev)
            counts_left = _build_counts(left_team)
            counts_right = _build_counts(right_team)

            global_max = max(counts_left.values.max(), counts_right.values.max(), 1)
            fig_left = build_turnover_field_figure(counts_left, max_ref=global_max, show_scale=True, view_half=view_half)
            fig_left.update_layout(title=f"{left_team} 압박 지도")
            fig_right = build_turnover_field_figure(counts_right, max_ref=global_max, show_scale=False, view_half=view_half)
            fig_right.update_layout(title=f"{right_team} 압박 지도")

            col_l, col_r = st.columns(2)
            col_l.plotly_chart(fig_left, use_container_width=False)
            col_r.plotly_chart(fig_right, use_container_width=False)

    with tab3:
        st.markdown("## 점유율 (Possession)")
        teams_main = order_teams(metrics_df['Team'].tolist())
        qdf = quarter_summary[quarter_summary['Team'].isin(teams_main)] if not quarter_summary.empty else pd.DataFrame()
        # 점유율 파이 차트는 팀별 색상을 반드시 다르게 고정
        pie_palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#17becf", "#9467bd"]
        team_color_map = {team: pie_palette[i % len(pie_palette)] for i, team in enumerate(teams_main)}

        st.markdown("#### 쿼터별 전체 점유율 (파이)")
        if qdf.empty:
            st.info("쿼터 정보가 없어 쿼터별 점유율을 표시할 수 없습니다.")
        else:
            quarter_order = sorted(qdf['Quarter'].unique().tolist(), key=_quarter_sort_key)
            cols = st.columns(min(4, len(quarter_order)))
            for idx, q in enumerate(quarter_order):
                q_slice = qdf[qdf['Quarter'] == q]
                # 누락 팀을 0으로 채워 고아 레이블 제거
                data_map = {team: float(q_slice[q_slice['Team'] == team]['Possession (%)'].iloc[0]) if not q_slice[q_slice['Team'] == team].empty else 0.0 for team in teams_main}
                labels = list(data_map.keys())
                values = list(data_map.values())
                colors = [team_color_map.get(lbl, "#1f77b4") for lbl in labels]
                fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, marker=dict(colors=colors))])
                fig_pie.update_layout(title=f"{q} 전체 점유율", showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
                fig_pie.update_traces(textinfo='label+percent')
                with cols[idx % len(cols)]:
                    st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 쿼터별 공격 점유율 (파이)")
        if qdf.empty or 'ATT Possession (%)' not in qdf.columns:
            st.info("쿼터 정보가 없어 쿼터별 공격 점유율을 표시할 수 없습니다.")
        else:
            quarter_order = sorted(qdf['Quarter'].unique().tolist(), key=_quarter_sort_key)
            cols_att = st.columns(min(4, len(quarter_order)))
            for idx, q in enumerate(quarter_order):
                q_slice = qdf[qdf['Quarter'] == q]
                data_map = {
                    team: float(q_slice[q_slice['Team'] == team]['ATT Possession (%)'].iloc[0])
                    if not q_slice[q_slice['Team'] == team].empty else 0.0
                    for team in teams_main
                }
                labels_att = list(data_map.keys())
                values_att = list(data_map.values())
                colors_att = [team_color_map.get(lbl, "#1f77b4") for lbl in labels_att]
                fig_pie_att = go.Figure(data=[go.Pie(labels=labels_att, values=values_att, hole=0.4, marker=dict(colors=colors_att))])
                fig_pie_att.update_layout(title=f"{q} 공격 점유율", showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
                fig_pie_att.update_traces(textinfo='label+percent')
                with cols_att[idx % len(cols_att)]:
                    st.plotly_chart(fig_pie_att, use_container_width=True)

else:
    st.info("CSV ?? XML ?? ??? ???? ???.")


