"""
MACD分时策略 v4.1 - 量价MACD三维背离完整版
========================================
基于"成交量+MACD红绿柱+股价"三者关系与背离交易规则

核心逻辑：
1. 三者正常协同关系（健康走势）
2. 各种背离组合与买卖信号
3. 信号有效性判断（零轴过滤、次数过滤）
4. 确认机制（金叉死叉、均线）

背离类型：
- 经典三重顶背离：股价新高 + 成交量没新高 + MACD没新高
- 量价顶背离：股价新高 + 成交量天量 + MACD没新高（放量滞涨）
- 无量顶背离：股价新高 + MACD新高 + 成交量没新高
- 经典三重底背离：股价新低 + 成交量没新低(放大) + MACD没新低
- 地量底背离：股价新低 + 成交量地量 + MACD没新低
- 有量底背离：股价新低 + MACD新低 + 成交量没新低

过滤机制：
- 零轴位置过滤
- 背离次数过滤（连续2-3次更强）
- 确认机制（均线、金叉死叉）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"


class SignalLevel(Enum):
    """信号强度"""
    STRONG = "强信号"      # 三重背离 + 零轴确认 + 顺势
    NORMAL = "普通信号"    # 单一背离
    WEAK = "弱信号"        # 需要确认


@dataclass
class DivergenceRecord:
    """背离记录"""
    time: pd.Timestamp
    signal_type: SignalType  # 背离发生时预期的方向
    divergence_type: str     # '顶背离'/'底背离'
    sub_type: str          # '三重'/'量价'/'无量'/'地量'/'有量'/'单纯'
    price_level: float     # 背离时的价格位置
    hist_level: float      # 背离时的MACD位置
    volume_level: float    # 背离时的成交量位置
    index: int


@dataclass
class TradeSignal:
    """交易信号"""
    time: pd.Timestamp
    signal_type: SignalType
    price: float
    intraday_avg_price: float
    avg_price_bias_pct: float
    avg_line_position: str
    reason: str
    signal_level: SignalLevel
    divergence_type: str
    sub_type: str
    confirmed: bool        # 是否经过确认
    confirm_type: str     # '金叉'/'死叉'/'均线'/'无'
    divergence_count: int  # 连续背离次数
    zero_axis: str        # 'above'/'below'/'crossing'
    trend_following: bool
    index: int


@dataclass
class MacdParams:
    """MACD参数"""
    short_period: int = 5
    long_period: int = 34
    dea_period: int = 5
    divergence_window: int = 20
    signal_interval: int = 600  # 信号间隔10分钟
    # 背离阈值
    price_threshold: float = 0.999
    volume_threshold: float = 0.9
    hist_threshold: float = 0.9
    # 次数过滤
    min_divergence_count: int = 1   # 最少背离次数
    # 确认参数
    ma_short: int = 5
    ma_medium: int = 20
    # 分时均价线过滤
    intraday_avg_tolerance_pct: float = 0.0005
    min_avg_price_bias_pct: float = 0.003
    same_signal_interval: int = 1800
    allow_weak_signals: bool = False


# ==================== 数据加载 ====================

def load_and_aggregate(csv_path: str, agg_seconds: int = 60, target_date: str = None) -> pd.DataFrame:
    """加载并聚合逐笔数据"""
    df = pd.read_csv(csv_path)
    df['时间'] = pd.to_datetime(df['时间'])
    df['成交额'] = df['价格'] * df['成交量']

    if target_date:
        year = target_date[:4]
        month = target_date[4:6]
        day = target_date[6:8]
        target_date_str = f"{year}-{month}-{day}"
        df = df[df['时间'].dt.strftime('%Y-%m-%d') == target_date_str]
        if len(df) == 0:
            raise ValueError(f"CSV中没有 {target_date} 的数据")

    if agg_seconds == 1:
        df.rename(columns={'价格': '收盘价'}, inplace=True)
        df = df[['时间', '成交量', '成交额', '收盘价']].copy()
        df['开盘价'] = df['收盘价']
        df['最高价'] = df['收盘价']
        df['最低价'] = df['收盘价']
        df['成交量均线'] = df['成交量'].rolling(window=100, min_periods=1).mean()
        cumulative_volume = df['成交量'].cumsum()
        df['分时均价'] = np.where(cumulative_volume > 0, df['成交额'].cumsum() / cumulative_volume, df['收盘价'])
        return df

    df['周期'] = df['时间'].dt.floor(f'{agg_seconds}s')
    agg = df.groupby('周期').agg({
        '价格': ['first', 'last', 'max', 'min'],
        '成交量': 'sum',
        '成交额': 'sum'
    }).reset_index()
    agg.columns = ['时间', '开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额']
    agg = agg.sort_values('时间').reset_index(drop=True)
    cumulative_volume = agg['成交量'].cumsum()
    agg['分时均价'] = np.where(cumulative_volume > 0, agg['成交额'].cumsum() / cumulative_volume, agg['收盘价'])
    return agg


def calculate_macd(df: pd.DataFrame, params: MacdParams) -> pd.DataFrame:
    """计算MACD指标"""
    close = df['收盘价']
    df['EMA_Short'] = close.ewm(span=params.short_period, adjust=False).mean()
    df['EMA_Long'] = close.ewm(span=params.long_period, adjust=False).mean()
    df['DIFF'] = df['EMA_Short'] - df['EMA_Long']
    df['DEA'] = df['DIFF'].ewm(span=params.dea_period, adjust=False).mean()
    df['Histogram'] = (df['DIFF'] - df['DEA']) * 2

    # 均线
    df['MA5'] = close.rolling(window=params.ma_short, min_periods=1).mean()
    df['MA20'] = close.rolling(window=params.ma_medium, min_periods=1).mean()

    # 成交量均线
    df['成交量均线'] = df['成交量'].rolling(window=20, min_periods=1).mean()

    return df


# ==================== 核心背离检测 ====================

def get_window_extremes(df: pd.DataFrame, idx: int, window: int) -> Dict:
    """获取窗口内的极值"""
    start = max(0, idx - window)
    end = idx + 1
    window_data = df.iloc[start:end]

    price_max = window_data['收盘价'].max()
    price_min = window_data['收盘价'].min()
    volume_max = window_data['成交量'].max()
    volume_min = window_data['成交量'].min()
    hist_max = window_data['Histogram'].max()
    hist_min = window_data['Histogram'].min()

    return {
        'price_max': price_max, 'price_min': price_min,
        'volume_max': volume_max, 'volume_min': volume_min,
        'hist_max': hist_max, 'hist_min': hist_min
    }


def is_price_new_high(df: pd.DataFrame, idx: int, extremes: Dict, params: MacdParams, lookback: int = 10) -> bool:
    """判断价格是否创新高（可设定回溯范围）"""
    if idx < lookback:
        return False
    lookback_data = df.iloc[idx - lookback:idx]
    recent_max = lookback_data['收盘价'].max()
    curr_price = df['收盘价'].iloc[idx]
    return curr_price >= recent_max * params.price_threshold


def is_price_new_low(df: pd.DataFrame, idx: int, extremes: Dict, params: MacdParams, lookback: int = 10) -> bool:
    """判断价格是否创新低"""
    if idx < lookback:
        return False
    lookback_data = df.iloc[idx - lookback:idx]
    recent_min = lookback_data['收盘价'].min()
    curr_price = df['收盘价'].iloc[idx]
    return curr_price <= recent_min * (2 - params.price_threshold)


def get_histogram_peak_info(df: pd.DataFrame, idx: int, window: int) -> Dict:
    """获取MACD红绿柱的峰值信息（用于判断是否前大后小）"""
    if idx < window:
        return {'has_peak': False, 'peak_hist': 0, 'prev_peak_hist': 0}

    # 找到当前波峰的MACD最大值
    curr_window = df.iloc[idx - window:idx + 1]
    curr_positive = curr_window[curr_window['Histogram'] > 0]['Histogram']
    curr_peak = curr_positive.max() if len(curr_positive) > 0 else 0

    # 找到前一波峰
    prev_window_start = max(0, idx - 2 * window)
    prev_window_end = idx - window
    if prev_window_end <= prev_window_start:
        return {'has_peak': False, 'peak_hist': curr_peak, 'prev_peak_hist': 0}

    prev_window = df.iloc[prev_window_start:prev_window_end]
    prev_positive = prev_window[prev_window['Histogram'] > 0]['Histogram']
    prev_peak = prev_positive.max() if len(prev_positive) > 0 else 0

    return {
        'has_peak': True,
        'peak_hist': curr_peak,
        'prev_peak_hist': prev_peak
    }


def detect_divergence_type(
    df: pd.DataFrame,
    idx: int,
    params: MacdParams,
    extremes: Dict
) -> Tuple[bool, str, str, Dict]:
    """
    检测背离类型

    Returns:
        (是否背离, 背离方向, 背离子类型, 详细信息)
    """
    if idx < params.divergence_window:
        return False, '', '', {}

    curr_price = df['收盘价'].iloc[idx]
    curr_volume = df['成交量'].iloc[idx]
    curr_hist = df['Histogram'].iloc[idx]

    price_new_high = is_price_new_high(df, idx, extremes, params, lookback=10)
    price_new_low = is_price_new_low(df, idx, extremes, params, lookback=10)

    # 成交量相对位置
    vol_range = extremes['volume_max'] - extremes['volume_min']
    vol_level = (curr_volume - extremes['volume_min']) / vol_range if vol_range > 0 else 0.5
    volume_high = vol_level > 0.8  # 成交量在高位（放量）
    volume_low = vol_level < 0.2   # 成交量在低位（缩量）

    # MACD相对位置
    hist_range = extremes['hist_max'] - extremes['hist_min']
    hist_level = (curr_hist - extremes['hist_min']) / hist_range if hist_range != 0 else 0.5
    hist_high = curr_hist > extremes['hist_max'] * params.hist_threshold and curr_hist > 0
    hist_low = curr_hist < extremes['hist_min'] * (2 - params.hist_threshold) and curr_hist < 0

    # MACD峰值信息（前大后小判断）
    peak_info = get_histogram_peak_info(df, idx, params.divergence_window)
    hist_diminishing = peak_info['has_peak'] and peak_info['peak_hist'] < peak_info['prev_peak_hist'] * 0.9

    details = {
        'price_new_high': price_new_high,
        'price_new_low': price_new_low,
        'volume_high': volume_high,
        'volume_low': volume_low,
        'hist_high': hist_high,
        'hist_low': hist_low,
        'hist_diminishing': hist_diminishing,
        'hist_level': hist_level,
        'volume_level': vol_level,
        'peak_info': peak_info
    }

    # ========== 顶背离检测 ==========
    if price_new_high:
        # 经典三重顶背离：股价新高 + 成交量没放大 + MACD没新高
        if not volume_high and not hist_high:
            return True, '顶背离', '三重', details

        # 量价顶背离（放量滞涨）：股价新高 + 成交量天量 + MACD没新高
        if volume_high and not hist_high:
            return True, '顶背离', '量价', details

        # 无量顶背离：股价新高 + MACD新高 + 成交量没新高
        if hist_high and not volume_high:
            return True, '顶背离', '无量', details

        # 单纯MACD顶背离：股价新高 + MACD红柱前大后小
        if hist_diminishing and curr_hist > 0:
            return True, '顶背离', '单纯', details

    # ========== 底背离检测 ==========
    if price_new_low:
        # 经典三重底背离：股价新低 + 成交量没新低(放大) + MACD没新低
        if not volume_low and not hist_low:
            return True, '底背离', '三重', details

        # 地量底背离：股价新低 + 成交量地量 + MACD没新低
        if volume_low and not hist_low:
            return True, '底背离', '地量', details

        # 有量底背离：股价新低 + MACD新低 + 成交量没新低
        if hist_low and not volume_low:
            return True, '底背离', '有量', details

        # 单纯MACD底背离：股价新低 + MACD绿柱前大后小（绝对值）
        if peak_info['has_peak']:
            curr_negative = abs(min(curr_hist, 0))
            prev_negative = abs(min(peak_info['prev_peak_hist'], 0))
            if curr_negative < prev_negative * 0.9:  # 绿柱绝对值前大后小
                return True, '底背离', '单纯', details

    return False, '', '', details


def check_zero_axis(df: pd.DataFrame, idx: int) -> str:
    """检查零轴位置"""
    if idx < 1:
        return 'crossing'
    curr_hist = df['Histogram'].iloc[idx]
    prev_hist = df['Histogram'].iloc[idx - 1]
    if curr_hist > 0 and prev_hist > 0:
        return 'above'  # 多头区域
    elif curr_hist < 0 and prev_hist < 0:
        return 'below'  # 空头区域
    else:
        return 'crossing'  # 穿越零轴


def check_confirm(df: pd.DataFrame, idx: int, signal_type: SignalType) -> Tuple[bool, str]:
    """
    检查确认信号
    - 买入确认：金叉、股价突破5日均线、出现红柱
    - 卖出确认：死叉、股价跌破5日均线、出现绿柱
    """
    if idx < 1:
        return False, '无'

    curr_diff = df['DIFF'].iloc[idx]
    curr_dea = df['DEA'].iloc[idx]
    prev_diff = df['DIFF'].iloc[idx - 1]
    prev_dea = df['DEA'].iloc[idx - 1]

    curr_hist = df['Histogram'].iloc[idx]
    prev_hist = df['Histogram'].iloc[idx - 1]
    curr_ma5 = df['MA5'].iloc[idx]
    curr_price = df['收盘价'].iloc[idx]

    if signal_type == SignalType.BUY:
        # 金叉确认
        if prev_diff < prev_dea and curr_diff >= curr_dea:
            return True, '金叉'
        # 均线确认
        if curr_price > curr_ma5:
            return True, '均线'
        # 红柱出现
        if curr_hist > 0 and prev_hist < 0:
            return True, '红柱'
    else:  # SELL
        # 死叉确认
        if prev_diff >= prev_dea and curr_diff < curr_dea:
            return True, '死叉'
        # 均线确认
        if curr_price < curr_ma5:
            return True, '均线'
        # 绿柱出现
        if curr_hist < 0 and prev_hist > 0:
            return True, '绿柱'

    return False, '无'


def check_intraday_avg_filter(
    df: pd.DataFrame,
    idx: int,
    signal_type: SignalType,
    tolerance_pct: float,
    min_bias_pct: float
) -> Tuple[bool, str, float, float]:
    """检查信号是否满足分时均价线做T条件。"""
    current_price = df['收盘价'].iloc[idx]
    intraday_avg_price = df['分时均价'].iloc[idx]

    if intraday_avg_price <= 0:
        return False, 'near', intraday_avg_price, 0.0

    bias_pct = (current_price - intraday_avg_price) / intraday_avg_price
    if bias_pct > tolerance_pct:
        position = 'above'
    elif bias_pct < -tolerance_pct:
        position = 'below'
    else:
        position = 'near'

    if signal_type == SignalType.SELL:
        return position == 'above' and bias_pct >= min_bias_pct, position, intraday_avg_price, bias_pct
    return position == 'below' and bias_pct <= -min_bias_pct, position, intraday_avg_price, bias_pct


def detect_signals(df: pd.DataFrame, params: MacdParams) -> List[TradeSignal]:
    """
    检测买卖信号 - 量价MACD三维背离完整版

    信号强度：
    - 强信号：三重背离 + 零轴确认 + 顺势
    - 普通信号：单一背离 + 确认
    - 弱信号：背离但无确认
    """
    signals = []
    divergence_history: List[DivergenceRecord] = []  # 记录背离历史
    last_signal_time = None
    last_signal_time_by_type: Dict[SignalType, pd.Timestamp] = {}

    for i in range(params.divergence_window, len(df) - 1):
        current_time = df['时间'].iloc[i]
        current_price = df['收盘价'].iloc[i]
        current_volume = df['成交量'].iloc[i]
        avg_volume = df['成交量均线'].iloc[i]

        # 信号间隔检查
        can_signal = (last_signal_time is None or
                     (current_time - last_signal_time).total_seconds() > params.signal_interval)
        if not can_signal:
            continue

        # 获取窗口极值
        extremes = get_window_extremes(df, i, params.divergence_window)

        # 检测背离类型
        has_div, div_direction, div_sub_type, div_details = detect_divergence_type(df, i, params, extremes)

        # 零轴位置
        zero_axis = check_zero_axis(df, i)

        # 顺势检测
        is_trend_following = False
        if i >= 1:
            price_up = current_price > df['收盘价'].iloc[i - 1]
            hist_up = df['Histogram'].iloc[i] > df['Histogram'].iloc[i - 1]
            vol_up = current_volume > avg_volume * 1.2
            is_trend_following = (price_up and hist_up and vol_up) or \
                                (not price_up and not hist_up and vol_up)

        # 统计背离次数（同一方向的连续背离）
        div_count = 0
        if has_div:
            # 计算最近同方向背离次数
            for rec in reversed(divergence_history):
                if rec.signal_type.name.lower() == div_direction.lower():
                    div_count += 1
                else:
                    break
            div_count += 1  # 加上当前这次

            # 记录背离
            rec = DivergenceRecord(
                time=current_time,
                signal_type=SignalType.BUY if div_direction == '底背离' else SignalType.SELL,
                divergence_type=div_direction,
                sub_type=div_sub_type,
                price_level=div_details.get('volume_level', 0.5),
                hist_level=div_details.get('hist_level', 0.5),
                volume_level=div_details.get('volume_level', 0.5),
                index=i
            )
            divergence_history.append(rec)
            # 保持历史记录不要过长
            if len(divergence_history) > 50:
                divergence_history = divergence_history[-30:]

        # 构建信号
        if has_div:
            signal_type = SignalType.BUY if div_direction == '底背离' else SignalType.SELL

            # 检查确认
            confirmed, confirm_type = check_confirm(df, i, signal_type)
            avg_line_valid, avg_line_position, intraday_avg_price, avg_price_bias_pct = check_intraday_avg_filter(
                df, i, signal_type, params.intraday_avg_tolerance_pct, params.min_avg_price_bias_pct
            )

            if not avg_line_valid:
                continue

            last_same_type_time = last_signal_time_by_type.get(signal_type)
            if last_same_type_time is not None:
                same_type_ok = (current_time - last_same_type_time).total_seconds() > params.same_signal_interval
                if not same_type_ok:
                    continue

            # 判断强度
            is_three_div = div_sub_type == '三重'
            zero_confirmed = (signal_type == SignalType.BUY and zero_axis == 'below') or \
                           (signal_type == SignalType.SELL and zero_axis == 'above')

            if is_three_div and zero_confirmed and is_trend_following and confirmed:
                level = SignalLevel.STRONG
            elif is_three_div or (confirmed and div_count >= 2):
                level = SignalLevel.NORMAL
            else:
                level = SignalLevel.WEAK

            if not params.allow_weak_signals and level == SignalLevel.WEAK:
                continue

            # 构建原因
            reasons = [div_direction]
            if div_sub_type != '单纯':
                reasons.append(f"{div_sub_type}")
            if zero_axis == 'above':
                reasons.append('零轴上')
            elif zero_axis == 'below':
                reasons.append('零轴下')
            reasons.append('均价上方' if avg_line_position == 'above' else '均价下方')
            reasons.append(f'偏离{avg_price_bias_pct:+.2%}')
            if is_trend_following:
                reasons.append('顺势')
            if confirmed:
                reasons.append(f'已确认({confirm_type})')
            if div_count > 1:
                reasons.append(f'连续{div_count}次')

            signal = TradeSignal(
                time=current_time,
                signal_type=signal_type,
                price=current_price,
                intraday_avg_price=intraday_avg_price,
                avg_price_bias_pct=avg_price_bias_pct,
                avg_line_position=avg_line_position,
                reason='+'.join(reasons),
                signal_level=level,
                divergence_type=div_direction,
                sub_type=div_sub_type,
                confirmed=confirmed,
                confirm_type=confirm_type,
                divergence_count=div_count,
                zero_axis=zero_axis,
                trend_following=is_trend_following,
                index=i
            )
            signals.append(signal)
            last_signal_time = current_time
            last_signal_time_by_type[signal_type] = current_time

    return signals


# ==================== 绘图 ====================

def plot_chart(
    df: pd.DataFrame,
    signals: List[TradeSignal],
    stock_code: str,
    date_str: str,
    save_path: str
) -> None:
    """绘制分时图"""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "STHeiti"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.facecolor"] = "#F8F8F8"
    plt.rcParams["figure.facecolor"] = "#FFFFFF"

    fig, (ax_price, ax_macd, ax_vol) = plt.subplots(
        3, 1, figsize=(18, 12),
        gridspec_kw={'height_ratios': [3, 1.2, 1], 'hspace': 0.08}
    )

    x = np.arange(len(df))
    times = df['时间'].dt.strftime('%H:%M')
    price = df['收盘价']
    intraday_avg = df['分时均价']
    open_price = df['开盘价'].iloc[0]
    ma5 = df['MA5']
    price_range = price.max() - price.min()
    price_offset_base = price_range if price_range > 0 else max(open_price * 0.01, 0.01)

    # ========== 价格面板 ==========
    ax_price.plot(x, price, color='#1E90FF', linewidth=1.6, label='分时价格')
    ax_price.plot(x, intraday_avg, color='#8B4513', linewidth=1.2, alpha=0.9, label='分时均价')
    ax_price.plot(x, ma5, color='orange', linewidth=0.8, linestyle='--', alpha=0.7, label='MA5')
    ax_price.axhline(
        y=open_price,
        color='gray',
        linewidth=0.8,
        linestyle='--',
        alpha=0.5,
        label=f"开盘价 {open_price:.2f}"
    )

    # 标注信号
    for signal in signals:
        is_buy = signal.signal_type == SignalType.BUY
        color = '#32CD32' if is_buy else '#FF4545'
        edge_color = 'darkgreen' if is_buy else 'darkred'
        marker = '^' if is_buy else 'v'
        label = '买入' if is_buy else '卖出'

        # 信号强度显示不同大小
        if signal.signal_level == SignalLevel.STRONG:
            size, alpha = 350, 1.0
        elif signal.signal_level == SignalLevel.NORMAL:
            size, alpha = 250, 0.9
        else:
            size, alpha = 180, 0.8

        ax_price.scatter(
            signal.index,
            signal.price,
            color=color,
            s=size,
            marker=marker,
            zorder=6,
            edgecolors=edge_color,
            linewidths=2,
            alpha=alpha
        )

        # 子类型标记
        sub_marker = {
            '三重': '★★★',
            '量价': '★★',
            '地量': '★★',
            '无量': '★',
            '有量': '★',
            '单纯': '☆'
        }.get(signal.sub_type, '')

        offset = -price_offset_base * 0.12 if is_buy else price_offset_base * 0.12
        ax_price.annotate(
            f"{sub_marker}{label}[{signal.signal_level.value}]\n{signal.price:.2f}\n{signal.reason}",
            xy=(signal.index, signal.price),
            xytext=(signal.index, signal.price + offset),
            fontsize=7,
            color=color,
            fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0)
        )

    ax_price.set_ylabel('价格 (元)', fontsize=11, color='#1E90FF')
    ax_price.set_ylim(price.min() - price_offset_base * 0.2, price.max() + price_offset_base * 0.45)
    ax_price.legend(loc='upper left')
    ax_price.grid(True, color='#DDDDDD')

    # X轴时间
    all_key_times = ['09:30', '10:00', '10:30', '11:30', '13:00', '13:30', '14:00', '14:30', '15:00']
    time_to_idx = {t: i for i, t in enumerate(times)}
    display_ticks, tick_labels = [], []
    skip_next = False

    for idx, time_label in enumerate(all_key_times):
        if skip_next:
            skip_next = False
            continue
        if time_label not in time_to_idx:
            continue
        pos = time_to_idx[time_label]
        if time_label == '11:30' and idx + 1 < len(all_key_times) and all_key_times[idx + 1] == '13:00':
            if '13:00' in time_to_idx:
                display_ticks.append(pos)
                tick_labels.append('11:30/13:00')
                skip_next = True
                continue
        display_ticks.append(pos)
        tick_labels.append(time_label)

    if display_ticks:
        ax_price.set_xticks(display_ticks)
        ax_price.set_xticklabels(tick_labels, fontsize=9)
        ax_price.set_xlim(display_ticks[0], display_ticks[-1])
    ax_price.set_xlabel('时间', fontsize=11)

    # 涨跌幅
    ax_pct = ax_price.twinx()
    pct_change = ((price - open_price) / open_price * 100).values
    ax_pct.set_ylabel('涨跌幅 (%)', fontsize=11)
    ax_pct.set_ylim(pct_change.min() - 0.5, pct_change.max() + 0.5)

    # ========== MACD面板 ==========
    ax_macd.plot(x, df['DIFF'], color='#1E90FF', linewidth=1.3, label='DIFF(5)')
    ax_macd.plot(x, df['DEA'], color='#FF8C00', linewidth=1.0, label='DEA(5)')

    for i in range(len(df)):
        if df['Histogram'].iloc[i] >= 0:
            ax_macd.bar(x[i], df['Histogram'].iloc[i], color='#FF4545', width=0.6, alpha=0.7)
        else:
            ax_macd.bar(x[i], df['Histogram'].iloc[i], color='#32CD32', width=0.6, alpha=0.7)

    ax_macd.axhline(y=0, color='gray', linewidth=0.8)
    ax_macd.set_ylabel('MACD', fontsize=11)
    ax_macd.legend(loc='upper left')
    ax_macd.grid(True, color='#DDDDDD')
    ax_macd.set_xticks([])

    # MACD信号标注
    for signal in signals:
        color = '#32CD32' if signal.signal_type == SignalType.BUY else '#FF4545'
        marker = '^' if signal.signal_type == SignalType.BUY else 'v'
        ax_macd.scatter(
            signal.index,
            df['DEA'].iloc[signal.index],
            color=color, s=80, marker=marker, zorder=6
        )

    # ========== 成交量面板 ==========
    for i in range(len(df)):
        color = '#FF4545' if (i > 0 and price.iloc[i] >= price.iloc[i - 1]) else '#00CD00'
        ax_vol.bar(x[i], df['成交量'].iloc[i], color=color, width=0.6, alpha=0.8)

    ax_vol.plot(x, df['成交量均线'], color='purple', linewidth=1.0, linestyle='--', alpha=0.7, label='成交量均线')
    ax_vol.set_ylabel('成交量', fontsize=11)
    ax_vol.legend(loc='upper left')
    ax_vol.set_xticks([])

    # ========== 标题 ==========
    close_price = df['收盘价'].iloc[-1]
    change = close_price - open_price
    pct = change / open_price * 100
    change_color = '#FF4545' if change >= 0 else '#32CD32'

    buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
    sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
    strong_count = sum(1 for s in signals if s.signal_level == SignalLevel.STRONG)
    normal_count = sum(1 for s in signals if s.signal_level == SignalLevel.NORMAL)
    weak_count = sum(1 for s in signals if s.signal_level == SignalLevel.WEAK)

    fig.text(0.28, 0.94, f'{stock_code} {date_str}', fontsize=14, fontweight='bold', va='top')
    fig.text(
        0.60,
        0.94,
        f'开盘 {open_price:.2f} → 收盘 {close_price:.2f}  {change:+.2f} ({pct:+.2f}%)',
        color=change_color, fontsize=11, va='top'
    )
    fig.text(
        0.28, 0.91,
        f'买入{buy_count}个(强{strong_count}/普{normal_count}/弱{weak_count}) | 卖出{sell_count}个 | 策略:量价MACD背离v4.2',
        fontsize=10, va='top'
    )
    fig.text(
        0.28, 0.88,
        '买点:底背离 + 分时均价下方 | 卖点:顶背离 + 分时均价上方',
        fontsize=9, va='top'
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def print_signals(signals: List[TradeSignal]) -> None:
    """打印信号详情"""
    print()
    print('=' * 120)
    print('MACD分时策略v4.2 - 信号列表 (量价MACD背离 + 分时均价做T)')
    print('=' * 120)
    print()
    print(f"{'时间':<10} {'信号':<4} {'价格':<8} {'均价':<8} {'偏离':<8} {'背离类型':<12} {'子类型':<8} {'确认':<8} {'强度':<6} {'说明'}")
    print('-' * 120)

    for s in signals:
        signal_label = '买入' if s.signal_type == SignalType.BUY else '卖出'
        confirm = s.confirm_type if s.confirmed else '-'
        count = f"×{s.divergence_count}" if s.divergence_count > 1 else ''

        print(
            f"{s.time.strftime('%H:%M'):<10} {signal_label:<4} {s.price:<8.2f} "
            f"{s.intraday_avg_price:<8.2f} {s.avg_price_bias_pct:+.2%} "
            f"{s.divergence_type:<12} {s.sub_type:<8} {confirm:<8} "
            f"{s.signal_level.value:<6} {s.reason}"
        )

    print()
    buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
    sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
    strong_count = sum(1 for s in signals if s.signal_level == SignalLevel.STRONG)

    print(f'买入信号: {buy_count}个 (强信号:{strong_count}个)')
    print(f'卖出信号: {sell_count}个 (强信号:{strong_count}个)')


# ==================== 工具函数 ====================

def find_csv_file(code: str, date_str: str = None) -> Tuple[str, str]:
    """查找CSV文件"""
    DATA_DIR = "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/回测数据/26逐笔"

    if date_str is None:
        date_dirs = []
        if os.path.exists(DATA_DIR):
            date_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()]
        if not date_dirs:
            raise FileNotFoundError(f"数据目录为空: {DATA_DIR}")
        date_dirs.sort(reverse=True)
        date_str = date_dirs[0]

    stocks_dir = os.path.join(DATA_DIR, date_str, "stocks")
    if not os.path.exists(stocks_dir):
        raise FileNotFoundError(f"数据目录不存在: {stocks_dir}")

    code = code.strip()
    if code.startswith(("60", "68")):
        patterns = [f"sh{code}_{date_str}.csv", f"{code}_{date_str}.csv"]
    elif code.startswith(("00", "30")):
        patterns = [f"sz{code}_{date_str}.csv", f"{code}_{date_str}.csv"]
    else:
        patterns = [f"{code}_{date_str}.csv"]

    for pattern in patterns:
        csv_path = os.path.join(stocks_dir, pattern)
        if os.path.exists(csv_path):
            return csv_path, date_str

    raise FileNotFoundError(f"未找到股票 {code} 在 {date_str} 的数据文件")


def get_random_stock(date_str: str = None) -> Tuple[str, str]:
    """随机选择股票"""
    DATA_DIR = "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/回测数据/26逐笔"

    if date_str is None:
        date_dirs = []
        if os.path.exists(DATA_DIR):
            date_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()]
        if not date_dirs:
            raise FileNotFoundError(f"数据目录为空: {DATA_DIR}")
        date_dirs.sort(reverse=True)
        date_str = date_dirs[0]

    stocks_dir = os.path.join(DATA_DIR, date_str, "stocks")
    csv_files = [f for f in os.listdir(stocks_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"没有股票数据: {stocks_dir}")

    base_name = random.choice(csv_files).replace('.csv', '')
    if base_name.startswith(('sh', 'sz', 'bj')):
        return base_name[2:].replace(f'_{date_str}', ''), date_str
    return base_name.replace(f'_{date_str}', ''), date_str


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MACD分时策略 v4.2 - 量价MACD背离 + 分时均价做T",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python v4_2.py --code 000559
  python v4_2.py --code 000559 --date 20260424
  python v4_2.py --random
        """
    )
    parser.add_argument("--code", "-c", default=None, help="股票代码")
    parser.add_argument("--date", "-d", default=None, help="交易日期，格式YYYYMMDD")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--random", "-r", action="store_true", help="随机选择股票")
    return parser.parse_args()


def main() -> None:
    """主入口"""
    args = parse_args()

    if args.random:
        stock_code, detected_date = get_random_stock(args.date)
        csv_path, _ = find_csv_file(stock_code, detected_date)
        print(f"[随机模式] 选中股票: {stock_code}")
    elif args.code:
        stock_code = args.code
        csv_path, detected_date = find_csv_file(stock_code, args.date)
    else:
        print("错误: 请指定 --code 或 --random")
        return

    print(f"使用数据文件: {csv_path}")
    print(f"数据日期: {detected_date}")

    output_dir = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    display_date = f"{detected_date[:4]}-{detected_date[4:6]}-{detected_date[6:8]}"

    params = MacdParams(
        short_period=5,
        long_period=34,
        dea_period=5,
        divergence_window=20,
        signal_interval=600,
        price_threshold=0.999,
        volume_threshold=0.9,
        hist_threshold=0.9,
        min_divergence_count=1,
        ma_short=5,
        ma_medium=20,
        intraday_avg_tolerance_pct=0.0005,
        min_avg_price_bias_pct=0.003,
        same_signal_interval=1800,
        allow_weak_signals=False
    )

    print("=" * 60)
    print("MACD分时策略v4.2 - 量价MACD背离 + 分时均价做T")
    print(f"参数: short={params.short_period}, long={params.long_period}, dea={params.dea_period}")
    print(f"背离窗口: {params.divergence_window}根K线, 信号间隔: {params.signal_interval}秒")
    print(f"分时均价过滤容忍度: {params.intraday_avg_tolerance_pct:.2%}")
    print(f"最小均价偏离: {params.min_avg_price_bias_pct:.2%}, 同方向冷却: {params.same_signal_interval}秒, 弱信号保留: {params.allow_weak_signals}")
    print("=" * 60)

    print("读取数据(1分钟聚合)...")
    df = load_and_aggregate(csv_path, agg_seconds=60, target_date=detected_date)
    print(f"聚合后: {len(df)}根K线")

    print("计算MACD指标...")
    df = calculate_macd(df, params)

    print("检测信号(量价MACD背离 + 分时均价做T)...")
    signals = detect_signals(df, params)
    print_signals(signals)

    save_path = os.path.join(output_dir, f"MACD分时v4.2_{stock_code}_{detected_date}.png")
    print(f"\n绘图: {save_path}")
    plot_chart(df, signals, stock_code, display_date, save_path)
    print(f"已保存: {save_path}")

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
