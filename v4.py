"""
MACD分时策略 v4 - 量价MACD三维背离策略
========================================
核心思想：成交量、MACD红绿柱、股价三者关系与背离分析

三者关系：
1. 成交量：反映市场参与度，是趋势的燃料
2. MACD Histogram：反映多空力量对比，是趋势的加速度
3. 股价：反映价格变动，是趋势的结果

背离逻辑：
- 顶背离（卖出信号）：价格创新高，但MACD或成交量未跟随
- 底背离（买入信号）：价格创新低，但MACD或成交量未跟随

顺势逻辑：
- 放量 + MACD放大 + 股价配合 = 趋势确认

参数：
- 短期EMA: 5分钟
- 长期EMA: 34分钟
- DEA平滑: 5分钟
- 信号间隔: 10分钟
- 背离窗口: 20根K线
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"


class SignalLevel(Enum):
    """信号强度"""
    STRONG = "强信号"      # 背离确认 + 顺势
    NORMAL = "普通信号"    # 单一背离信号
    WEAK = "弱信号"        # 需要谨慎


@dataclass
class TradeSignal:
    """交易信号"""
    time: pd.Timestamp
    signal_type: SignalType
    price: float
    reason: str
    signal_level: SignalLevel
    # 三维度数据
    price_level: float       # 价格所处位置 (相对窗口高点的比例)
    volume_level: float      # 成交量所处位置
    hist_level: float       # MACD Histogram所处位置
    # 背离类型
    divergence: bool        # 是否存在背离
    divergence_type: str     # '顶背离'/'底背离'/'量价背离'/''
    # 顺势确认
    trend_following: bool   # 是否顺势
    trend_direction: str    # 'up'/'down'/''
    index: int


@dataclass
class MacdParams:
    """MACD参数"""
    short_period: int = 5
    long_period: int = 34
    dea_period: int = 5
    divergence_window: int = 20
    signal_interval: int = 600  # 信号间隔10分钟
    # 背离确认阈值
    price_threshold: float = 0.999  # 价格创新高/新低的阈值
    volume_threshold: float = 0.9   # 成交量阈值
    hist_threshold: float = 0.9     # MACD阈值


def load_and_aggregate(csv_path: str, agg_seconds: int = 60, target_date: str = None) -> pd.DataFrame:
    """加载并聚合逐笔数据"""
    df = pd.read_csv(csv_path)
    df['时间'] = pd.to_datetime(df['时间'])

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
        df = df[['时间', '成交量', '收盘价']].copy()
        df['开盘价'] = df['收盘价']
        df['最高价'] = df['收盘价']
        df['最低价'] = df['收盘价']
        df['成交量均线'] = df['成交量'].rolling(window=100, min_periods=1).mean()
        return df

    df['周期'] = df['时间'].dt.floor(f'{agg_seconds}s')
    agg = df.groupby('周期').agg({
        '价格': ['first', 'last', 'max', 'min'],
        '成交量': 'sum'
    }).reset_index()
    agg.columns = ['时间', '开盘价', '收盘价', '最高价', '最低价', '成交量']
    agg = agg.sort_values('时间').reset_index(drop=True)
    return agg


def calculate_macd(df: pd.DataFrame, params: MacdParams) -> pd.DataFrame:
    """计算MACD指标"""
    close = df['收盘价']
    df['EMA_Short'] = close.ewm(span=params.short_period, adjust=False).mean()
    df['EMA_Long'] = close.ewm(span=params.long_period, adjust=False).mean()
    df['DIFF'] = df['EMA_Short'] - df['EMA_Long']
    df['DEA'] = df['DIFF'].ewm(span=params.dea_period, adjust=False).mean()
    df['Histogram'] = (df['DIFF'] - df['DEA']) * 2

    # 成交量均线
    df['成交量均线'] = df['成交量'].rolling(window=20, min_periods=1).mean()

    return df


def detect_divergence_3d(
    df: pd.DataFrame,
    idx: int,
    params: MacdParams
) -> Tuple[bool, str, Dict]:
    """
    三维度背离检测：成交量、MACD Histogram、股价

    Returns:
        (是否存在背离, 背离类型, 详细信息)
    """
    if idx < params.divergence_window:
        return False, '', {}

    window = df.iloc[idx - params.divergence_window:idx + 1]

    # 当前值
    curr_price = df['收盘价'].iloc[idx]
    curr_volume = df['成交量'].iloc[idx]
    curr_hist = df['Histogram'].iloc[idx]

    # 窗口内的极值
    price_max = window['收盘价'].max()
    price_min = window['收盘价'].min()
    volume_max = window['成交量'].max()
    volume_min = window['成交量'].min()
    hist_max = window['Histogram'].max()
    hist_min = window['Histogram'].min()

    details = {
        'price_level': 0.0,    # 价格位置 (0=最低, 1=最高)
        'volume_level': 0.0,   # 成交量位置
        'hist_level': 0.0,     # MACD位置
        'price_new_high': False,
        'price_new_low': False,
        'volume_new_high': False,
        'volume_new_low': False,
        'hist_new_high': False,
        'hist_new_low': False,
    }

    # 计算相对位置 (0-1)
    price_range = price_max - price_min
    volume_range = volume_max - volume_min
    hist_range = hist_max - hist_min if hist_max != hist_min else 1

    if price_range > 0:
        details['price_level'] = (curr_price - price_min) / price_range
    if volume_range > 0:
        details['volume_level'] = (curr_volume - volume_min) / volume_range
    if hist_range != 0:
        details['hist_level'] = (curr_hist - hist_min) / hist_range

    # 判断是否创新高/新低
    details['price_new_high'] = curr_price >= price_max * params.price_threshold
    details['price_new_low'] = curr_price <= price_min * (2 - params.price_threshold)
    details['volume_new_high'] = curr_volume >= volume_max * params.volume_threshold
    details['volume_new_low'] = curr_volume <= volume_min * (2 - params.volume_threshold)
    details['hist_new_high'] = curr_hist >= hist_max * params.hist_threshold and curr_hist > 0
    details['hist_new_low'] = curr_hist <= hist_min * (2 - params.hist_threshold) and curr_hist < 0

    # ========== 顶背离检测（看跌信号）==========
    # 情况1: 价格创新高，但MACD没创新高 -> 经典顶背离
    if details['price_new_high'] and not details['hist_new_high'] and curr_hist < hist_max * params.hist_threshold:
        return True, '顶背离(MACD)', details

    # 情况2: 价格创新高，但成交量没放大 -> 量价背离
    if details['price_new_high'] and not details['volume_new_high']:
        return True, '顶背离(量价)', details

    # 情况3: 成交量放大，但价格没涨 -> 滞涨
    if details['volume_new_high'] and details['price_level'] < 0.7 and curr_hist < 0:
        return True, '滞涨', details

    # ========== 底背离检测（看涨信号）==========
    # 情况1: 价格创新低，但MACD没创新低 -> 经典底背离
    if details['price_new_low'] and not details['hist_new_low'] and curr_hist > hist_min * (2 - params.hist_threshold):
        return True, '底背离(MACD)', details

    # 情况2: 价格创新低，但成交量没放大 -> 加速赶底
    if details['price_new_low'] and not details['volume_new_low']:
        return True, '加速赶底', details

    # 情况3: 成交量新低，但价格没新低 -> 止跌
    if details['volume_new_low'] and details['price_level'] > 0.3 and curr_hist > 0:
        return True, '缩量止跌', details

    return False, '', details


def detect_trend_following(
    df: pd.DataFrame,
    idx: int,
    params: MacdParams
) -> Tuple[bool, str]:
    """
    顺势检测：成交量、MACD、股价三者配合

    Returns:
        (是否顺势, 趋势方向)
    """
    if idx < 2:
        return False, ''

    curr_price = df['收盘价'].iloc[idx]
    prev_price = df['收盘价'].iloc[idx - 1]
    curr_volume = df['成交量'].iloc[idx]
    avg_volume = df['成交量均线'].iloc[idx]
    curr_hist = df['Histogram'].iloc[idx]
    prev_hist = df['Histogram'].iloc[idx - 1]

    vol_ratio = curr_volume / avg_volume if avg_volume > 0 else 1

    # 上涨顺势：成交量放大 + MACD红柱放大 + 股价上涨
    price_up = curr_price > prev_price
    hist_up = curr_hist > prev_hist and curr_hist > 0
    volume_up = vol_ratio > 1.2

    if price_up and hist_up and volume_up:
        return True, 'up'

    # 下跌顺势：成交量放大 + MACD绿柱放大 + 股价下跌
    price_down = curr_price < prev_price
    hist_down = curr_hist < prev_hist and curr_hist < 0
    volume_down = vol_ratio > 1.2

    if price_down and hist_down and volume_down:
        return True, 'down'

    return False, ''


def detect_signals(df: pd.DataFrame, params: MacdParams) -> List[TradeSignal]:
    """
    检测买卖信号 - 量价MACD三维背离策略

    信号优先级：
    1. 强信号：背离确认 + 顺势交易
    2. 普通信号：单一背离信号
    3. 弱信号：需要观望
    """
    signals = []
    last_signal_time = None

    for i in range(params.divergence_window, len(df) - 1):
        current_time = df['时间'].iloc[i]
        current_price = df['收盘价'].iloc[i]
        current_volume = df['成交量'].iloc[i]
        avg_volume = df['成交量均线'].iloc[i]
        current_hist = df['Histogram'].iloc[i]

        # 信号间隔检查
        can_signal = (last_signal_time is None or
                     (current_time - last_signal_time).total_seconds() > params.signal_interval)
        if not can_signal:
            continue

        # ========== 三维度背离检测 ==========
        has_divergence, divergence_type, div_details = detect_divergence_3d(df, i, params)

        # ========== 顺势检测 ==========
        is_trend_following, trend_direction = detect_trend_following(df, i, params)

        # ========== 构建信号 ==========
        if has_divergence:
            vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # 根据背离类型和顺势情况确定信号
            if '顶背离' in divergence_type:
                # 顶背离 = 卖出信号
                if is_trend_following and trend_direction == 'down':
                    level = SignalLevel.STRONG
                    reason = f"{divergence_type}+顺势下跌"
                else:
                    level = SignalLevel.NORMAL
                    reason = divergence_type

                signals.append(TradeSignal(
                    time=current_time,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    reason=reason,
                    signal_level=level,
                    price_level=div_details.get('price_level', 0),
                    volume_level=div_details.get('volume_level', 0),
                    hist_level=div_details.get('hist_level', 0),
                    divergence=True,
                    divergence_type=divergence_type,
                    trend_following=is_trend_following and trend_direction == 'down',
                    trend_direction=trend_direction,
                    index=i
                ))
                last_signal_time = current_time

            elif '底背离' in divergence_type or '滞涨' not in divergence_type:
                # 底背离 = 买入信号
                if is_trend_following and trend_direction == 'up':
                    level = SignalLevel.STRONG
                    reason = f"{divergence_type}+顺势上涨"
                else:
                    level = SignalLevel.NORMAL
                    reason = divergence_type

                signals.append(TradeSignal(
                    time=current_time,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    reason=reason,
                    signal_level=level,
                    price_level=div_details.get('price_level', 0),
                    volume_level=div_details.get('volume_level', 0),
                    hist_level=div_details.get('hist_level', 0),
                    divergence=True,
                    divergence_type=divergence_type,
                    trend_following=is_trend_following and trend_direction == 'up',
                    trend_direction=trend_direction,
                    index=i
                ))
                last_signal_time = current_time

    return signals


def plot_chart(
    df: pd.DataFrame,
    signals: List[TradeSignal],
    stock_name: str,
    stock_code: str,
    date_str: str,
    save_path: str
) -> None:
    """绘制分时图 + MACD + 成交量 + 信号标注"""
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
    open_price = df['开盘价'].iloc[0]
    price_range = price.max() - price.min()
    price_offset_base = price_range if price_range > 0 else max(open_price * 0.01, 0.01)

    # ========== 价格面板 ==========
    ax_price.plot(x, price, color='#1E90FF', linewidth=1.6, label='分时价格')
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
            size = 300
            alpha = 1.0
        elif signal.signal_level == SignalLevel.NORMAL:
            size = 200
            alpha = 0.9
        else:
            size = 150
            alpha = 0.8

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

        # 背离标记
        div_marker = '★' if signal.divergence else ''
        offset = -price_offset_base * 0.12 if is_buy else price_offset_base * 0.12

        ax_price.annotate(
            f"{div_marker}{label}[{signal.signal_level.value}]\n{signal.price:.2f}\n{signal.reason}",
            xy=(signal.index, signal.price),
            xytext=(signal.index, signal.price + offset),
            fontsize=8,
            color=color,
            fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0)
        )

    ax_price.set_ylabel('价格 (元)', fontsize=11, color='#1E90FF')
    ax_price.set_ylim(price.min() - price_offset_base * 0.2, price.max() + price_offset_base * 0.4)
    ax_price.legend(loc='upper left')
    ax_price.grid(True, color='#DDDDDD')

    # X轴时间标签
    all_key_times = ['09:30', '10:00', '10:30', '11:30', '13:00', '13:30', '14:00', '14:30', '15:00']
    time_to_idx = {t: i for i, t in enumerate(times)}
    display_ticks = []
    tick_labels = []
    skip_next = False

    for idx, time_label in enumerate(all_key_times):
        if skip_next:
            skip_next = False
            continue
        if time_label not in time_to_idx:
            continue

        position = time_to_idx[time_label]
        if time_label == '11:30' and idx + 1 < len(all_key_times) and all_key_times[idx + 1] == '13:00':
            if '13:00' in time_to_idx:
                display_ticks.append(position)
                tick_labels.append('11:30/13:00')
                skip_next = True
                continue

        display_ticks.append(position)
        tick_labels.append(time_label)

    if display_ticks:
        ax_price.set_xticks(display_ticks)
        ax_price.set_xticklabels(tick_labels, fontsize=9)
        ax_price.set_xlim(display_ticks[0], display_ticks[-1])
    ax_price.set_xlabel('时间', fontsize=11)

    # 右侧涨跌幅
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

    # MACD上标注信号
    for signal in signals:
        is_buy = signal.signal_type == SignalType.BUY
        color = '#32CD32' if is_buy else '#FF4545'
        marker = '^' if is_buy else 'v'
        ax_macd.scatter(
            signal.index,
            df['DEA'].iloc[signal.index],
            color=color,
            s=80,
            marker=marker,
            zorder=6
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

    fig.text(0.30, 0.94, f'{stock_code} {stock_name} {date_str}', fontsize=14, fontweight='bold', va='top')
    fig.text(
        0.65,
        0.94,
        f'开盘 {open_price:.2f} → 收盘 {close_price:.2f}  涨跌 {change:+.2f} ({pct:+.2f}%)',
        color=change_color,
        fontsize=11,
        va='top'
    )
    fig.text(
        0.30,
        0.91,
        f'买入{buy_count}个 | 卖出{sell_count}个 | 强信号{strong_count}个 | 策略:量价MACD背离',
        fontsize=10,
        va='top'
    )
    fig.text(
        0.30,
        0.88,
        '买点:底背离+顺势 | 卖点:顶背离+顺势',
        fontsize=9,
        va='top'
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def print_signals(signals: List[TradeSignal]) -> None:
    """打印信号详情"""
    print()
    print('=' * 100)
    print('MACD分时策略v4 - 信号列表 (量价MACD三维背离)')
    print('=' * 100)
    print()
    print(f"{'时间':<10} {'信号':<4} {'价格':<8} {'类型':<16} {'强度':<6} {'顺势':<6} {'说明'}")
    print('-' * 100)

    for s in signals:
        signal_label = '买入' if s.signal_type == SignalType.BUY else '卖出'
        trend = '↑' if s.trend_direction == 'up' else ('↓' if s.trend_direction == 'down' else '-')
        div = s.divergence_type if s.divergence else ''

        print(
            f"{s.time.strftime('%H:%M'):<10} {signal_label:<4} {s.price:<8.2f} "
            f"{div:<16} {s.signal_level.value:<6} {trend:<6} {s.reason}"
        )

    print()
    buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
    sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
    strong_count = sum(1 for s in signals if s.signal_level == SignalLevel.STRONG)
    normal_count = sum(1 for s in signals if s.signal_level == SignalLevel.NORMAL)

    print(f'买入信号: {buy_count}个 (强:{strong_count} 普通:{normal_count})')
    print(f'卖出信号: {sell_count}个 (强:{strong_count} 普通:{normal_count})')


def find_csv_file(code: str, date_str: str = None) -> Tuple[str, str]:
    """在逐笔数据目录中查找股票CSV"""
    DATA_DIR = "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/回测数据/26逐笔"

    if date_str is None:
        date_dirs = []
        if os.path.exists(DATA_DIR):
            date_dirs = [
                d for d in os.listdir(DATA_DIR)
                if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()
            ]
        if not date_dirs:
            raise FileNotFoundError(f"数据目录为空或不存在: {DATA_DIR}")
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
    """从数据目录中随机选择一只股票"""
    DATA_DIR = "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/回测数据/26逐笔"

    if date_str is None:
        date_dirs = []
        if os.path.exists(DATA_DIR):
            date_dirs = [
                d for d in os.listdir(DATA_DIR)
                if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()
            ]
        if not date_dirs:
            raise FileNotFoundError(f"数据目录为空或不存在: {DATA_DIR}")
        date_dirs.sort(reverse=True)
        date_str = date_dirs[0]

    stocks_dir = os.path.join(DATA_DIR, date_str, "stocks")
    if not os.path.exists(stocks_dir):
        raise FileNotFoundError(f"股票目录不存在: {stocks_dir}")

    csv_files = [f for f in os.listdir(stocks_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"没有找到任何股票数据文件: {stocks_dir}")

    base_name = random.choice(csv_files).replace('.csv', '')
    if base_name.startswith(('sh', 'sz', 'bj')):
        return base_name[2:].replace(f'_{date_str}', ''), date_str
    return base_name.replace(f'_{date_str}', ''), date_str


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MACD分时策略 v4 - 量价MACD三维背离",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python v4.py --code 000559
  python v4.py --code 000559 --date 20260424
  python v4.py --random
        """
    )
    parser.add_argument("--code", "-c", default=None, help="股票代码")
    parser.add_argument("--date", "-d", default=None, help="交易日期，格式YYYYMMDD")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--random", "-r", action="store_true", help="随机选择一个股票进行分析")
    return parser.parse_args()


def main() -> None:
    """命令行主入口"""
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
        print("示例: python v4.py --code 000559")
        print("示例: python v4.py --random")
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
        hist_threshold=0.9
    )

    print("=" * 60)
    print("MACD分时策略v4 - 量价MACD三维背离")
    print(f"参数: short={params.short_period}, long={params.long_period}, dea={params.dea_period}")
    print(f"背离窗口: {params.divergence_window}根K线, 信号间隔: {params.signal_interval}秒")
    print("=" * 60)

    print("读取数据(1分钟聚合)...")
    df = load_and_aggregate(csv_path, agg_seconds=60, target_date=detected_date)
    print(f"聚合后: {len(df)}根K线")

    print("计算MACD指标...")
    df = calculate_macd(df, params)

    print("检测信号(量价MACD三维背离)...")
    signals = detect_signals(df, params)
    print_signals(signals)

    save_path = os.path.join(output_dir, f"MACD分时v4_{stock_code}_{detected_date}.png")
    print(f"\n绘图: {save_path}")
    plot_chart(df, signals, stock_code, stock_code, display_date, save_path)
    print(f"已保存: {save_path}")

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
