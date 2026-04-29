"""
MACD分时买卖策略 v3 - 金叉死叉顺势交易
========================================
核心策略：
1. 买入（金叉）：DIFF上穿DEA + Histogram在零轴上方（顺势）+ 缩量
2. 卖出（死叉）：DIFF下穿DEA + Histogram在零轴下方（顺势）+ 放量

逻辑：
- 金叉时顺势：要求Histogram>0，表示多头趋势中回踩后再次上攻
- 死叉时顺势：要求Histogram<0，表示空头趋势中反弹后再次下杀
- 量能过滤：缩量金叉更可靠，放量死叉更可靠

参数：
- 短期EMA: 5分钟
- 长期EMA: 34分钟
- DEA平滑: 5分钟
- 信号间隔: 10分钟
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import date, timedelta

# 添加 eltdx 模块路径
sys.path.insert(0, "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/test/通达信分时")
from eltdx import TdxClient


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"


class SignalLevel(Enum):
    """信号强度"""
    STRONG = "强信号"
    NORMAL = "普通信号"
    WEAK = "弱信号"


@dataclass
class TradeSignal:
    """交易信号"""
    time: pd.Timestamp
    signal_type: SignalType
    price: float
    histogram_prev: float
    histogram_curr: float
    volume: int
    avg_volume: float
    signal_level: SignalLevel
    divergence: bool
    divergence_type: str
    reason: str
    index: int


@dataclass
class MacdParams:
    """MACD参数"""
    short_period: int = 5
    long_period: int = 34
    dea_period: int = 5
    divergence_window: int = 20
    signal_interval: int = 900  # 信号间隔15分钟
    min_hist_change: float = 0.015  # Histogram最小变化幅度
    min_volume_ratio: float = 0.8  # 买入最小量比（缩量）
    max_volume_ratio: float = 1.2  # 卖出最大量比（放量）


def get_last_close_price(code: str, target_date: str) -> Tuple[float, float]:
    """获取股票的昨收价和开盘价

    Args:
        code: 股票代码，如 "600487"
        target_date: 目标日期，格式YYYYMMDD

    Returns:
        (last_close_price, open_price) - 昨收价和开盘价
    """
    # 标准化代码
    if code.startswith("60") or code.startswith("68"):
        full_code = f"sh{code}"
    elif code.startswith("00") or code.startswith("30"):
        full_code = f"sz{code}"
    else:
        raise ValueError(f"无法识别的股票代码: {code}")

    target = date(int(target_date[:4]), int(target_date[4:6]), int(target_date[6:8]))

    with TdxClient() as client:
        kline = client.get_kline_all('day', full_code)

        for item in kline.items:
            if item.time.date() == target:
                return item.last_close_price, item.open_price

    raise ValueError(f"未找到 {code} 在 {target_date} 的K线数据")


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

    df['成交量均线'] = df['成交量'].rolling(window=20, min_periods=1).mean()

    return df


def detect_divergence(df: pd.DataFrame, idx: int, params: MacdParams) -> Tuple[bool, str]:
    """检测背离"""
    if idx < params.divergence_window:
        return False, ''

    window_data = df.iloc[idx - params.divergence_window:idx + 1]

    price_current = df['收盘价'].iloc[idx]
    hist_current = df['Histogram'].iloc[idx]

    price_max = window_data['收盘价'].max()
    price_min = window_data['收盘价'].min()
    hist_max = window_data['Histogram'].max()
    hist_min = window_data['Histogram'].min()

    # 顶背离：价格创新高，但Histogram没创新高
    if price_current >= price_max * 0.999 and hist_current < hist_max * 0.9:
        return True, '顶背离'

    # 底背离：价格创新低，但Histogram没创新低
    if price_current <= price_min * 1.001 and hist_current > hist_min * 1.1:
        return True, '底背离'

    return False, ''


def detect_signals(df: pd.DataFrame, params: MacdParams) -> List[TradeSignal]:
    """检测买卖信号 - 金叉死叉顺势交易

    买入（金叉）：
      - DIFF上穿DEA（prev_diff<prev_dea 且 curr_diff>=curr_dea）
      - Histogram在零轴上方（顺势，多头趋势）
      - 缩量（量比<params.min_volume_ratio）

    卖出（死叉）：
      - DIFF下穿DEA（prev_diff>=prev_dea 且 curr_diff<curr_dea）
      - Histogram在零轴下方（顺势，空头趋势）
      - 放量（量比>params.max_volume_ratio）
    """
    signals = []
    last_signal_time = None

    for i in range(1, len(df) - 1):
        prev_diff = df['DIFF'].iloc[i - 1]
        curr_diff = df['DIFF'].iloc[i]
        next_diff = df['DIFF'].iloc[i + 1]

        prev_dea = df['DEA'].iloc[i - 1]
        curr_dea = df['DEA'].iloc[i]
        next_dea = df['DEA'].iloc[i + 1]

        curr_hist = df['Histogram'].iloc[i]

        current_time = df['时间'].iloc[i]
        current_price = df['收盘价'].iloc[i]
        current_volume = df['成交量'].iloc[i]
        avg_volume = df['成交量均线'].iloc[i]

        # 信号间隔
        can_signal = (last_signal_time is None or
                     (current_time - last_signal_time).total_seconds() > params.signal_interval)

        if not can_signal:
            continue

        vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # ========== 金叉检测（买入） ==========
        # DIFF上穿DEA：prev_diff < prev_dea 且 curr_diff >= curr_dea
        golden_cross = (prev_diff < prev_dea) and (curr_diff >= curr_dea)
        # 顺势：Histogram在零轴上方（多头趋势）
        bullish = curr_hist > 0
        # 缩量：量比低于阈值
        volume_filter = vol_ratio < params.min_volume_ratio

        if golden_cross and bullish and volume_filter:
            level = SignalLevel.STRONG
            reason = f"金叉({prev_diff:.4f}/{prev_dea:.4f}→{curr_diff:.4f}/{curr_dea:.4f})量比{vol_ratio:.1f}"

            signals.append(TradeSignal(
                time=current_time,
                signal_type=SignalType.BUY,
                price=current_price,
                histogram_prev=prev_diff - prev_dea,
                histogram_curr=curr_hist,
                volume=current_volume,
                avg_volume=avg_volume,
                signal_level=level,
                divergence=False,
                divergence_type='',
                reason=reason,
                index=i
            ))
            last_signal_time = current_time

        # ========== 死叉检测（卖出） ==========
        # DIFF下穿DEA：prev_diff >= prev_dea 且 curr_diff < curr_dea
        dead_cross = (prev_diff >= prev_dea) and (curr_diff < curr_dea)
        # 顺势：Histogram在零轴下方（空头趋势）
        bearish = curr_hist < 0
        # 放量：量比高于阈值
        volume_filter = vol_ratio > params.max_volume_ratio

        if dead_cross and bearish and volume_filter:
            level = SignalLevel.STRONG
            reason = f"死叉({prev_diff:.4f}/{prev_dea:.4f}→{curr_diff:.4f}/{curr_dea:.4f})量比{vol_ratio:.1f}"

            signals.append(TradeSignal(
                time=current_time,
                signal_type=SignalType.SELL,
                price=current_price,
                histogram_prev=prev_diff - prev_dea,
                histogram_curr=curr_hist,
                volume=current_volume,
                avg_volume=avg_volume,
                signal_level=level,
                divergence=False,
                divergence_type='',
                reason=reason,
                index=i
            ))
            last_signal_time = current_time

    return signals


def plot_chart(df: pd.DataFrame, signals: List[TradeSignal],
               stock_name: str, stock_code: str, date_str: str,
               save_path: str, last_close_price: float = None):
    """绑制分时图

    Args:
        last_close_price: 昨收价，用于计算相对昨收的涨跌幅
    """
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
    price_range = price.max() - price.min()

    # ========== 价格面板 ==========
    ax_price.plot(x, price, color='#1E90FF', linewidth=1.6, label='分时价格')

    # 如果有昨收价，绘制昨收价线
    if last_close_price is not None:
        ax_price.axhline(y=last_close_price, color='purple', linewidth=1.0,
                        linestyle='--', alpha=0.7, label=f"昨收 {last_close_price:.2f}")

    ax_price.axhline(y=df['开盘价'].iloc[0], color='gray', linewidth=0.8,
                     linestyle='--', alpha=0.5, label=f"开盘 {df['开盘价'].iloc[0]:.2f}")

    for s in signals:
        idx = s.index
        if s.signal_type == SignalType.SELL:
            color = '#FF4545'
            marker_color = 'darkred'
        else:
            color = '#32CD32'
            marker_color = 'darkgreen'

        ax_price.scatter(idx, s.price, color=color, s=300, marker='v' if s.signal_type == SignalType.SELL else '^',
                        zorder=6, edgecolors=marker_color, linewidths=2)

        label = f"{'卖出' if s.signal_type == SignalType.SELL else '买入'}\n{s.price:.2f}\n{s.reason}"

        offset = price_range * 0.12 if s.signal_type == SignalType.SELL else -price_range * 0.15
        ax_price.annotate(label, xy=(idx, s.price),
                         xytext=(idx, s.price + offset),
                         fontsize=8, color=color, fontweight='bold', ha='center',
                         arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax_price.set_ylabel('价格 (元)', fontsize=11)
    ax_price.set_ylim(price.min() - price_range * 0.2, price.max() + price_range * 0.35)
    ax_price.legend(loc='upper left')
    ax_price.grid(True, color='#DDDDDD')

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

    # 标注背离
    for s in signals:
        if s.divergence:
            idx = s.index
            y = df['DEA'].iloc[idx]
            color = '#FF4545' if s.divergence_type == '顶背离' else '#32CD32'
            ax_macd.scatter(idx, y, color=color, s=200, marker='*', zorder=6)

    # ========== 成交量面板 ==========
    for i in range(len(df)):
        color = '#FF4545' if (i > 0 and price.iloc[i] >= price.iloc[i - 1]) else '#00CD00'
        ax_vol.bar(x[i], df['成交量'].iloc[i], color=color, width=0.6, alpha=0.8)

    ax_vol.plot(x, df['成交量均线'], color='purple', linewidth=1.0, linestyle='--', alpha=0.7, label='成交量均线')
    ax_vol.set_ylabel('成交量', fontsize=11)
    ax_vol.legend(loc='upper left')

    # X轴
    key_ticks = [i for i, t in enumerate(times) if t in ['09:15', '09:25', '09:30', '10:00', '10:30', '11:00', '11:30',
                                                          '13:00', '13:30', '14:00', '14:30', '15:00']]
    ax_vol.set_xticks(key_ticks)
    ax_vol.set_xticklabels([times.iloc[i] for i in key_ticks], fontsize=9)
    ax_vol.set_xlim(-1, len(df))
    ax_vol.set_xlabel('时间', fontsize=11)

    # 集合竞价时段
    auction_start = next((i for i, t in enumerate(times) if t == '09:15'), None)
    auction_end = next((i for i, t in enumerate(times) if t == '09:25'), None)
    if auction_start is not None and auction_end is not None:
        ax_vol.axvspan(auction_start, auction_end, alpha=0.15, color='orange', label='集合竞价')

    # ========== 标题 ==========
    open_p = df['开盘价'].iloc[0]
    close_p = df['收盘价'].iloc[-1]
    change = close_p - open_p
    pct = change / open_p * 100

    buy_count = len([s for s in signals if s.signal_type == SignalType.BUY])
    sell_count = len([s for s in signals if s.signal_type == SignalType.SELL])

    fig.text(0.02, 0.975, f'{stock_name} [{stock_code}]  {date_str}  MACD分时策略v3',
             fontsize=14, fontweight='bold', va='top', transform=fig.transFigure)

    # 显示相对昨收的涨跌幅
    if last_close_price is not None:
        open_change = (open_p - last_close_price) / last_close_price * 100
        close_change = (close_p - last_close_price) / last_close_price * 100
        open_color = '#FF4545' if open_change >= 0 else '#32CD32'
        close_color = '#FF4545' if close_change >= 0 else '#32CD32'

        fig.text(0.02, 0.935,
                f'昨收 {last_close_price:.2f}  开盘 {open_p:.2f}({open_change:+.2f}%)  '
                f'收盘 {close_p:.2f}({close_change:+.2f}%)',
                fontsize=11, va='top', transform=fig.transFigure)
    else:
        change_color = '#FF4545' if change >= 0 else '#32CD32'
        fig.text(0.02, 0.935, f'开盘 {open_p:.2f} → 收盘 {close_p:.2f}  涨跌 {change:+.2f} ({pct:+.2f}%)',
                color=change_color, fontsize=11, va='top', transform=fig.transFigure)

    fig.text(0.02, 0.895, f"买入{buy_count}个 | 卖出{sell_count}个 | 策略:金叉死叉顺势",
             fontsize=9, va='top', transform=fig.transFigure)

    fig.text(0.02, 0.855, '买点:金叉+顺势(Histogram>0)+缩量 | 卖点:死叉+顺势(Histogram<0)+放量',
             fontsize=8, va='top', transform=fig.transFigure)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def print_signals(signals: List[TradeSignal]):
    """打印信号详情"""
    print()
    print('=' * 90)
    print('MACD分时策略v3 - 信号列表')
    print('=' * 90)
    print()
    print(f'{"时间":<10} {"信号":<4} {"价格":<8} {"量比":<6} {"原因"}')
    print('-' * 90)

    for s in signals:
        vol_ratio = s.volume / s.avg_volume if s.avg_volume > 0 else 1
        signal = '买入' if s.signal_type == SignalType.BUY else '卖出'

        print(f"{s.time.strftime('%H:%M'):<10} {signal:<4} {s.price:<8.2f} "
              f"{vol_ratio:<6.1f} {s.reason}")

    print()
    buy_count = len([s for s in signals if s.signal_type == SignalType.BUY])
    sell_count = len([s for s in signals if s.signal_type == SignalType.SELL])
    print(f'买入信号: {buy_count}个')
    print(f'卖出信号: {sell_count}个')


def find_csv_file(code: str, date_str: str = None) -> Tuple[str, str]:
    """在数据文件夹中查找CSV文件

    新数据路径: /Users/zanet/Desktop/自己开的项目/待完成-量化交易/定时获取全市场逐笔/data
    文件命名格式: {code}_{date}.csv 或 {market}{code}_{date}.csv

    Args:
        code: 股票代码
        date_str: 指定日期，若为None则查找该股票最新的CSV文件

    Returns:
        (csv_path, detected_date) - 文件路径和检测到的日期
    """
    # 新数据目录
    NEW_DATA_DIR = "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/回测数据/26逐笔"

    # 如果没有指定日期，查找最新的日期目录
    if date_str is None:
        # 查找最新的日期文件夹
        date_dirs = []
        if os.path.exists(NEW_DATA_DIR):
            date_dirs = [d for d in os.listdir(NEW_DATA_DIR)
                        if os.path.isdir(os.path.join(NEW_DATA_DIR, d)) and d.isdigit()]
        if date_dirs:
            date_dirs.sort(reverse=True)
            date_str = date_dirs[0]
        else:
            raise FileNotFoundError(f"数据目录为空或不存在: {NEW_DATA_DIR}")

    # 检查日期目录是否存在
    date_dir = os.path.join(NEW_DATA_DIR, date_str)
    stocks_dir = os.path.join(date_dir, "stocks")

    if not os.path.exists(stocks_dir):
        raise FileNotFoundError(f"数据目录不存在: {stocks_dir}")

    # 标准化股票代码
    code = code.strip()
    if code.startswith("60") or code.startswith("68"):
        # 沪市 - 尝试带前缀和不带前缀两种文件名
        patterns = [
            f"sh{code}_{date_str}.csv",
            f"{code}_{date_str}.csv"
        ]
    elif code.startswith("00") or code.startswith("30"):
        # 深市
        patterns = [
            f"sz{code}_{date_str}.csv",
            f"{code}_{date_str}.csv"
        ]
    else:
        patterns = [f"{code}_{date_str}.csv"]

    # 查找文件
    for pattern in patterns:
        csv_path = os.path.join(stocks_dir, pattern)
        if os.path.exists(csv_path):
            return csv_path, date_str

    raise FileNotFoundError(
        f"未找到股票 {code} 在 {date_str} 的数据文件\n"
        f"请检查文件是否存在，尝试的模式: {patterns}"
    )


def process_stock(csv_path: str, stock_name: str, stock_code: str, date_str: str, params: MacdParams, output_dir: str, target_date: str = None):
    """处理单只股票"""
    print(f"\n{'='*60}")
    print(f"处理: {stock_name} [{stock_code}] {date_str}")
    print(f"{'='*60}")

    print("读取数据(1分钟聚合)...")
    df = load_and_aggregate(csv_path, agg_seconds=60, target_date=target_date)
    print(f"聚合后: {len(df)}根K线")

    print("计算MACD指标...")
    df = calculate_macd(df, params)

    # 获取昨收价
    last_close_price = None
    if target_date:
        try:
            print("获取昨收价...")
            last_close_price, open_price = get_last_close_price(stock_code, target_date)
            stock_open_change = (open_price - last_close_price) / last_close_price * 100
            print(f"  昨收: {last_close_price}, 开盘: {open_price}, 开盘涨跌幅: {stock_open_change:+.2f}%")
        except Exception as e:
            print(f"  获取昨收价失败: {e}")

    print("检测信号...")
    signals = detect_signals(df, params)

    print_signals(signals)

    # 保存文件名用原始日期格式（target_date），显示用date_str
    file_date = target_date if target_date else date_str.replace('-', '')
    save_path = os.path.join(output_dir, f"MACD分时v3_{stock_code}_{file_date}.png")
    print(f"\n绘图: {save_path}")
    plot_chart(df, signals, stock_name, stock_code, date_str, save_path, last_close_price)
    print(f"已保存: {save_path}")


def get_random_stock(date_str: str = None) -> Tuple[str, str]:
    """从数据目录中随机选择一个股票

    Returns:
        (stock_code, date_str) - 股票代码和日期
    """
    NEW_DATA_DIR = "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/回测数据/26逐笔"

    # 如果没有指定日期，查找最新的日期目录
    if date_str is None:
        date_dirs = []
        if os.path.exists(NEW_DATA_DIR):
            date_dirs = [d for d in os.listdir(NEW_DATA_DIR)
                        if os.path.isdir(os.path.join(NEW_DATA_DIR, d)) and d.isdigit()]
        if date_dirs:
            date_dirs.sort(reverse=True)
            date_str = date_dirs[0]
        else:
            raise FileNotFoundError(f"数据目录为空或不存在: {NEW_DATA_DIR}")

    stocks_dir = os.path.join(NEW_DATA_DIR, date_str, "stocks")
    if not os.path.exists(stocks_dir):
        raise FileNotFoundError(f"股票目录不存在: {stocks_dir}")

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(stocks_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"没有找到任何股票数据文件: {stocks_dir}")

    # 随机选择一个文件
    random_file = random.choice(csv_files)
    # 文件名格式: sh600487_20260410.csv 或 600487_20260410.csv
    base_name = random_file.replace('.csv', '')

    # 提取股票代码（去掉市场前缀）
    if base_name.startswith('sh') or base_name.startswith('sz') or base_name.startswith('bj'):
        # 去掉市场前缀和日期后缀
        market_prefix = base_name[:2]
        code_with_date = base_name[2:]
        stock_code = code_with_date.replace(f'_{date_str}', '')
    else:
        # 直接去掉日期后缀
        stock_code = base_name.replace(f'_{date_str}', '')

    return stock_code, date_str


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MACD分时买卖策略 v3 - 零轴穿越",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python MACD分时策略v3.py --code 600487
  python MACD分时策略v3.py --code 600487 --date 20260410
  python MACD分时策略v3.py --random          # 随机选择一个股票
  python MACD分时策略v3.py --random --date 20260410
        """
    )
    parser.add_argument("--code", "-c", default=None, help="股票代码")
    parser.add_argument("--date", "-d", default=None, help="交易日期，格式YYYYMMDD")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--random", "-r", action="store_true", help="随机选择一个股票进行分析")
    args = parser.parse_args()

    # 处理随机模式
    if args.random:
        stock_code, detected_date = get_random_stock(args.date)
        date_str = detected_date
        csv_path, _ = find_csv_file(stock_code, date_str)
        print(f"[随机模式] 选中股票: {stock_code}")
        print(f"使用数据文件: {csv_path}")
        print(f"数据日期: {date_str}")
    elif args.code:
        csv_path, detected_date = find_csv_file(args.code, args.date)
        stock_code = args.code
        date_str = detected_date
        print(f"使用数据文件: {csv_path}")
        print(f"数据日期: {date_str}")
    else:
        print("错误: 请指定 --code 或 --random")
        print("示例: python MACD分时策略v3.py --code 600487")
        print("示例: python MACD分时策略v3.py --random")
        return
    year = date_str[:4]
    display_date = f"{year[-2:]}-{date_str[4:6]}-{date_str[6:8]}"

    output_dir = args.output
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "plots")

    params = MacdParams(
        short_period=5,
        long_period=34,
        dea_period=5,
        divergence_window=20,
        signal_interval=900,
        min_hist_change=0.015,
        min_volume_ratio=0.8,
        max_volume_ratio=1.2
    )

    print("=" * 60)
    print(f"MACD分时策略v3 - 金叉死叉顺势")
    print(f"参数: short={params.short_period}, long={params.long_period}, dea={params.dea_period}")
    print(f"信号间隔: {params.signal_interval}秒, 缩量阈值:{params.min_volume_ratio}, 放量阈值:{params.max_volume_ratio}")
    print("=" * 60)

    process_stock(csv_path, stock_code, stock_code, display_date, params, output_dir, target_date=detected_date)

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
