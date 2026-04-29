"""
MACD分时策略 v2 - 核心策略模块
========================================
提供MACD指标计算和信号检测的核心逻辑

核心规律（右侧确认）：
1. 买入：Histogram由下降转为上升（curr<0 且 next>curr）
2. 卖出：Histogram由上升转为下降（curr>0 且 next<curr）

特点：
- 使用1分钟K线聚合（标准分时周期）
- 右侧确认：在转折发生后立即标记
- 信号间隔：5分钟

参数：
- 短期EMA: 5分钟
- 长期EMA: 34分钟
- DEA平滑: 5分钟

用法:
    from macd_strategy_v2 import load_and_aggregate, calculate_macd, detect_signals, MacdParams
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"


class SignalLevel(Enum):
    """信号强度"""
    STRONG = "强信号"      # 转折确认
    NORMAL = "普通信号"    # 一般情况
    WEAK = "弱信号"        # 需要谨慎


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
    divergence: bool       # 是否存在背离
    divergence_type: str   # '顶背离'/'底背离'/''
    reason: str
    index: int


@dataclass
class MacdParams:
    """MACD参数"""
    short_period: int = 5
    long_period: int = 34
    dea_period: int = 5
    divergence_window: int = 20
    signal_interval: int = 600  # 信号间隔10分钟（减少频繁信号）
    min_hist_change: float = 0.015  # Histogram最小变化幅度（过滤小波动）


def load_and_aggregate(csv_path: str, agg_seconds: int = 60, target_date: str = None) -> pd.DataFrame:
    """加载并聚合逐笔数据

    Args:
        csv_path: CSV文件路径
        agg_seconds: 聚合周期（秒），默认60秒(1分钟)
        target_date: 目标日期，格式YYYYMMDD，若为None则加载所有数据
    """
    df = pd.read_csv(csv_path)
    df['时间'] = pd.to_datetime(df['时间'])

    # 过滤只保留目标日期的数据
    if target_date:
        year = target_date[:4]
        month = target_date[4:6]
        day = target_date[6:8]
        target_date_str = f"{year}-{month}-{day}"
        df = df[df['时间'].dt.strftime('%Y-%m-%d') == target_date_str]
        if len(df) == 0:
            raise ValueError(f"CSV中没有 {target_date} 的数据")

    if agg_seconds == 1:
        # 逐笔级别：每笔数据作为一行
        # 成交量是每笔的，需要保留
        df.rename(columns={'价格': '收盘价'}, inplace=True)
        df = df[['时间', '成交量', '收盘价']].copy()
        # 添加必要的列（用于计算指标）
        df['开盘价'] = df['收盘价']  # 逐笔数据中，每笔价格既是开盘也是收盘
        df['最高价'] = df['收盘价']
        df['最低价'] = df['收盘价']
        df['成交量均线'] = df['成交量'].rolling(window=100, min_periods=1).mean()
        return df

    # 按指定周期聚合
    df['周期'] = df['时间'].dt.floor(f'{agg_seconds}s')
    agg = df.groupby('周期').agg({
        '价格': ['first', 'last', 'max', 'min'],
        '成交量': 'sum'
    }).reset_index()
    agg.columns = ['时间', '开盘价', '收盘价', '最高价', '最低价', '成交量']
    agg = agg.sort_values('时间').reset_index(drop=True)
    return agg


def calculate_macd(df: pd.DataFrame, params: MacdParams) -> pd.DataFrame:
    """计算MACD指标

    Args:
        df: 包含收盘价和成交量的DataFrame
        params: MACD参数

    Returns:
        添加了MACD相关列的DataFrame
    """
    close = df['收盘价']
    df['EMA_Short'] = close.ewm(span=params.short_period, adjust=False).mean()
    df['EMA_Long'] = close.ewm(span=params.long_period, adjust=False).mean()
    df['DIFF'] = df['EMA_Short'] - df['EMA_Long']
    df['DEA'] = df['DIFF'].ewm(span=params.dea_period, adjust=False).mean()
    df['Histogram'] = (df['DIFF'] - df['DEA']) * 2

    # 计算成交量均线
    df['成交量均线'] = df['成交量'].rolling(window=20, min_periods=1).mean()

    return df


def detect_divergence(df: pd.DataFrame, idx: int, params: MacdParams) -> Tuple[bool, str]:
    """检测背离

    Args:
        df: 包含MACD指标的DataFrame
        idx: 当前索引位置
        params: MACD参数

    Returns:
        (是否存在背离, 背离类型)
    """
    if idx < params.divergence_window:
        return False, ''

    window_data = df.iloc[idx - params.divergence_window:idx + 1]

    price_current = df['收盘价'].iloc[idx]
    dea_current = df['DEA'].iloc[idx]
    hist_current = df['Histogram'].iloc[idx]

    # 窗口内的极值
    price_max = window_data['收盘价'].max()
    price_min = window_data['收盘价'].min()
    hist_max = window_data['Histogram'].max()
    hist_min = window_data['Histogram'].min()

    # 顶背离：价格创新高，但Histogram没创新高（价格创新高是出货信号）
    if price_current >= price_max * 0.999 and hist_current < hist_max * 0.9:
        return True, '顶背离'

    # 底背离：价格创新低，但Histogram没创新低（价格新低是抄底机会）
    if price_current <= price_min * 1.001 and hist_current > hist_min * 1.1:
        return True, '底背离'

    return False, ''


def detect_signals(df: pd.DataFrame, params: MacdParams) -> List[TradeSignal]:
    """检测买卖信号 - 转折点检测

    买入：Histogram由下降转为上升（curr<0 且 next>curr）
    卖出：Histogram由上升转为下降（curr>0 且 next<curr）
    不考虑持仓状态，标记所有信号点

    Args:
        df: 包含MACD指标的DataFrame
        params: MACD参数

    Returns:
        交易信号列表
    """
    signals = []
    last_signal_time = None

    for i in range(1, len(df) - 1):
        prev_hist = df['Histogram'].iloc[i - 1]
        curr_hist = df['Histogram'].iloc[i]
        next_hist = df['Histogram'].iloc[i + 1]

        current_time = df['时间'].iloc[i]
        current_price = df['收盘价'].iloc[i]
        current_volume = df['成交量'].iloc[i]
        avg_volume = df['成交量均线'].iloc[i]

        # 信号间隔
        can_signal = (last_signal_time is None or
                     (current_time - last_signal_time).total_seconds() > params.signal_interval)

        if not can_signal:
            continue

        # ========== 右侧确认：Histogram转折点检测 ==========
        # 买入：Histogram由下降转为上升（curr<0 且 next>curr）
        # 卖出：Histogram由上升转为下降（curr>0 且 next<curr）
        if curr_hist < 0 and next_hist > curr_hist:
            hist_change = next_hist - curr_hist
            level = SignalLevel.STRONG
            reason = f"转折向上({curr_hist:.4f}→{next_hist:.4f})"

            signals.append(TradeSignal(
                time=current_time,
                signal_type=SignalType.BUY,
                price=current_price,
                histogram_prev=prev_hist,
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

        elif curr_hist > 0 and next_hist < curr_hist:
            hist_change = curr_hist - next_hist
            level = SignalLevel.STRONG
            reason = f"转折向下({curr_hist:.4f}→{next_hist:.4f})"

            signals.append(TradeSignal(
                time=current_time,
                signal_type=SignalType.SELL,
                price=current_price,
                histogram_prev=prev_hist,
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_LIST_FILE = os.path.join(SCRIPT_DIR, 'stock_list.csv')
DATA_DIR = "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/回测数据/26逐笔"
_stock_name_map = None


def get_stock_name(code: str) -> str:
    """根据股票代码获取股票名称。"""
    global _stock_name_map

    if _stock_name_map is None:
        if os.path.exists(STOCK_LIST_FILE):
            stock_df = pd.read_csv(STOCK_LIST_FILE, dtype={'symbol': str})
            _stock_name_map = dict(zip(stock_df['symbol'].str.zfill(6), stock_df['name']))
        else:
            _stock_name_map = {}

    return _stock_name_map.get(code.zfill(6), code)


def plot_chart(
    df: pd.DataFrame,
    signals: List[TradeSignal],
    stock_code: str,
    date_str: str,
    save_path: str
) -> None:
    """绘制带买卖点标记的 v2 分时图。"""
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

    ax_price.plot(x, price, color='#1E90FF', linewidth=1.6, label='分时价格')
    ax_price.axhline(
        y=open_price,
        color='gray',
        linewidth=0.8,
        linestyle='--',
        alpha=0.5,
        label=f"开盘价 {open_price:.2f}"
    )

    for signal in signals:
        is_buy = signal.signal_type == SignalType.BUY
        color = '#32CD32' if is_buy else '#FF4545'
        edge_color = 'darkgreen' if is_buy else 'darkred'
        marker = '^' if is_buy else 'v'
        label = '买入' if is_buy else '卖出'
        offset = -price_offset_base * 0.10 if is_buy else price_offset_base * 0.10

        ax_price.scatter(
            signal.index,
            signal.price,
            color=color,
            s=180,
            marker=marker,
            zorder=6,
            edgecolors=edge_color,
            linewidths=1.5
        )
        ax_price.annotate(
            f"{label}\n{signal.price:.2f}\n{signal.reason}",
            xy=(signal.index, signal.price),
            xytext=(signal.index, signal.price + offset),
            fontsize=8,
            color=color,
            fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0)
        )

    ax_price.set_ylabel('价格 (元)', fontsize=11, color='#1E90FF')
    ax_price.set_ylim(price.min() - price_offset_base * 0.2, price.max() + price_offset_base * 0.35)
    ax_price.legend(loc='upper left')
    ax_price.grid(True, color='#DDDDDD')

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

    ax_pct = ax_price.twinx()
    pct_change = ((price - open_price) / open_price * 100).values
    ax_pct.set_ylabel('涨跌幅 (%)', fontsize=11)
    ax_pct.set_ylim(pct_change.min() - 0.5, pct_change.max() + 0.5)

    ax_macd.plot(x, df['DIFF'], color='#1E90FF', linewidth=1.3, label='DIFF(5)')
    ax_macd.plot(x, df['DEA'], color='#FF8C00', linewidth=1.0, label='DEA(5)')

    for i in range(len(df)):
        color = '#FF4545' if df['Histogram'].iloc[i] >= 0 else '#32CD32'
        ax_macd.bar(x[i], df['Histogram'].iloc[i], color=color, width=0.6, alpha=0.7)

    for signal in signals:
        ax_macd.scatter(
            signal.index,
            df['DEA'].iloc[signal.index],
            color='#32CD32' if signal.signal_type == SignalType.BUY else '#FF4545',
            s=80,
            marker='^' if signal.signal_type == SignalType.BUY else 'v',
            zorder=6
        )

    ax_macd.axhline(y=0, color='gray', linewidth=0.8)
    ax_macd.set_ylabel('MACD', fontsize=11)
    ax_macd.legend(loc='upper left')
    ax_macd.grid(True, color='#DDDDDD')
    ax_macd.set_xticks([])

    for i in range(len(df)):
        color = '#FF4545' if (i > 0 and price.iloc[i] >= price.iloc[i - 1]) else '#00CD00'
        ax_vol.bar(x[i], df['成交量'].iloc[i], color=color, width=0.6, alpha=0.8)

    ax_vol.plot(
        x,
        df['成交量均线'],
        color='purple',
        linewidth=1.0,
        linestyle='--',
        alpha=0.7,
        label='成交量均线'
    )
    ax_vol.set_ylabel('成交量', fontsize=11)
    ax_vol.legend(loc='upper left')
    ax_vol.set_xticks([])

    stock_name = get_stock_name(stock_code)
    close_price = df['收盘价'].iloc[-1]
    change = close_price - open_price
    pct = change / open_price * 100
    change_color = '#FF4545' if change >= 0 else '#32CD32'
    buy_count = sum(1 for signal in signals if signal.signal_type == SignalType.BUY)
    sell_count = sum(1 for signal in signals if signal.signal_type == SignalType.SELL)

    fig.text(0.35, 0.94, f'{stock_code} {stock_name} {date_str}', fontsize=14, fontweight='bold', va='top')
    fig.text(
        0.68,
        0.94,
        f'开盘 {open_price:.2f} → 收盘 {close_price:.2f}  涨跌 {change:+.2f} ({pct:+.2f}%)',
        color=change_color,
        fontsize=11,
        va='top'
    )
    fig.text(0.35, 0.91, f'买点 {buy_count} 个 | 卖点 {sell_count} 个 | 策略: Histogram转折确认', fontsize=10, va='top')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def print_signals(signals: List[TradeSignal]) -> None:
    """打印信号详情。"""
    print()
    print('=' * 90)
    print('MACD分时策略v2 - 信号列表')
    print('=' * 90)
    print()
    print(f'{"时间":<10} {"信号":<4} {"价格":<8} {"量比":<6} {"原因"}')
    print('-' * 90)

    for signal in signals:
        vol_ratio = signal.volume / signal.avg_volume if signal.avg_volume > 0 else 1
        signal_label = '买入' if signal.signal_type == SignalType.BUY else '卖出'
        print(
            f"{signal.time.strftime('%H:%M'):<10} {signal_label:<4} "
            f"{signal.price:<8.2f} {vol_ratio:<6.1f} {signal.reason}"
        )

    print()
    print(f"买入信号: {sum(1 for signal in signals if signal.signal_type == SignalType.BUY)}个")
    print(f"卖出信号: {sum(1 for signal in signals if signal.signal_type == SignalType.SELL)}个")


def find_csv_file(code: str, date_str: str = None) -> Tuple[str, str]:
    """在逐笔数据目录中查找股票 CSV。"""
    if date_str is None:
        date_dirs = []
        if os.path.exists(DATA_DIR):
            date_dirs = [
                directory for directory in os.listdir(DATA_DIR)
                if os.path.isdir(os.path.join(DATA_DIR, directory)) and directory.isdigit()
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

    raise FileNotFoundError(f"未找到股票 {code} 在 {date_str} 的数据文件，尝试的模式: {patterns}")


def get_random_stock(date_str: str = None) -> Tuple[str, str]:
    """从数据目录中随机选择一只股票。"""
    if date_str is None:
        date_dirs = []
        if os.path.exists(DATA_DIR):
            date_dirs = [
                directory for directory in os.listdir(DATA_DIR)
                if os.path.isdir(os.path.join(DATA_DIR, directory)) and directory.isdigit()
            ]
        if not date_dirs:
            raise FileNotFoundError(f"数据目录为空或不存在: {DATA_DIR}")
        date_dirs.sort(reverse=True)
        date_str = date_dirs[0]

    stocks_dir = os.path.join(DATA_DIR, date_str, "stocks")
    if not os.path.exists(stocks_dir):
        raise FileNotFoundError(f"股票目录不存在: {stocks_dir}")

    csv_files = [file_name for file_name in os.listdir(stocks_dir) if file_name.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"没有找到任何股票数据文件: {stocks_dir}")

    base_name = random.choice(csv_files).replace('.csv', '')
    if base_name.startswith(('sh', 'sz', 'bj')):
        return base_name[2:].replace(f'_{date_str}', ''), date_str
    return base_name.replace(f'_{date_str}', ''), date_str


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="MACD分时策略 v2 - Histogram转折确认",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python v2.py --code 600487
  python v2.py --code 600487 --date 20260410
  python v2.py --random
        """
    )
    parser.add_argument("--code", "-c", default=None, help="股票代码")
    parser.add_argument("--date", "-d", default=None, help="交易日期，格式YYYYMMDD")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--random", "-r", action="store_true", help="随机选择一个股票进行分析")
    return parser.parse_args()


def main() -> None:
    """命令行主入口。"""
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
        print("示例: python v2.py --code 600487")
        print("示例: python v2.py --random")
        return

    print(f"使用数据文件: {csv_path}")
    print(f"数据日期: {detected_date}")

    output_dir = args.output or os.path.join(SCRIPT_DIR, "plots")
    display_date = f"{detected_date[:4]}-{detected_date[4:6]}-{detected_date[6:8]}"
    params = MacdParams(short_period=5, long_period=34, dea_period=5, divergence_window=20, signal_interval=600)

    print("=" * 60)
    print("MACD分时策略v2 - Histogram转折确认")
    print(f"参数: short={params.short_period}, long={params.long_period}, dea={params.dea_period}")
    print(f"信号间隔: {params.signal_interval}秒")
    print("=" * 60)

    print("读取数据(1分钟聚合)...")
    df = load_and_aggregate(csv_path, agg_seconds=60, target_date=detected_date)
    print(f"聚合后: {len(df)}根K线")

    print("计算MACD指标...")
    df = calculate_macd(df, params)

    print("检测信号...")
    signals = detect_signals(df, params)
    print_signals(signals)

    save_path = os.path.join(output_dir, f"MACD分时v2_{stock_code}_{detected_date}.png")
    print(f"\n绘图: {save_path}")
    plot_chart(df, signals, stock_code, display_date, save_path)
    print(f"已保存: {save_path}")


if __name__ == '__main__':
    main()
