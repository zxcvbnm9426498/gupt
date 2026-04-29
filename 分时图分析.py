"""
分时图分析工具
========================================
功能：
1. 加载逐笔数据并聚合为分时数据
2. 计算MACD指标（DIFF、DEA、Histogram）
3. 绘制分时图 + MACD副图 + 成交量副图
4. 支持获取前收盘价计算真实涨跌幅

用法:
    python 分时图分析.py --code 600487
    python 分时图分析.py --code 600487 --date 20260410
    python 分时图分析.py --random          # 随机选择一个股票
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional
import argparse
import random

# 导入策略模块获取数据处理和MACD计算
try:
    from v2 import (
        SignalType, SignalLevel, TradeSignal, MacdParams,
        load_and_aggregate, calculate_macd, detect_signals
    )
except ModuleNotFoundError:
    from macd_strategy_v2 import (
        SignalType, SignalLevel, TradeSignal, MacdParams,
        load_and_aggregate, calculate_macd, detect_signals
    )

# 加载股票名单
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_LIST_FILE = os.path.join(SCRIPT_DIR, 'stock_list.csv')
_stock_name_map = None

def get_stock_name(code: str) -> str:
    """根据股票代码获取股票名称"""
    global _stock_name_map
    if _stock_name_map is None:
        if os.path.exists(STOCK_LIST_FILE):
            df = pd.read_csv(STOCK_LIST_FILE, dtype={'symbol': str})  # 确保symbol为字符串
            _stock_name_map = dict(zip(df['symbol'].str.zfill(6), df['name']))  # 补齐6位
        else:
            _stock_name_map = {}
    return _stock_name_map.get(code.zfill(6), code)  # 找不到就返回代码


def get_previous_close(code: str, date_str: str) -> Optional[float]:
    """获取前一个交易日的收盘价（使用Tushare）

    Args:
        code: 股票代码
        date_str: 当前日期，格式YYYYMMDD

    Returns:
        前收盘价，如果无法获取则返回None
    """
    import datetime
    import tushare as ts

    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    current_date = datetime.date(year, month, day)

    # 周一到周四的前一个交易日是前一天
    # 周五的前一个交易日是周四
    if current_date.weekday() == 0:  # 周一
        prev_date = current_date - datetime.timedelta(days=3)  # 上周五
    else:
        prev_date = current_date - datetime.timedelta(days=1)  # 前一天

    prev_date_str = prev_date.strftime('%Y%m%d')

    try:
        # Tushare需要带后缀的代码
        if code.startswith('60'):
            ts_code = f"{code}.SH"
        elif code.startswith('68'):
            ts_code = f"{code}.SH"
        elif code.startswith(('00', '30')):
            ts_code = f"{code}.SZ"
        else:
            ts_code = code

        df = ts.pro_bar(ts_code=ts_code, start_date=prev_date_str, end_date=prev_date_str, adj='qfq')
        if df is not None and len(df) > 0:
            return float(df['close'].iloc[0])
        return None
    except Exception as e:
        print(f"Tushare获取前收盘价失败: {e}")
        return None


def plot_chart(df: pd.DataFrame, signals: List[TradeSignal],
               stock_name: str, stock_code: str, date_str: str,
               save_path: str, strategy_label: str = "Histogram转折确认",
               detail_line: str = None, raw_date_str: str = None, t0_trades: list = None):
    """绑制分时图（价格 + MACD + 成交量）

    Args:
        df: 包含价格和MACD指标的DataFrame
        signals: 买卖信号列表
        stock_name: 股票名称
        stock_code: 股票代码
        date_str: 日期字符串
        save_path: 图片保存路径
        strategy_label: 策略标题
        detail_line: 额外说明
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
    open_price = df['开盘价'].iloc[0]
    price_range = price.max() - price.min()

    # 计算涨跌幅（以开盘价为基准）
    pct_change = ((price - open_price) / open_price * 100).values

    # ========== 价格面板（双Y轴：左侧价格，右侧涨跌幅） ==========
    ax_price.plot(x, price, color='#1E90FF', linewidth=1.6, label='分时价格')
    ax_price.axhline(y=open_price, color='gray', linewidth=0.8,
                     linestyle='--', alpha=0.5, label=f"开盘价 {open_price:.2f}")

    price_offset_base = price_range if price_range > 0 else max(open_price * 0.01, 0.01)

    # 只显示配对成功的T+0交易信号（满仓先卖后买）
    if t0_trades:
        for trade in t0_trades:
            sell_sig = trade['sell']
            buy_sig = trade['buy']
            pnl = trade['pnl']

            # 卖出信号（红色三角）
            ax_price.scatter(
                sell_sig.index,
                sell_sig.price,
                color='#FF4545',
                s=200,
                marker='v',
                zorder=6,
                edgecolors='darkred',
                linewidths=1.5
            )
            ax_price.annotate(
                f"卖出\n{sell_sig.price:.2f}",
                xy=(sell_sig.index, sell_sig.price),
                xytext=(sell_sig.index, sell_sig.price + price_offset_base * 0.08),
                fontsize=9,
                color='#FF4545',
                fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#FF4545', lw=1.0)
            )

            # 买入信号（绿色三角）
            ax_price.scatter(
                buy_sig.index,
                buy_sig.price,
                color='#32CD32',
                s=200,
                marker='^',
                zorder=6,
                edgecolors='darkgreen',
                linewidths=1.5
            )
            ax_price.annotate(
                f"买回\n{buy_sig.price:.2f}\n({pnl:+.0f}元)",
                xy=(buy_sig.index, buy_sig.price),
                xytext=(buy_sig.index, buy_sig.price - price_offset_base * 0.12),
                fontsize=9,
                color='#32CD32',
                fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#32CD32', lw=1.0)
            )
    else:
        # 没有配对交易时，显示所有信号
        for signal in signals:
            marker = '^' if signal.signal_type == SignalType.BUY else 'v'
            color = '#32CD32' if signal.signal_type == SignalType.BUY else '#FF4545'
            edge_color = 'darkgreen' if signal.signal_type == SignalType.BUY else 'darkred'
            label = '买入' if signal.signal_type == SignalType.BUY else '卖出'
            level = getattr(signal.signal_level, 'value', '')
            offset = -price_offset_base * 0.10 if signal.signal_type == SignalType.BUY else price_offset_base * 0.10

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
                f"{label}[{level}]\n{signal.price:.2f}\n{signal.reason}",
                xy=(signal.index, signal.price),
                xytext=(signal.index, signal.price + offset),
                fontsize=8,
                color=color,
                fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0)
            )

    ax_price.set_ylabel('价格 (元)', fontsize=11, color='#1E90FF')
    ax_price.set_ylim(price.min() - price_range * 0.2, price.max() + price_range * 0.35)
    ax_price.legend(loc='upper left')
    ax_price.grid(True, color='#DDDDDD')

    # 主图横坐标：时间（11:30和13:00合并显示为"11:30/13:00"，从9:30开始）
    all_key_times = ['09:30', '10:00', '10:30', '11:30', '13:00', '13:30', '14:00', '14:30', '15:00']

    # 找到所有时间对应的索引位置
    time_to_idx = {t: i for i, t in enumerate(times)}

    display_ticks = []
    tick_labels = []
    skip_next = False

    for idx, t in enumerate(all_key_times):
        if skip_next:
            skip_next = False
            continue
        if t not in time_to_idx:
            continue
        pos = time_to_idx[t]
        if t == '11:30' and idx + 1 < len(all_key_times) and all_key_times[idx + 1] == '13:00':
            if '13:00' in time_to_idx:
                display_ticks.append(pos)
                tick_labels.append('11:30/13:00')
                skip_next = True  # 跳过下一次循环（13:00）
                continue
        display_ticks.append(pos)
        tick_labels.append(t)

    ax_price.set_xticks(display_ticks)
    ax_price.set_xticklabels(tick_labels, fontsize=9)
    # 设置xlim使第一个和最后一个tick分别顶在左右两侧
    first_tick = display_ticks[0]
    last_tick = display_ticks[-1]
    ax_price.set_xlim(first_tick, last_tick)
    ax_price.set_xlabel('时间', fontsize=11)

    # 右侧Y轴：涨跌幅（只显示刻度，不标注数字）
    ax_pct = ax_price.twinx()
    ax_pct.set_ylabel('涨跌幅 (%)', fontsize=11)
    ax_pct.set_ylim(pct_change.min() - 0.5, pct_change.max() + 0.5)
    # 不需要标注数字，让右侧Y轴只显示刻度即可

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
    ax_macd.set_xticks([])  # 取消横坐标时间显示

    # 只在MACD上显示配对的T+0信号
    if t0_trades:
        for trade in t0_trades:
            sell_sig = trade['sell']
            buy_sig = trade['buy']
            ax_macd.scatter(sell_sig.index, df['DEA'].iloc[sell_sig.index],
                           color='#FF4545', s=80, marker='v', zorder=6)
            ax_macd.scatter(buy_sig.index, df['DEA'].iloc[buy_sig.index],
                           color='#32CD32', s=80, marker='^', zorder=6)
    else:
        for signal in signals:
            signal_color = '#32CD32' if signal.signal_type == SignalType.BUY else '#FF4545'
            signal_marker = '^' if signal.signal_type == SignalType.BUY else 'v'
            ax_macd.scatter(
                signal.index,
                df['DEA'].iloc[signal.index],
                color=signal_color,
                s=80,
                marker=signal_marker,
                zorder=6
            )

    # ========== 成交量面板 ==========
    for i in range(len(df)):
        color = '#FF4545' if (i > 0 and price.iloc[i] >= price.iloc[i - 1]) else '#00CD00'
        ax_vol.bar(x[i], df['成交量'].iloc[i], color=color, width=0.6, alpha=0.8)

    ax_vol.plot(x, df['成交量均线'], color='purple', linewidth=1.0, linestyle='--', alpha=0.7, label='成交量均线')
    ax_vol.set_ylabel('成交量', fontsize=11)
    ax_vol.legend(loc='upper left')
    ax_vol.set_xticks([])  # 取消横坐标时间显示

    # ========== 标题 ==========
    open_p = df['开盘价'].iloc[0]
    close_p = df['收盘价'].iloc[-1]

    # 使用前收盘价计算涨跌幅（与软件一致）
    if raw_date_str is None:
        raw_date_str = date_str.replace('-', '')
    prev_close = get_previous_close(stock_code, raw_date_str)
    if prev_close is None:
        prev_close = open_p  # 如果获取不到前收盘，用开盘价代替
    change = close_p - prev_close
    pct = change / prev_close * 100

    change_color = '#FF4545' if change >= 0 else '#32CD32'

    # 计算T+0总收益
    t0_profit = sum(t['pnl'] for t in t0_trades) if t0_trades else 0
    base_profit = change * 500  # 底仓500股收益
    total_profit = base_profit + t0_profit

    # 标题行（居中）：股票代码+股票名称+日期
    stock_name = get_stock_name(stock_code)
    fig.text(0.35, 0.94, f'{stock_code} {stock_name} {date_str}',
             fontsize=14, fontweight='bold', va='top', transform=fig.transFigure)

    # 开盘收盘涨跌信息（居中偏右）使用前收盘计算涨跌幅
    fig.text(0.68, 0.94, f'前收 {prev_close:.2f}  开盘 {open_p:.2f} → 收盘 {close_p:.2f}  涨跌 {change:+.2f} ({pct:+.2f}%)',
             color=change_color, fontsize=11, va='top', transform=fig.transFigure)

    # T+0交易信息
    if t0_trades:
        fig.text(0.35, 0.91, f'满仓做T: {len(t0_trades)}笔 | T+0收益: {t0_profit:+.0f}元 | 底仓收益: {base_profit:+.0f}元 | 总收益: {total_profit:+.0f}元',
                 fontsize=10, va='top', transform=fig.transFigure)
    else:
        fig.text(0.35, 0.91, f'策略: {strategy_label}', fontsize=10, va='top', transform=fig.transFigure)
    if detail_line:
        fig.text(0.35, 0.88, detail_line, fontsize=9, va='top', transform=fig.transFigure)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="分时图分析工具 - 绘制分时图+MACD+成交量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 分时图分析.py --code 600487
  python 分时图分析.py --code 600487 --date 20260410
  python 分时图分析.py --random          # 随机选择一个股票
  python 分时图分析.py --random --date 20260410
        """
    )
    parser.add_argument(
        "--code", "-c",
        default=None,
        help="股票代码，如 600487、000037、300342"
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="交易日期，格式YYYYMMDD，默认为最新日期"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录，默认使用项目目录下的 plots 文件夹"
    )
    parser.add_argument(
        "--random", "-r",
        action="store_true",
        help="随机选择一个股票进行分析"
    )
    return parser.parse_args()


def find_csv_file(code: str, date_str: str = None) -> Tuple[str, str]:
    """在数据文件夹中查找CSV文件"""
    NEW_DATA_DIR = "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/回测数据/26逐笔"

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
        raise FileNotFoundError(f"数据目录不存在: {stocks_dir}")

    code = code.strip()
    if code.startswith("60") or code.startswith("68"):
        patterns = [f"sh{code}_{date_str}.csv", f"{code}_{date_str}.csv"]
    elif code.startswith("00") or code.startswith("30"):
        patterns = [f"sz{code}_{date_str}.csv", f"{code}_{date_str}.csv"]
    else:
        patterns = [f"{code}_{date_str}.csv"]

    for pattern in patterns:
        csv_path = os.path.join(stocks_dir, pattern)
        if os.path.exists(csv_path):
            return csv_path, date_str

    raise FileNotFoundError(f"未找到股票 {code} 在 {date_str} 的数据文件")


def get_random_stock(date_str: str = None) -> Tuple[str, str]:
    """从数据目录中随机选择一个股票"""
    NEW_DATA_DIR = "/Users/zanet/Desktop/自己开的项目/待完成-量化交易/回测数据/26逐笔"

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

    csv_files = [f for f in os.listdir(stocks_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"没有找到任何股票数据文件: {stocks_dir}")

    random_file = random.choice(csv_files)
    base_name = random_file.replace('.csv', '')

    if base_name.startswith('sh') or base_name.startswith('sz') or base_name.startswith('bj'):
        stock_code = base_name[2:].replace(f'_{date_str}', '')
    else:
        stock_code = base_name.replace(f'_{date_str}', '')

    return stock_code, date_str


def main():
    """主函数"""
    args = parse_args()

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
        print("示例: python 分时图分析.py --code 600487")
        print("示例: python 分时图分析.py --random")
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
        dea_period=5
    )

    print("=" * 60)
    print(f"分时图分析 - 股票 {stock_code}")
    print(f"MACD参数: short={params.short_period}, long={params.long_period}, dea={params.dea_period}")
    print("=" * 60)

    # 1. 加载数据
    print("读取数据(1分钟聚合)...")
    df = load_and_aggregate(csv_path, agg_seconds=60, target_date=date_str)
    print(f"聚合后: {len(df)}根K线")

    # 2. 计算MACD
    print("计算MACD指标...")
    df = calculate_macd(df, params)

    # 3. 生成买卖信号
    print("检测买卖信号...")
    signals = detect_signals(df, params)
    buy_count = sum(1 for signal in signals if signal.signal_type == SignalType.BUY)
    sell_count = sum(1 for signal in signals if signal.signal_type == SignalType.SELL)
    print(f"检测完成: 买入信号 {buy_count} 个, 卖出信号 {sell_count} 个")

    # 4. 计算配对的T+0交易（满仓500股，先卖后买）
    t0_trades = []
    sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
    buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]

    sell_idx = 0
    buy_idx = 0

    # 配对T+0交易: 每个卖出配对之后最近的买入（确保先卖后买）
    while sell_idx < len(sell_signals):
        sell_sig = sell_signals[sell_idx]

        # 找卖出之后的下一个买入
        while buy_idx < len(buy_signals) and buy_signals[buy_idx].time <= sell_sig.time:
            buy_idx += 1

        if buy_idx >= len(buy_signals):
            break

        buy_sig = buy_signals[buy_idx]

        # 配对成功: 先卖后买，每次操作100股
        pnl = (sell_sig.price - buy_sig.price) * 100
        t0_trades.append({'sell': sell_sig, 'buy': buy_sig, 'pnl': pnl})

        sell_idx += 1
        buy_idx += 1

    t0_profit = sum(t['pnl'] for t in t0_trades)
    print(f"T+0配对交易: {len(t0_trades)} 笔, 收益 {t0_profit:+.2f}元")

    # 5. 绘图（只显示配对成功的T+0信号）
    save_path = os.path.join(output_dir, f"分时图_{stock_code}_{date_str}.png")
    print(f"\n绘图: {save_path}")
    plot_chart(df, signals, stock_code, stock_code, display_date, save_path, raw_date_str=date_str, t0_trades=t0_trades)
    print(f"已保存: {save_path}")

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
