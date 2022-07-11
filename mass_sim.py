import pandas as pd
import datetime as dt
import utils
import instrument
import math
import numpy as np


def MBB_BB_OS_PX(df):
    conds = [
        (df['MBB_PX'] == 1) | (df['BB_OS_PX'] == 1),
        (df['MBB_PX'] == -1) | (df['BB_OS_PX'] == -1)
    ]

    choices = [1, -1]

    df['MBB_BB_OS_PX'] = np.select(conds, choices)


def MBB_PX(df):
    conds = [
        (df['MBB_SIGNAL'] == 1) & (df['PX'] == 1),
        (df['MBB_SIGNAL'] == -1) & (df['PX'] == -1)
    ]

    choices = [1, -1]

    df['MBB_PX'] = np.select(conds, choices)


def BB_OS_PX(df):
    conds = [
        (df['PREV_BB_SIGNAL'] == 1) & (
            df['PREV_OS_SIGNAL'] == 1) & (df['PX'] == 1),
        (df['PREV_BB_SIGNAL'] == -1) & (df['PREV_OS_SIGNAL'] == -1) & (df['PX'] == -1)
    ]

    choices = [1, -1]

    df['BB_OS_PX'] = np.select(conds, choices)


def Get_MBB_Signal(row):
    if row.DIRECTION == 1 and row.PEN_DIRECTION == -1 and row.TRE_DIRECTION == -1 and row.TRE_PERC_EIGHTY > row.MA_TP:
        if row.PREV_BODY_LOW < row.MA_TP and row.PREV_BODY_HIGH > row.MA_TP:
            return 1
    if row.DIRECTION == -1 and row.PEN_DIRECTION == 1 and row.TRE_DIRECTION == 1 and row.TRE_PERC_EIGHTY < row.MA_TP:
        if row.PREV_BODY_HIGH > row.MA_TP and row.PREV_BODY_LOW < row.MA_TP:
            return -1
    return 0


def Get_BB_Signal(row):
    if row.mid_h > row.BB_UPPER and row.mid_c > row.mid_o:
        return -1
    if row.mid_l < row.BB_LOWER and row.mid_c < row.mid_o:
        return 1
    return 0


def Get_SO_Signal(df):
    minone = df['K'].round(1).gt(stoch_K)
    one = df['K'].round(1).lt(stoch_D)

    df.loc[one, "OS_SIGNAL"] = 1
    df.loc[minone, "OS_SIGNAL"] = -1


def apply_px(row):
    if row.PREV_BODY_PERC < PREV_PERCENTAGE and row.BODY_PERC > row.PREV_BODY_PERC:
        if row.DIRECTION == 1 and (row.mid_h > row.PREV_BODY_HIGH and row.mid_c > row.PREV_BODY_CLOSE):
            return 1
        elif row.DIRECTION == -1 and (row.mid_l < row.PREV_BODY_LOW and row.mid_c < row.PREV_BODY_CLOSE):
            return -1
    return 0


def apply_stoploss(df):
    conds = [
        (df['DIRECTION'] == 1),
        (df['DIRECTION'] == -1)
    ]

    choices = [(df.mid_h - (df.mid_h - df.mid_l)*SL_ENTRY_TP[0]), (
        df.mid_l + (df.mid_h - df.mid_l)*SL_ENTRY_TP[0])]

    df['STOPLOSS'] = np.select(conds, choices)


def apply_entry(df):
    conds = [
        (df['DIRECTION'] == 1),
        (df['DIRECTION'] == -1)
    ]

    choices = [(df.mid_h - (df.mid_h - df.mid_l)*SL_ENTRY_TP[1]), (
        df.mid_l + (df.mid_h - df.mid_l)*SL_ENTRY_TP[1])]

    df['ENTRY'] = np.select(conds, choices)


def apply_takeprofit(df):
    conds = [
        (df['DIRECTION'] == 1),
        (df['DIRECTION'] == -1)
    ]

    choices = [(
        df.mid_h - (df.mid_h - df.mid_l)*SL_ENTRY_TP[2]), (
        df.mid_l + (df.mid_h - df.mid_l)*SL_ENTRY_TP[2])]

    df['TAKEPROFIT'] = np.select(conds, choices)


def get_entry_limit(df):
    df['ENTRY'] = np.where((df.MBB_BB_OS_PX != 0), df.ENTRY, 0)


def get_stop_loss(df):
    df['STOPLOSS'] = np.where((df.MBB_BB_OS_PX != 0), df.STOPLOSS, 0)


def get_take_profit(df):
    df['TAKEPROFIT'] = np.where((df.MBB_BB_OS_PX != 0), df.TAKEPROFIT, 0)


def triggered(direction, current_price, signal_price):
    if direction == 1 and current_price > signal_price:
        return True
    elif direction == -1 and current_price < signal_price:
        return True
    return False


def end_hit_calc(direction, pipLoc, price, start_price):

    fraction = abs(price - start_price) / math.pow(10, pipLoc)

    if direction == 1 and price >= start_price:
        return fraction
    elif direction == 1 and price < start_price:
        return -fraction
    elif direction == -1 and price <= start_price:
        return fraction
    elif direction == -1 and price > start_price:
        return -fraction


def process_buy(TP, SL, ask_prices, bid_prices, entry_price, pipLoc):
    for index, price in enumerate(ask_prices):
        if triggered(1, price, entry_price) == True:
            for live_price in bid_prices[index:]:
                if live_price >= TP:
                    return ((live_price - entry_price) / math.pow(10, pipLoc))
                elif live_price <= SL:
                    return ((live_price - entry_price) / math.pow(10, pipLoc))
            return end_hit_calc(1, pipLoc, live_price, entry_price)
    return 0.0


def process_sell(TP, SL, ask_prices, bid_prices, entry_price, pipLoc):
    for index, price in enumerate(bid_prices):
        if triggered(-1, price, entry_price) == True:
            for live_price in ask_prices[index:]:
                if live_price <= TP:
                    return abs((live_price - entry_price) / math.pow(10, pipLoc))
                elif live_price >= SL:
                    return ((entry_price - live_price) / math.pow(10, pipLoc))
            return end_hit_calc(-1, pipLoc, live_price, entry_price)
    return 0.0


def get_test_pairs(pair_str):
    existing_pairs = instrument.Instrument.get_instruments_dict().keys()
    pairs = pair_str.split(",")

    test_list = []
    for p1 in pairs:
        for p2 in pairs:
            p = f"{p1}_{p2}"
            if p in existing_pairs:
                test_list.append(p)

    return test_list


def get_trades_df(df_raw):
    df = df_raw.copy()
    df.reset_index(drop=True, inplace=True)
    df['BODY_PERC'] = abs(df.mid_c - df.mid_o) / (df.mid_h - df.mid_l)
    df['DIRECTION'] = df.mid_c - df.mid_o
    df['DIRECTION'] = df['DIRECTION'].apply(lambda x: 1 if x >= 0 else -1)
    df['PREV_BODY_PERC'] = df.BODY_PERC.shift(1)
    df['PREV_DIRECTION'] = df.DIRECTION.shift(1)
    df['PEN_DIRECTION'] = df.DIRECTION.shift(2)
    df['TRE_DIRECTION'] = df.DIRECTION.shift(3)
    df['PERC_EIGHTY'] = df.mid_h - (df.mid_h - df.mid_l)*0.8
    df['TRE_PERC_EIGHTY'] = df.PERC_EIGHTY.shift(3)
    df['PREV_BODY_HIGH'] = df.mid_h.shift(1)
    df['PREV_BODY_CLOSE'] = df.mid_c.shift(1)
    df['PREV_BODY_LOW'] = df.mid_l.shift(1)
    df['TP'] = (df.mid_h + df.mid_l + df.mid_c) / 3
    df['MA_TP'] = df.TP.rolling(window=20).mean()
    df['STD_TP'] = df.TP.rolling(window=20).std()
    df['BB_UPPER'] = df.MA_TP + 2 * df.STD_TP
    df['BB_LOWER'] = df.MA_TP - 2 * df.STD_TP
    df['FOURT_LOW'] = df['mid_l'].rolling(window=14).min()
    df['FOURT_HIGH'] = df['mid_h'].rolling(window=14).max()
    df['K'] = (df.mid_c - df.FOURT_LOW)*100 / (df.FOURT_HIGH - df.FOURT_LOW)
    df['D'] = df['K'].rolling(3).mean()

    df.dropna(inplace=True)

    df['PX'] = df.apply(apply_px, axis=1)
    df['BB_SIGNAL'] = df.apply(Get_BB_Signal, axis=1)
    df['PREV_BB_SIGNAL'] = df.BB_SIGNAL.shift(1)
    df['MBB_SIGNAL'] = df.apply(Get_MBB_Signal, axis=1)
    MBB_PX(df)
    Get_SO_Signal(df)
    df['PREV_OS_SIGNAL'] = df.OS_SIGNAL.shift(1)
    BB_OS_PX(df)
    MBB_BB_OS_PX(df)

    apply_entry(df)
    apply_stoploss(df)
    apply_takeprofit(df)

    get_entry_limit(df)
    get_stop_loss(df)
    get_take_profit(df)
    df.dropna(inplace=True)

    df_trades = df[df.MBB_BB_OS_PX != 0].copy()
    df_trades["next"] = df_trades["time"].shift(-1)
    df_trades['trade_end'] = df_trades.next + dt.timedelta(hours=3, minutes=55)
    df_trades['trade_start'] = df_trades.time + dt.timedelta(hours=4)

    df_trades.reset_index(drop=True, inplace=True)
    print(df_trades.shape[0])

    return df_trades


def evaluate_pair_v2(row, m5_data, pipLoc):
    m5_slice = m5_data[(m5_data.time >= row.trade_start)
                       & (m5_data.time <= row.trade_end)]
    if row.MBB_BB_OS_PX == 1:
        r = process_buy(row.TAKEPROFIT, row.STOPLOSS,
                        m5_slice.ask_c.values, m5_slice.bid_c.values, row.ENTRY, pipLoc)
    else:
        r = process_sell(row.TAKEPROFIT, row.STOPLOSS,
                         m5_slice.ask_c.values, m5_slice.bid_c.values, row.ENTRY, pipLoc)
    return r


def modify_df(df_trades, m5_data, pipLoc):
    df = df_trades.copy()
    df['result'] = df.apply(evaluate_pair_v2, axis=1, args=[m5_data, pipLoc])
    return df, df['result'].sum()


def run():
    global SL_ENTRY_TP
    global LOSS, GAIN
    SL_ENTRY_TP = [1, -0.236,  -2.7]
    granularities = ["H4"]
    PREV_PERCENTAGEs = [0.2, 0.3, 0.5]
    stok_K = [50, 70]
    stok_D = [50, 30]
    entry_point = [-0.1, -0.213, -0.3, -0.35]
    loss_stop = [1, 0.618, 0.5, 0.33, 0.23]
    take_profit = [-1, -1.213, -1.5, -2, -2.5]
    for h in granularities:
        global granularity
        granularity = h

        for k in PREV_PERCENTAGEs:
            global PREV_PERCENTAGE
            PREV_PERCENTAGE = k

            for l in stok_K:
                global stoch_K
                stoch_K = l

                for g in stok_D:
                    global stoch_D
                    stoch_D = g

                    for i in loss_stop:
                        SL_ENTRY_TP[0] = i

                        for f in take_profit:
                            SL_ENTRY_TP[2] = f

                            for j in entry_point:
                                SL_ENTRY_TP[1] = j
                                currencies = "GBP,EUR,USD,CAD,JPY"
                                test_pairs = get_test_pairs(currencies)
                                # GBP,EUR,USD,CAD,JPY,NZD,CHF
                                grand_total = 0
                                print(
                                    f"Granularity= {granularity}. Stoch %K: {stoch_K}, %D: {stoch_D}. Prev body percentage: {PREV_PERCENTAGE}. SL/ENTRY/TP: {SL_ENTRY_TP[0]}/{SL_ENTRY_TP[1]}/{SL_ENTRY_TP[2]} ")
                                empty_list = []
                                for pairname in test_pairs:

                                    i_pair = instrument.Instrument.get_instruments_dict()[
                                        pairname]

                                    pipLoc = instrument.Instrument.get_pipLocation(
                                        pairname)

                                    h4_data = pd.read_pickle(
                                        utils.get_his_data_filename(pairname, granularity))
                                    m5_data = pd.read_pickle(
                                        utils.get_his_data_filename(pairname, "M5"))

                                    df_trades = get_trades_df(h4_data)

                                    df, score = modify_df(
                                        df_trades, m5_data, pipLoc)
                                    grand_total += score
                                    print(f"{pairname} {score:.0f}")
                                    empty_list.append([pairname, score])
                                print(f"TOTAL {grand_total:.0f}")
                                fd = pd.DataFrame(
                                    empty_list, columns=['Pair', 'Score'])
                                fd.to_pickle(
                                    f"pipresults/{granularity}_stochs_{stoch_K}_{stoch_D}_Prev_body_{PREV_PERCENTAGE}_SL_EN_TP_{SL_ENTRY_TP[0]}_{SL_ENTRY_TP[1]}_{SL_ENTRY_TP[2]}.pkl")


if __name__ == "__main__":
    run()
