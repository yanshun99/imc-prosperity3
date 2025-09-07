from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Tuple, Any
import numpy as np
from math import log, sqrt, floor, ceil
from statistics import NormalDist
import jsonpickle
import json
import math
import pandas as pd
from collections import defaultdict
############################################
# 产品定义
############################################


import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = str(value)[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

class Product:
    # Round 1 产品
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    # Round 2 产品
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUIDINK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    # 个股产品
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    JAMS = "JAMS"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

############################################
# 参数及持仓限制
############################################
PARAMS = {
    # VOLCANIC_ROCK 系列券参数（调整了 iv_deviation_threshold, trade_cooldown, delta_buffer）
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.0225,
        "threshold": 0.001,
        "strike": 9500,
        "starting_time_to_expiry": 1.0,   # 初始视为7个交易日
        "std_window": 10,
        "zscore_threshold": 5,
        "iv_deviation_threshold": 0.002,    # 降低阈值
        "trade_cooldown": 10,               # 降低冷却周期
        "delta_buffer": 5,                  # 降低对冲容忍带
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.0225,
        "threshold": 0.001,
        "strike": 9750,
        "starting_time_to_expiry": 1.0,
        "std_window": 10,
        "zscore_threshold": 5,
        "iv_deviation_threshold": 0.002,
        "trade_cooldown": 10,
        "delta_buffer": 5,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.06,
        "threshold": 0.001,
        "strike": 10000,
        "starting_time_to_expiry": 1.0,
        "std_window": 10,              # 增加窗口长度
        "zscore_threshold": 5,         # 提高触发门槛
        "trade_cooldown": 100,         # 增加 cooldown
        "delta_buffer": 10             # Delta 容忍区间
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.0225,
        "threshold": 0.001,
        "strike": 10250,
        "starting_time_to_expiry": 1.0,
        "std_window": 10,
        "zscore_threshold": 5,
        "iv_deviation_threshold": 0.002,
        "trade_cooldown": 10,
        "delta_buffer": 5,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.0225,
        "threshold": 0.001,
        "strike": 10500,
        "starting_time_to_expiry": 1.0,
        "std_window": 10,
        "zscore_threshold": 5,
        "iv_deviation_threshold": 0.002,
        "trade_cooldown": 10,
        "delta_buffer": 5,
    },
    # 非券产品参数（示例）
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000, 
        "take_width": 1, 
        "clear_width": 0, 
        "disregard_edge": 1, 
        "join_edge": 2, 
        "default_edge": 4, 
        "soft_position_limit": 25
    },
    Product.KELP: {
        "take_width": 1, 
        "clear_width": 0, 
        "prevent_adverse": True, 
        "adverse_volume": 15, 
        "reversion_beta": -0.229, 
        "default_edge": 1, 
        "soft_position_limit": 25, 
        "disregard_edge": 1,
        "join_edge": 0
    },
    Product.SQUIDINK: {
        "take_width": 1, 
        "clear_width": 0, 
        "prevent_adverse": True, 
        "adverse_volume": 15, 
        "reversion_beta": -0.25, 
        "default_edge": 1, 
        "soft_position_limit": 25
    },
    Product.PICNIC_BASKET1: {"fair_value": 0},
    Product.PICNIC_BASKET2: {"fair_value": 0},
    # 个股产品的示例
    Product.CROISSANTS: {"fair_value": 10000},
    Product.DJEMBES: {"fair_value": 10000},
    Product.JAMS: {"fair_value": 10000},
    Product.MAGNIFICENT_MACARONS: {
            "make_probability": 0.8,
            # CSI
            "csi_window": 5,
            "sun_history_len": 180,
            "static_csi": False,      # for smoke tests
            # sugar & fee
            "sugar_history_len": 180,
            "fee_history_len": 180,
            # pyramidal
            "scale_steps": 3,
            "profit_targets": [0.02, 0.04, 0.06],
            # stop-loss & hold
            "stop_loss_pct": 0.02,
            "max_hold_ticks": 200,
            # dynamic percentiles
            "csi_pct": 10,
            "sugar_pct": 90,
            "fee_pct": 90,
            # debug
            "debug": False,
            "take_width": 1, 
            "clear_width": 0, 
            "prevent_adverse": True, 
            "adverse_volume": 15, 
            "reversion_beta": -0.229, 
            "default_edge": 1, 
            "soft_position_limit": 25, 
            "disregard_edge": 1,
            "join_edge": 0
    }
}

LIMIT = {
    Product.VOLCANIC_ROCK: 400,
    Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
    Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
    Product.RAINFOREST_RESIN: 50,
    Product.KELP: 50,
    Product.SQUIDINK: 50,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
    Product.CROISSANTS: 250,
    Product.DJEMBES: 60,
    Product.JAMS: 350,
    Product.MAGNIFICENT_MACARONS: 75,
}

############################################
# IV Smile 模型函数
############################################
def iv_smile(m: float) -> float:
    # 拟合隐含波动率（fitted iv）的公式
    return 0.2013 * m * m + 0.0219 * m + 0.1972

############################################
# Black-Scholes 工具类
############################################
class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        return spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

############################################
# Trader 类（完整优化版）
############################################
class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = LIMIT
        self.vol_cache = {}       # 隐含波动率缓存
        # 内部状态保存：用于各券和其他产品
        self.traderData = {}
        self.resin_prices = []
        self.resin_vwap = []
        self.kelp_prices = []
        self.squid_prices = []
        self.pnl_history = []
        self.band_history = []
        self.trade_log = []
        self.last_trade_price = None
        self.sun_history: List[float] = []
        self.sugar_history: List[float] = []
        self.fee_history: List[float] = []
        self.price_history: List[float] = []
        self.tick = 0

    def update_histories(self, obs: ConversionObservation):
        p = self.params[Product.MAGNIFICENT_MACARONS]
        self.sun_history.append(obs.sunlightIndex)
        if len(self.sun_history) > p["sun_history_len"]:
            self.sun_history.pop(0)
        self.sugar_history.append(getattr(obs, 'sugarPrice', obs.bidPrice))
        if len(self.sugar_history) > p["sugar_history_len"]:
            self.sugar_history.pop(0)
        self.fee_history.append(obs.transportFees)
        if len(self.fee_history) > p["fee_history_len"]:
            self.fee_history.pop(0)
        self.price_history.append((obs.bidPrice + obs.askPrice) / 2)

    def compute_panic(self) -> bool:
        p = self.params[Product.MAGNIFICENT_MACARONS]
        if p.get("static_csi"):
            return True
        if len(self.sun_history) < p["csi_window"]:
            return False
        csi_thr = np.percentile(self.sun_history, p["csi_pct"])
        csi = all(x < csi_thr for x in self.sun_history[-p["csi_window"]:])
        sugar_thr = np.percentile(self.sugar_history, p["sugar_pct"])
        sugar = self.sugar_history[-1] > sugar_thr
        fee_thr = np.percentile(self.fee_history, p["fee_pct"])
        fee = self.fee_history[-1] > fee_thr
        if p.get("debug"):
            print(f"Tick {self.tick}, signals: CSI={csi}, sugar={sugar}, fee={fee}")
        return sum([csi, sugar, fee]) >= 2

    def implied_bid_ask(self, obs: ConversionObservation) -> (float, float):
        return (
            obs.bidPrice - obs.exportTariff - obs.transportFees,
            obs.askPrice + obs.importTariff + obs.transportFees,
        )

    def arb_take(self, depth: OrderDepth, obs: ConversionObservation, pos: int):
        orders, buy_vol, sell_vol = [], 0, 0
        limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        ibid, iask = self.implied_bid_ask(obs)
        buy_qty, sell_qty = limit - pos, limit + pos
        edge = max(0.0, (round(obs.askPrice)-2 - iask) * self.params[Product.MAGNIFICENT_MACARONS]["make_probability"])
        if self.compute_panic(): edge += (iask-ibid)*0.2
        for price, qty in sorted(depth.sell_orders.items()):
            if price <= ibid - 0.1 + edge and buy_qty>0:
                q=min(abs(qty), buy_qty)
                orders.append(Order(Product.MAGNIFICENT_MACARONS, price, q)); buy_qty-=q; buy_vol+=q
        for price, qty in sorted(depth.buy_orders.items(), reverse=True):
            if price >= iask + edge and sell_qty>0:
                q=min(abs(qty), sell_qty)
                orders.append(Order(Product.MAGNIFICENT_MACARONS, price, -q)); sell_qty-=q; sell_vol+=q
        return orders, buy_vol, sell_vol

    def arb_make(self, obs: ConversionObservation, pos: int, buy_vol: int, sell_vol: int):
        orders = []
        limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        ibid, iask = self.implied_bid_ask(obs)
        a_bid, a_ask = round(obs.bidPrice)+2, round(obs.askPrice)-2
        bid = a_bid if a_bid<ibid-0.1 else ibid-0.5
        ask = a_ask if a_ask>=iask+0.5 else iask+0.5
        if self.compute_panic(): bid+=(iask-ibid)*0.1
        bq = limit-(pos+buy_vol); sq = limit+(pos-sell_vol)
        if bq>0: orders.append(Order(Product.MAGNIFICENT_MACARONS, round(bid), bq))
        if sq>0: orders.append(Order(Product.MAGNIFICENT_MACARONS, round(ask), -sq))
        return orders, buy_vol, sell_vol

    def clear_pos(self, pos:int)->int:
        return -pos

    def get_voucher_mid_price(self, voucher_order_depth: OrderDepth, traderData: Dict[str, any]):
        if voucher_order_depth.buy_orders and voucher_order_depth.sell_orders:
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            traderData["prev_voucher_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData.get("prev_voucher_price", self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["strike"])

    def delta_hedge_voucher_position(self, rock_order_depth: OrderDepth, voucher_position: int, rock_position: int,
                                     rock_buy_orders: int, rock_sell_orders: int, delta: float) -> List[Order]:
        target_rock_position = -int(delta * voucher_position)
        hedge_quantity = target_rock_position - (rock_position + rock_buy_orders - rock_sell_orders)
        delta_buffer = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000].get("delta_buffer", 0)
        if abs(hedge_quantity) < delta_buffer:
            return []
        orders = []
        if hedge_quantity > 0:
            best_ask = min(rock_order_depth.sell_orders.keys())
            quantity = min(abs(hedge_quantity), -rock_order_depth.sell_orders[best_ask])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] - (rock_position + rock_buy_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif hedge_quantity < 0:
            best_bid = max(rock_order_depth.buy_orders.keys())
            quantity = min(abs(hedge_quantity), rock_order_depth.buy_orders[best_bid])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] + (rock_position - rock_sell_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))
        return orders

    def delta_hedge_voucher_orders(self, rock_order_depth: OrderDepth, voucher_orders: List[Order],
                                   rock_position: int, rock_buy_orders: int, rock_sell_orders: int, delta: float) -> List[Order]:
        if not voucher_orders:
            return []
        net_voucher_quantity = sum(order.quantity for order in voucher_orders)
        target_rock_quantity = -int(delta * net_voucher_quantity)
        delta_buffer = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000].get("delta_buffer", 0)
        if abs(target_rock_quantity) < delta_buffer:
            return []
        orders = []
        if target_rock_quantity > 0:
            best_ask = min(rock_order_depth.sell_orders.keys())
            quantity = min(abs(target_rock_quantity), -rock_order_depth.sell_orders[best_ask])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] - (rock_position + rock_buy_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif target_rock_quantity < 0:
            best_bid = max(rock_order_depth.buy_orders.keys())
            quantity = min(abs(target_rock_quantity), rock_order_depth.buy_orders[best_bid])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] + (rock_position - rock_sell_orders))
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))
        return orders

    def voucher_hedge_orders(self, rock_order_depth: OrderDepth, voucher_order_depth: OrderDepth,
                             voucher_take_orders: List[Order], rock_position: int, voucher_position: int,
                             delta: float) -> List[Order]:
        self.traderData = {}
        traderData = self.traderData
        traderData["current_step"] = traderData.get("current_step", 0) + 1
        current_step = traderData["current_step"]
        last_trade_step = traderData.get("last_trade_step", -1000)
        cooldown = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["trade_cooldown"]

        if current_step - last_trade_step < cooldown:
            return []

        traderData["last_trade_step"] = current_step

        if not voucher_take_orders:
            voucher_position_after_trade = voucher_position
        else:
            voucher_position_after_trade = voucher_position + sum(order.quantity for order in voucher_take_orders)

        target_rock_position = -int(delta * voucher_position_after_trade)
        hedge_quantity = target_rock_position - rock_position
        delta_buffer = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000].get("delta_buffer", 0)
        if abs(hedge_quantity) < delta_buffer:
            return []

        orders = []
        if hedge_quantity > 0:
            best_ask = min(rock_order_depth.sell_orders.keys())
            quantity = min(abs(hedge_quantity), -rock_order_depth.sell_orders[best_ask])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] - rock_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif hedge_quantity < 0:
            best_bid = max(rock_order_depth.buy_orders.keys())
            quantity = min(abs(hedge_quantity), rock_order_depth.buy_orders[best_bid])
            quantity = min(quantity, self.LIMIT[Product.VOLCANIC_ROCK] + rock_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))

        return orders


    # 根据券盘口信息和隐含波动率信号生成券订单。当隐含 vol 的 z-score 超过阈值时对应建仓（多头或空头）
    def voucher_orders(self, voucher_order_depth: OrderDepth, voucher_position: int,
                       traderData: Dict[str, any], volatility: float) -> Tuple[List[Order], List[Order]]:
        traderData.setdefault("past_voucher_vol", [])
        traderData.setdefault("last_trade_step", -1000)  # 用于防止频繁交易
        traderData["past_voucher_vol"].append(volatility)

        window = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["std_window"]
        mean_vol = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["mean_volatility"]
        zscore_threshold = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["zscore_threshold"]

        # 控制窗口长度
        if len(traderData["past_voucher_vol"]) > window:
            traderData["past_voucher_vol"].pop(0)

        # 若数据不足，先不交易
        if len(traderData["past_voucher_vol"]) < window:
            return None, None

        # 避免标准差过小导致的极端 z-score
        std_vol = np.std(traderData["past_voucher_vol"], ddof=1)
        if std_vol < 1e-4:
            return None, None

        vol_z_score = (volatility - mean_vol) / std_vol

        # 获取盘口价差
        if not voucher_order_depth.buy_orders or not voucher_order_depth.sell_orders:
            return None, None
        best_bid = max(voucher_order_depth.buy_orders.keys())
        best_ask = min(voucher_order_depth.sell_orders.keys())
        spread = best_ask - best_bid

        min_spread = 5  # 设定最小价差保护，防止在大价差时交易
        if spread > min_spread:
            return None, None

        # 设置交易 cooldown，避免高频震荡
        current_step = traderData.get("step", 0)
        if current_step - traderData["last_trade_step"] < 10:
            return None, None

        # 卖出信号
        if vol_z_score >= zscore_threshold and voucher_position > -self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10000]:
            quantity = min(abs(voucher_position + self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10000]),
                           abs(voucher_order_depth.buy_orders[best_bid]))
            traderData["last_trade_step"] = current_step
            return [Order(Product.VOLCANIC_ROCK_VOUCHER_10000, best_bid, -quantity)], []

        # 买入信号
        elif vol_z_score <= -zscore_threshold and voucher_position < self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10000]:
            quantity = min(abs(self.LIMIT[Product.VOLCANIC_ROCK_VOUCHER_10000] - voucher_position),
                           abs(voucher_order_depth.sell_orders[best_ask]))
            traderData["last_trade_step"] = current_step
            return [Order(Product.VOLCANIC_ROCK_VOUCHER_10000, best_ask, quantity)], []

        return None, None

    # 处理 VOLCANIC_ROCK_VOUCHER 部分，利用盘口数据、剩余期限计算隐含 vol、delta，并生成券及对应的 VOLCANIC_ROCK 对冲订单
    def process_volcanic_rock_voucher(self, state: TradingState, traderData: Dict[str, any], result: Dict[str, List[Order]]):
        if Product.VOLCANIC_ROCK_VOUCHER_10000 not in traderData:
            traderData[Product.VOLCANIC_ROCK_VOUCHER_10000] = {
                "prev_voucher_price": self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["strike"],
                "past_voucher_vol": []
            }
        if Product.VOLCANIC_ROCK_VOUCHER_10000 in state.order_depths and Product.VOLCANIC_ROCK in state.order_depths:
            voucher_position = state.position.get(Product.VOLCANIC_ROCK_VOUCHER_10000, 0)
            rock_position = state.position.get(Product.VOLCANIC_ROCK, 0)
            rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
            voucher_order_depth = state.order_depths[Product.VOLCANIC_ROCK_VOUCHER_10000]
            if rock_order_depth.buy_orders and rock_order_depth.sell_orders:
                rock_mid_price = (min(rock_order_depth.sell_orders.keys()) + max(rock_order_depth.buy_orders.keys())) / 2
            else:
                rock_mid_price = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["strike"]
            voucher_mid_price = self.get_voucher_mid_price(voucher_order_depth, traderData[Product.VOLCANIC_ROCK_VOUCHER_10000])
            # 剩余到期时间：初始时间为1.0（7天），用 state.timestamp（单位微秒）换算经过的天数，简化假设 1e6 微秒 = 1 天
            tte = self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["starting_time_to_expiry"] - (state.timestamp / 1e6) / 7
            # 反推隐含 vol：用券中间价作为市场价格
            volatility = BlackScholes.implied_volatility(voucher_mid_price, rock_mid_price,
                                                         self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["strike"],
                                                         tte)
            delta = BlackScholes.delta(rock_mid_price, self.params[Product.VOLCANIC_ROCK_VOUCHER_10000]["strike"],
                                       tte, volatility)
            voucher_take_orders, voucher_make_orders = self.voucher_orders(voucher_order_depth, voucher_position, traderData[Product.VOLCANIC_ROCK_VOUCHER_10000], volatility)
            rock_orders = self.voucher_hedge_orders(rock_order_depth, voucher_order_depth, voucher_take_orders, rock_position, voucher_position, delta)
            if voucher_take_orders or voucher_make_orders:
                result[Product.VOLCANIC_ROCK_VOUCHER_10000] = (voucher_take_orders or []) + (voucher_make_orders or [])
            if rock_orders:
                result[Product.VOLCANIC_ROCK] = rock_orders


    ############################################
    # 非券产品部分： Resin, Kelp, SquidInk, Picnic Baskets
    ############################################
    def zscore(self, prices: List[float], window: int = 14) -> float:
        if len(prices) < window:
            return 0.0
        mean = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        return (prices[-1] - mean) / std if std > 0 else 0

    def sma(self, prices: List[float], window: int = 10) -> float:
        if len(prices) < window:
            return prices[-1]
        return np.mean(prices[-window:])

    def slope(self, prices: List[float], window: int = 10) -> float:
        if len(prices) < window:
            return 0.0
        return np.polyfit(range(window), prices[-window:], 1)[0]

    def bias(self, prices: List[float], window: int = 10) -> float:
        prices_series = pd.Series(prices)
        ma = prices_series.rolling(window=window, min_periods=1).mean()
        bias = prices_series / ma - 1
        alpha = bias.rolling(window=window, min_periods=1).mean()
        return alpha.iloc[-1]

    def mtm_std(self, prices: List[float], window: int = 10) -> float:
        prices_series = pd.Series(prices)
        mtm = prices_series.pct_change(window)
        mtm_mean = mtm.rolling(window=window, min_periods=1).mean()
        mtm_std = mtm.rolling(window=window, min_periods=1).std()
        alpha = mtm_mean * mtm_std
        return alpha.iloc[-1]

    def detect_v_shape(self, prices: List[float], threshold: float = 0.2) -> bool:
        if len(prices) < 7:
            return False
        pre3 = prices[-7:-4]
        mid = prices[-4:-2]
        post3 = prices[-2:]
        return np.mean(mid) < np.mean(pre3)*(1 - threshold) and np.mean(post3) > np.mean(mid)*(1 + threshold)

    def best_band_coef(self) -> float:
        coefs = [1.0, 1.2, 1.5, 1.8, 2.0]
        best_coef = 1.5
        best_score = -float('inf')
        for c in coefs:
            pnl = 0
            for i in range(30, len(self.squid_prices)):
                window = self.squid_prices[i-20:i]
                mid = self.squid_prices[i]
                sma_val = np.mean(window)
                std = np.std(window)
                upper, lower = sma_val + c*std, sma_val - c*std
                if mid < lower:
                    pnl += window[-1] - mid
                elif mid > upper:
                    pnl += mid - window[-1]
            if pnl > best_score:
                best_score = pnl
                best_coef = c
        return best_coef
    
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> Tuple[int, int]:
        position_limit = self.LIMIT[product]
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order2(self, orders: List[Order], order_depth: OrderDepth, position: int,
                              position_limit: int, product: str, buy_order_volume: int,
                              sell_order_volume: int, fair_value: float, width: int) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)
        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def clear_position_order3(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int,
                             position_limit: int, product: str, buy_order_volume: int,
                             sell_order_volume: int, fair_value: float, width: int) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)
        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def resin_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        elif method == "mid_price_with_vol_filter":
            if (len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) == 0 or 
                len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) == 0):
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                return (best_ask + best_bid) / 2
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                return (best_ask + best_bid) / 2

    def resin_orders(self, order_depth: OrderDepth, timespan: int, width: float, resin_take_width: float,
                     position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid
            mmmid_price = (mm_ask + mm_bid) / 2
            self.resin_prices.append(mmmid_price)
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-order_depth.sell_orders[best_ask]) + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.resin_vwap.append({"vol": volume, "vwap": vwap})
            if len(self.resin_vwap) > timespan:
                self.resin_vwap.pop(0)
            if len(self.resin_prices) > timespan:
                self.resin_prices.pop(0)
            fair_value = mmmid_price  # 这里也可以用 VWAP 加权均价
            if best_ask <= fair_value - resin_take_width:
                ask_amount = -order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + resin_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
                        sell_order_volume += quantity
            buy_order_volume, sell_order_volume = self.clear_position_order2(
                orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 2)
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))
            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))
        return orders

    ############################################
    # 辅助函数：take_orders, clear_orders, make_orders, kelp_orders（代码同前，仅未变动）
    ############################################
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order3(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get("KELP_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get("KELP_last_price", None) is not None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return 10  # 若盘口数据不足，返回默认值

    def kelp_orders(self, order_depth: OrderDepth, position: int, position_limit: int, window_size: int = 10, threshold: float = 1.5) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        if order_depth.sell_orders and order_depth.buy_orders:
            current_mid = (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) / 2
        else:
            current_mid = 10  # 默认值
        self.kelp_prices.append(current_mid)
        if len(self.kelp_prices) > 10:
            self.kelp_prices = self.kelp_prices[-10:]
        if len(self.kelp_prices) >= 5:
            mean_price = np.mean(self.kelp_prices)
            std_price = np.std(self.kelp_prices)
            dynamic_fair_value = mean_price
        else:
            dynamic_fair_value = current_mid
        fair_value = dynamic_fair_value
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -quantity))
                    sell_order_volume += quantity
        buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 1)
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", bbbf + 1, buy_quantity))
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", baaf - 1, -sell_quantity))
        return orders
    
    def squid_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        orders = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        # 取得订单簿最佳报价
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2.0
        self.squid_prices.append(mid_price)
        if len(self.squid_prices) > 100:
            self.squid_prices = self.squid_prices[-100:]
        if len(self.squid_prices) < 30:
            return orders

        # 计算动态布林带系数
        band_coef = self.best_band_coef()
        self.band_history.append(band_coef)
        
        # 计算均值和标准差，确定布林带上轨、下轨
        sma_val = self.sma(self.squid_prices, 20)
        std_val = np.std(self.squid_prices[-20:])
        upper_band = sma_val + band_coef * std_val
        lower_band = sma_val - band_coef * std_val

        # 计算其他技术指标
        slope_val = self.slope(self.squid_prices, 10)
        zscore_val = self.zscore(self.squid_prices, 14)
        bias_val = self.bias(self.squid_prices, 10)
        mtm_std_val = self.mtm_std(self.squid_prices, 10)

        # 从订单簿中获取成交量
        bid_volume = order_depth.buy_orders[best_bid]
        ask_volume = -order_depth.sell_orders[best_ask]

        # 自适应交易量（根据近期盈亏调整）
        recent_pnl = sum(self.pnl_history[-20:]) if len(self.pnl_history) >= 20 else 0
        if recent_pnl < -3000:
            max_trade_size = 5
        elif recent_pnl > 2000:
            max_trade_size = 25
        else:
            max_trade_size = 10

        # ---- Modified Risk Filter 1: 波动率过滤 ----
        # 计算 vol_ratio = std / sma; 原阈值为 0.05，现调整为 0.08
        vol_ratio = std_val / sma_val if sma_val != 0 else 0
        if vol_ratio > 0.08:
            self.trade_log.append("High volatility condition met; skipping active trades")
            grid_spread = 3
            passive_qty = 5
            if position < position_limit:
                orders.append(Order("SQUID_INK", best_bid - grid_spread, passive_qty))
            if position > -position_limit:
                orders.append(Order("SQUID_INK", best_ask + grid_spread, -passive_qty))
            return orders

        # ---- Modified Risk Filter 2: 亏损过滤 ----
        # 原 threshold -5000，现调整为 -8000，以便减少主动交易跳过频率
        if recent_pnl < -8000:
            self.trade_log.append("Recent pnl too negative; skipping active trades")
            grid_spread = 3
            passive_qty = 5
            if position < position_limit:
                orders.append(Order("SQUID_INK", best_bid - grid_spread, passive_qty))
            if position > -position_limit:
                orders.append(Order("SQUID_INK", best_ask + grid_spread, -passive_qty))
            return orders

        # ---- 主动交易信号（基于 V形反转与 zscore） ----
        # BUY 信号：若价格低于下轨且短期斜率为正
        if mid_price < lower_band and slope_val > 0:
            # Modified：将 zscore 阈值从 -1.5 调整为 -1.2
            if self.detect_v_shape(self.squid_prices, threshold=0.2) or zscore_val < -1.2:
                quantity = min(min(position_limit - position, ask_volume), max_trade_size)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_ask, quantity))
                    # pnl 计算：买入时成本 best_ask，记录盈亏供后续参考
                    pnl_update = (self.last_trade_price if self.last_trade_price is not None else mid_price) - best_ask
                    self.pnl_history.append(pnl_update)
                    self.last_trade_price = best_ask
                    self.trade_log.append(f"BUY at {best_ask}, qty {quantity}, mid {mid_price}")
        # SELL 信号：若价格高于上轨且短期斜率为负
        elif mid_price > upper_band and slope_val < 0:
            if zscore_val > 1.2:
                quantity = min(min(position + position_limit, bid_volume), max_trade_size)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_bid, -quantity))
                    pnl_update = best_bid - (self.last_trade_price if self.last_trade_price is not None else mid_price)
                    self.pnl_history.append(pnl_update)
                    self.last_trade_price = best_bid
                    self.trade_log.append(f"SELL at {best_bid}, qty {quantity}, mid {mid_price}")

        # ---- 被动网格订单（补充流动性） ----
        grid_spread = 3
        passive_qty = 5
        if position < position_limit:
            orders.append(Order("SQUID_INK", best_bid - grid_spread, passive_qty))
        if position > -position_limit:
            orders.append(Order("SQUID_INK", best_ask + grid_spread, -passive_qty))

        return orders
    ############################################
    # run 函数：整合券与非券产品的交易逻辑
    ############################################
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # 1. 处理 VOLCANIC_ROCK 系列券
        #self.process_volcanic_rock_voucher(state, traderObject, result)

        # ----------------------------
        # RAINFOREST_RESIN 参数设置
        resin_position_limit = 50
        resin_width = 2

        # ----------------------------
        # KELP 参数设置
        kelp_position_limit = 50
        kelp_window_size = 10
        kelp_threshold = 1.5
        KELP_make_width = 3.5
        KELP_take_width = 1

        # ----------------------------
        # 处理 RAINFOREST_RESIN
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_order_depth = state.order_depths["RAINFOREST_RESIN"]
            resin_orders = self.resin_orders(
                resin_order_depth,
                kelp_window_size,
                resin_width,
                kelp_threshold,
                state.position.get("RAINFOREST_RESIN", 0),
                resin_position_limit
            )
            result["RAINFOREST_RESIN"] = resin_orders

        # ----------------------------
        # KELP 策略
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = state.position.get(Product.KELP, 0)
            KELP_fair_value = self.KELP_fair_value(state.order_depths[Product.KELP], traderObject)
            KELP_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                self.params[Product.KELP]["take_width"],
                KELP_position,
                self.params[Product.KELP].get("prevent_adverse", False),
                self.params[Product.KELP].get("adverse_volume", 0)
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                self.params[Product.KELP]["clear_width"],
                KELP_position,
                buy_order_volume,
                sell_order_volume
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = KELP_take_orders + KELP_clear_orders + KELP_make_orders
        
        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_order_depth = state.order_depths["SQUID_INK"]
            result["SQUID_INK"] = self.squid_orders(squid_order_depth, squid_position, 100)

        # 1. MAGNIFICENT_MACARONS 金字塔 + 套利 + 风控逻辑
        if Product.MAGNIFICENT_MACARONS in state.order_depths:
            td = traderObject
            obs = state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS]
            # 更新历史/计数器
            self.update_histories(obs)
            self.tick += 1
            pos = state.position.get(Product.MAGNIFICENT_MACARONS, 0)
            limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
            params = self.params[Product.MAGNIFICENT_MACARONS]

            # Pyramidal 入场
            level = td.get("scale_level", 0)
            panic = self.compute_panic()
            target = int(limit * (level + 1) / params["scale_steps"]) if (panic and level < params["scale_steps"]) else 0
            diff = target - pos
            if diff > 0 and pos == 0:
                td["entry_price"] = self.price_history[-1]
                td["entry_tick"] = self.tick
                td["scale_level"] = level + 1

            # Spot 市场 Arbitrage
            depth = state.order_depths[Product.MAGNIFICENT_MACARONS]
            take_orders, bv, sv = self.arb_take(depth, obs, pos)
            make_orders, _, _ = self.arb_make(obs, pos, bv, sv)
            if make_orders:
                conversions = self.clear_pos(pos)
                result[Product.MAGNIFICENT_MACARONS] = take_orders + make_orders
            else:
                # 风控：止损/时间止损/分批止盈
                if pos > 0:
                    ep = td.get("entry_price", self.price_history[0])
                    # 止损
                    if obs.bidPrice < ep * (1 - params["stop_loss_pct"]):
                        conversions = -pos
                    # 时间止损
                    elif self.tick - td.get("entry_tick", 0) > params["max_hold_ticks"]:
                        conversions = -pos
                    else:
                        # 分批止盈
                        for lvl, pct in enumerate(params["profit_targets"][:level]):
                            if self.price_history[-1] >= ep * (1 + pct):
                                conversions = -int(limit / params["scale_steps"])
                                td["scale_level"] = level - 1
                                break
                # 达到目标建仓
                if conversions == 0 and diff != 0:
                    conversions = diff
                # 若无转换，则保留套利吃单
                if conversions == 0 and take_orders:
                    result[Product.MAGNIFICENT_MACARONS] = take_orders
        


        # 6. 处理简单产品：个股及 Picnic Baskets
        simple_products = [
            Product.CROISSANTS, Product.DJEMBES, Product.JAMS,
            Product.PICNIC_BASKET1, Product.PICNIC_BASKET2,
        ]

        simple_products1 = [
            Product.VOLCANIC_ROCK, Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250, Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750, Product.VOLCANIC_ROCK_VOUCHER_10500
        ]

        # 🔧 Initialize data structures
        if "start_positions" not in traderObject:
            traderObject["start_positions"] = {
                product: state.position.get(product, 0)
                for product in simple_products
            }

        if "pnl_history" not in traderObject:
            traderObject["pnl_history"] = []

        # 🔍 Compute current unrealized PnL for simple_products
        value_now = 0
        value_start = 0
        for product in simple_products1:
            pos_now = state.position.get(product, 0)
            pos_start = traderObject["start_positions"].get(product, 0)
            if product in state.order_depths:
                od = state.order_depths[product]
                mid_price = None
                if od.buy_orders and od.sell_orders:
                    mid_price = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2
                elif od.buy_orders:
                    mid_price = max(od.buy_orders.keys())
                elif od.sell_orders:
                    mid_price = min(od.sell_orders.keys())
                if mid_price is not None:
                    value_now += pos_now * mid_price
                    value_start += pos_start * mid_price

        vol_pnl = value_now - value_start

        # 🧠 Append current PnL to history
        traderObject["pnl_history"].append(vol_pnl)

        # 🔁 Keep only last 200,000 entries
        if len(traderObject["pnl_history"]) > 2000:
            traderObject["pnl_history"] = traderObject["pnl_history"][-2000:]

        # ⚠️ Check PnL from 200,000 timestamps ago
        acceptable_price_1 = None
        if len(traderObject["pnl_history"]) >= 2000:
            past_pnl = traderObject["pnl_history"][0]
            if past_pnl <= -50000:
                acceptable_price_1 = 0.1
                print("⚠️ Past PnL too low, setting acceptable_price to 0.1")

        for product in simple_products:
            if product not in state.order_depths:
                continue  # 没有该产品的订单簿

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # 估算的合理价格，这里为示例
            acceptable_price = 10

            print(f"Acceptable price for {product}: {acceptable_price}")
            print("Buy Order depth: " + str(len(order_depth.buy_orders)) +
                ", Sell order depth: " + str(len(order_depth.sell_orders)))

            # 当前持仓
            current_position = state.position.get(product, 0)
            max_limit = LIMIT[product]

            # 卖方报价：我们作为买家
            if order_depth.sell_orders:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                best_ask_amount = -best_ask_amount  # 转成正数表示我们最多可以买的量

                if best_ask < acceptable_price:
                    max_buyable = max_limit - current_position
                    qty = min(best_ask_amount, max(0, max_buyable))
                    if qty > 0:
                        print("BUY", qty, "x", best_ask, "for", product)
                        orders.append(Order(product, best_ask, qty))

            # 买方报价：我们作为卖家
            if order_depth.buy_orders:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                if best_bid > acceptable_price:
                    max_sellable = max_limit + current_position  # current_position is negative when short
                    qty = min(best_bid_amount, max(0, max_sellable))
                    if qty > 0:
                        print("SELL", qty, "x", best_bid, "for", product)
                        orders.append(Order(product, best_bid, -qty))  # 卖出是负数量

            result[product] = orders

        for product in simple_products1:
            if product not in state.order_depths:
                continue  # 没有该产品的订单簿

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # 估算的合理价格，这里为示例
            acceptable_price_1 = 1000000

            print(f"Acceptable price for {product}: {acceptable_price}")
            print("Buy Order depth: " + str(len(order_depth.buy_orders)) +
                ", Sell order depth: " + str(len(order_depth.sell_orders)))

            # 当前持仓
            current_position = state.position.get(product, 0)
            max_limit = LIMIT[product]

            # 卖方报价：我们作为买家
            if order_depth.sell_orders:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                best_ask_amount = -best_ask_amount  # 转成正数表示我们最多可以买的量

                if best_ask < acceptable_price_1:
                    max_buyable = max_limit - current_position
                    qty = min(best_ask_amount, max(0, max_buyable))
                    if qty > 0:
                        print("BUY", qty, "x", best_ask, "for", product)
                        orders.append(Order(product, best_ask, qty))

            # 买方报价：我们作为卖家
            if order_depth.buy_orders:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                if best_bid > acceptable_price_1:
                    max_sellable = max_limit + current_position  # current_position is negative when short
                    qty = min(best_bid_amount, max(0, max_sellable))
                    if qty > 0:
                        print("SELL", qty, "x", best_bid, "for", product)
                        orders.append(Order(product, best_bid, -qty))  # 卖出是负数量

            result[product] = orders

        conversions = 1
        traderDataStr = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderObject)
        return result, conversions, traderDataStr