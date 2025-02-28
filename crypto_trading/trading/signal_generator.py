"""
Signal generation for cryptocurrency trading.

This module provides an enhanced signal generator that uses model predictions
and market context to generate trading signals.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

from ..utils.logging_utils import exception_handler


class EnhancedSignalProducer:
    """Generates trading signals based on model predictions and market context."""

    def __init__(self, confidence_threshold: float = 0.4,
                 strong_signal_threshold: float = 0.7,
                 atr_multiplier_sl: float = 1.5,
                 use_regime_filter: bool = True,
                 use_volatility_filter: bool = True,
                 min_adx_threshold: int = 20,
                 max_vol_percentile: int = 85,
                 correlation_threshold: float = 0.6,
                 use_oi_filter: bool = True,
                 use_funding_filter: bool = True,
                 logger: Optional[logging.Logger] = None):
        """Initialize signal producer.

        Args:
            confidence_threshold: Minimum confidence for signals
            strong_signal_threshold: Threshold for strong signals
            atr_multiplier_sl: Multiplier for ATR to determine stop loss
            use_regime_filter: Whether to use market regime filter
            use_volatility_filter: Whether to use volatility filter
            min_adx_threshold: Minimum ADX for trend filter
            max_vol_percentile: Maximum volatility percentile allowed
            correlation_threshold: Threshold for timeframe agreement
            use_oi_filter: Whether to use open interest filter
            use_funding_filter: Whether to use funding rate filter
            logger: Logger to use
        """
        self.confidence_threshold = confidence_threshold
        self.strong_signal_threshold = strong_signal_threshold
        self.atr_multiplier_sl = atr_multiplier_sl
        self.use_regime_filter = use_regime_filter
        self.use_volatility_filter = use_volatility_filter
        self.min_adx_threshold = min_adx_threshold
        self.max_vol_percentile = max_vol_percentile
        self.correlation_threshold = correlation_threshold
        self.use_oi_filter = use_oi_filter
        self.use_funding_filter = use_funding_filter
        self.logger = logger or logging.getLogger('SignalProducer')

    @exception_handler(reraise=False, fallback_value={"signal_type": "NoTrade", "reason": "Error"})
    def get_signal(self, model_probs: np.ndarray, df: pd.DataFrame,
                   current_positions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate a trading signal.

        Args:
            model_probs: Model prediction probabilities
            df: DataFrame with market data
            current_positions: Current open positions

        Returns:
            Signal dictionary
        """
        if len(df) < 2:
            return {"signal_type": "NoTrade", "reason": "InsufficientData"}

        # Get base probabilities
        P_positive = model_probs[3] + model_probs[4]  # Classes 3 & 4 = bullish
        P_negative = model_probs[0] + model_probs[1]  # Classes 0 & 1 = bearish
        P_neutral = model_probs[2]  # Class 2 = neutral
        max_confidence = max(P_positive, P_negative)

        # Get current market conditions
        current_price = df['close'].iloc[-1]

        # Get market regime if available
        market_regime = 0
        if 'market_regime' in df.columns:
            market_regime = df['market_regime'].iloc[-1]

        # Get volatility regime if available
        volatility_regime = 0
        if 'volatility_regime' in df.columns:
            volatility_regime = df['volatility_regime'].iloc[-1]

        # Get trend strength if available
        trend_strength = 0
        if 'trend_strength' in df.columns:
            trend_strength = df['trend_strength'].iloc[-1]

        # Check if ATR data exists for stop loss calculation
        if 'd1_ATR_14' not in df.columns:
            # Calculate ATR if not available
            atr = self._compute_atr(df).iloc[-1]
        else:
            atr = df['d1_ATR_14'].iloc[-1]

        # Get historical volatility if available
        hist_vol = 0
        if 'hist_vol_20' in df.columns:
            hist_vol = df['hist_vol_20'].iloc[-1]
        else:
            hist_vol = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)

        # Apply filters

        # Filter 1: Confidence threshold
        if max_confidence < self.confidence_threshold:
            return {
                "signal_type": "NoTrade",
                "confidence": max_confidence,
                "reason": "LowConfidence"
            }

        # Filter 2: Volatility filter - avoid extremely high volatility
        if self.use_volatility_filter:
            # Calculate volatility percentile if we have enough data
            if len(df['hist_vol_20'].dropna()) > 50:
                vol_percentile = pd.Series(df['hist_vol_20'].dropna()).rank(pct=True).iloc[-1] * 100

                if vol_percentile > self.max_vol_percentile:
                    return {
                        "signal_type": "NoTrade",
                        "reason": "ExtremeVolatility",
                        "vol_percentile": vol_percentile
                    }

            # Alternative volatility check
            if volatility_regime > 0 and hist_vol > df['hist_vol_20'].dropna().mean() * 2:
                return {
                    "signal_type": "NoTrade",
                    "reason": "HighVolatility",
                    "volatility_regime": volatility_regime,
                    "hist_vol": hist_vol
                }

        # Filter 3: Trend filter
        has_trend_data = all(col in df.columns for col in ['h4_SMA_50', 'h4_SMA_200', 'h4_ADX'])

        if has_trend_data:
            adx_value = df['h4_ADX'].iloc[-1]
            trend_up = df['h4_SMA_50'].iloc[-1] > df['h4_SMA_200'].iloc[-1] and adx_value > self.min_adx_threshold
            trend_down = df['h4_SMA_50'].iloc[-1] < df['h4_SMA_200'].iloc[-1] and adx_value > self.min_adx_threshold

            # Only apply trend filter if we have strong trend
            if adx_value > self.min_adx_threshold:
                if P_positive > P_negative and not trend_up:
                    return {
                        "signal_type": "NoTrade",
                        "reason": "TrendMismatch",
                        "signal_direction": "bullish",
                        "trend_direction": "bearish" if trend_down else "neutral",
                        "adx": adx_value
                    }

                if P_negative > P_positive and not trend_down:
                    return {
                        "signal_type": "NoTrade",
                        "reason": "TrendMismatch",
                        "signal_direction": "bearish",
                        "trend_direction": "bullish" if trend_up else "neutral",
                        "adx": adx_value
                    }

        # Filter 4: Market regime filter
        if self.use_regime_filter and market_regime != 0:
            regime_aligned = (market_regime > 0 and P_positive > P_negative) or \
                             (market_regime < 0 and P_negative > P_positive)

            if not regime_aligned:
                return {
                    "signal_type": "NoTrade",
                    "reason": "RegimeMismatch",
                    "market_regime": market_regime,
                    "signal_direction": "bullish" if P_positive > P_negative else "bearish"
                }

        # Filter 5: Multi-timeframe confirmation
        timeframe_agreement_bull = False
        timeframe_agreement_bear = False

        if all(col in df.columns for col in ['d1_RSI_14', 'h4_RSI_14']):
            # Check alignment between daily and 4h RSI
            daily_rsi = df['d1_RSI_14'].iloc[-1]
            h4_rsi = df['h4_RSI_14'].iloc[-1]

            timeframe_agreement_bull = daily_rsi > 50 and h4_rsi > 50
            timeframe_agreement_bear = daily_rsi < 50 and h4_rsi < 50

        # Filter 6: Open Interest filter
        oi_signal = 0
        if self.use_oi_filter and 'oi_openInterest_change' in df.columns:
            # Check recent OI changes
            oi_changes = df['oi_openInterest_change'].iloc[-5:].values
            price_changes = df['close'].pct_change().iloc[-5:].values

            # Positive OI with positive price = bullish
            # Positive OI with negative price = bearish
            # Decreasing OI with price movement = lower conviction

            oi_increasing = np.mean(oi_changes) > 0
            price_increasing = np.mean(price_changes) > 0

            if oi_increasing and price_increasing:
                oi_signal = 1  # Bullish
            elif oi_increasing and not price_increasing:
                oi_signal = -1  # Bearish

            # Apply OI filter
            if (P_positive > P_negative and oi_signal < 0) or (P_negative > P_positive and oi_signal > 0):
                # Reduce confidence when OI indicates the opposite direction
                confidence_modifier = 0.8
                if P_positive > P_negative:
                    P_positive *= confidence_modifier
                else:
                    P_negative *= confidence_modifier

                # If confidence now below threshold, reject signal
                if max(P_positive, P_negative) < self.confidence_threshold:
                    return {
                        "signal_type": "NoTrade",
                        "reason": "OIFilter",
                        "oi_signal": oi_signal,
                        "adjusted_confidence": max(P_positive, P_negative)
                    }

        # Filter 7: Funding Rate filter
        funding_signal = 0
        if self.use_funding_filter and 'funding_fundingRate' in df.columns:
            # Check recent funding rates
            funding_rates = df['funding_fundingRate'].iloc[-3:].values
            avg_funding = np.mean(funding_rates)

            # High positive funding = bearish (traders paying premium to be long)
            # High negative funding = bullish (traders paying premium to be short)

            if avg_funding > 0.001:  # 0.1% per 8 hours is significant
                funding_signal = -1  # Bearish
            elif avg_funding < -0.001:
                funding_signal = 1  # Bullish

            # Apply funding filter only for strong opposite signals
            if abs(funding_signal) == 1:
                if (P_positive > P_negative and funding_signal < 0) or (P_negative > P_positive and funding_signal > 0):
                    # Reduce confidence when funding indicates the opposite direction
                    confidence_modifier = 0.8
                    if P_positive > P_negative:
                        P_positive *= confidence_modifier
                    else:
                        P_negative *= confidence_modifier

                    # If confidence now below threshold, reject signal
                    if max(P_positive, P_negative) < self.confidence_threshold:
                        return {
                            "signal_type": "NoTrade",
                            "reason": "FundingFilter",
                            "funding_signal": funding_signal,
                            "adjusted_confidence": max(P_positive, P_negative)
                        }

        # Generate signal
        if P_positive > P_negative:
            # For long signals

            # Apply timeframe agreement filter
            if not timeframe_agreement_bull:
                confidence_modifier = 0.9  # Reduce confidence when timeframes don't agree
                P_positive *= confidence_modifier

                # If confidence now below threshold, reject signal
                if P_positive < self.confidence_threshold:
                    return {
                        "signal_type": "NoTrade",
                        "reason": "TimeframeDisagreement",
                        "adjusted_confidence": P_positive
                    }

            # Dynamic stop loss based on volatility
            volatility_factor = 1.0 + (0.5 * volatility_regime)  # Increase for higher volatility
            stop_loss_price = current_price - (self.atr_multiplier_sl * atr * volatility_factor)

            # Determine signal strength
            if P_positive >= self.strong_signal_threshold:
                signal_str = "StrongBuy"
            else:
                signal_str = "Buy"

            return {
                "signal_type": signal_str,
                "confidence": float(P_positive),
                "stop_loss": round(float(stop_loss_price), 2),
                "regime": int(market_regime),
                "volatility": float(hist_vol),
                "timeframe_agreement": timeframe_agreement_bull,
                "oi_signal": oi_signal,
                "funding_signal": funding_signal,
                "derivative_signal": oi_signal + funding_signal,  # Combined derivative market signal
                "trend_strength": float(trend_strength),
                "adx": float(df['h4_ADX'].iloc[-1]) if 'h4_ADX' in df.columns else 0.0
            }

        elif P_negative > P_positive:
            # For short signals

            # Apply timeframe agreement filter
            if not timeframe_agreement_bear:
                confidence_modifier = 0.9  # Reduce confidence when timeframes don't agree
                P_negative *= confidence_modifier

                # If confidence now below threshold, reject signal
                if P_negative < self.confidence_threshold:
                    return {
                        "signal_type": "NoTrade",
                        "reason": "TimeframeDisagreement",
                        "adjusted_confidence": P_negative
                    }

            # Dynamic stop loss based on volatility
            volatility_factor = 1.0 + (0.5 * volatility_regime)  # Increase for higher volatility
            stop_loss_price = current_price + (self.atr_multiplier_sl * atr * volatility_factor)

            # Determine signal strength
            if P_negative >= self.strong_signal_threshold:
                signal_str = "StrongSell"
            else:
                signal_str = "Sell"

            return {
                "signal_type": signal_str,
                "confidence": float(P_negative),
                "stop_loss": round(float(stop_loss_price), 2),
                "regime": int(market_regime),
                "volatility": float(hist_vol),
                "timeframe_agreement": timeframe_agreement_bear,
                "oi_signal": oi_signal,
                "funding_signal": funding_signal,
                "derivative_signal": oi_signal + funding_signal,  # Combined derivative market signal
                "trend_strength": float(trend_strength),
                "adx": float(df['h4_ADX'].iloc[-1]) if 'h4_ADX' in df.columns else 0.0
            }

        # Default to no trade
        return {"signal_type": "NoTrade", "reason": "Indecision"}

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range.

        Args:
            df: DataFrame with OHLCV data
            period: Period for ATR calculation

        Returns:
            Series with ATR values
        """
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()

        # True Range is the maximum of the three
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # ATR is the moving average of True Range
        return tr.rolling(window=period).mean()

    def _detect_divergence(self, df: pd.DataFrame, lookback: int = 20,
                           threshold: float = 0.05) -> Tuple[bool, bool]:
        """Detect price-indicator divergence.

        Args:
            df: DataFrame with OHLCV and indicator data
            lookback: Number of periods to look back
            threshold: Threshold for divergence detection

        Returns:
            Tuple of (bullish divergence, bearish divergence)
        """
        if 'h4_RSI_14' not in df.columns or len(df) < lookback:
            return False, False

        # Get data for analysis
        price_subset = df['close'].iloc[-lookback:].values
        rsi_subset = df['h4_RSI_14'].iloc[-lookback:].values

        # Find local extremes
        price_highs = self._find_local_extremes(price_subset, find_max=True)
        price_lows = self._find_local_extremes(price_subset, find_max=False)
        rsi_highs = self._find_local_extremes(rsi_subset, find_max=True)
        rsi_lows = self._find_local_extremes(rsi_subset, find_max=False)

        # Check for bearish divergence (price making higher high but RSI making lower high)
        bearish_div = False
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            # Check if price is making higher highs
            if price_highs[-1][1] > price_highs[-2][1] * (1 + threshold):
                # Check if RSI is making lower highs
                if rsi_highs[-1][1] < rsi_highs[-2][1] * (1 - threshold):
                    bearish_div = True

        # Check for bullish divergence (price making lower low but RSI making higher low)
        bullish_div = False
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            # Check if price is making lower lows
            if price_lows[-1][1] < price_lows[-2][1] * (1 - threshold):
                # Check if RSI is making higher lows
                if rsi_lows[-1][1] > rsi_lows[-2][1] * (1 + threshold):
                    bullish_div = True

        return bullish_div, bearish_div

    def _find_local_extremes(self, data: np.ndarray, find_max: bool = True,
                             window: int = 3) -> List[Tuple[int, float]]:
        """Find local maximums or minimums in data.

        Args:
            data: Data array
            find_max: Whether to find maximums (True) or minimums (False)
            window: Window size for extreme detection

        Returns:
            List of (index, value) tuples for extremes
        """
        extremes = []

        for i in range(window, len(data) - window):
            # Get window of values
            left_window = data[i - window:i]
            right_window = data[i:i + window + 1]

            # Check if this point is an extreme
            if find_max:
                if data[i] == max(left_window) and data[i] == max(right_window):
                    extremes.append((i, data[i]))
            else:
                if data[i] == min(left_window) and data[i] == min(right_window):
                    extremes.append((i, data[i]))

        return extremes

    def should_close_position(self, position: Dict[str, Any], current_data: pd.DataFrame,
                              model_probs: np.ndarray) -> Dict[str, Any]:
        """Determine whether to close an existing position.

        Args:
            position: Position dictionary
            current_data: Current market data
            model_probs: Model prediction probabilities

        Returns:
            Dictionary with close decision
        """
        # Get current price
        current_price = current_data['close'].iloc[-1]

        # Get ATR for trailing stop calculation
        if 'd1_ATR_14' in current_data.columns:
            current_atr = current_data['d1_ATR_14'].iloc[-1]
        else:
            current_atr = self._compute_atr(current_data).iloc[-1]

        # Get position details
        entry_price = position['entry_price']
        direction = position['direction']
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)

        # Calculate current profit/loss
        if direction == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Check for stop loss hit
        stop_hit = (direction == 'long' and current_price <= stop_loss) or \
                   (direction == 'short' and current_price >= stop_loss)

        if stop_hit:
            return {
                "should_close": True,
                "reason": "StopLoss",
                "price": stop_loss
            }

        # Check for take profit hit
        tp_hit = (direction == 'long' and current_price >= take_profit) or \
                 (direction == 'short' and current_price <= take_profit)

        if tp_hit:
            return {
                "should_close": True,
                "reason": "TakeProfit",
                "price": take_profit
            }

        # Check for reversal signal
        # Get base probabilities
        P_positive = model_probs[3] + model_probs[4]  # Classes 3 & 4 = bullish
        P_negative = model_probs[0] + model_probs[1]  # Classes 0 & 1 = bearish

        signal_reversal = (
                                      direction == 'long' and P_negative > P_positive and P_negative > self.confidence_threshold) or \
                          (direction == 'short' and P_positive > P_negative and P_positive > self.confidence_threshold)

        if signal_reversal:
            return {
                "should_close": True,
                "reason": "SignalReversal",
                "price": current_price,
                "confidence": P_negative if direction == 'long' else P_positive
            }

        # Check for trailing stop adjustment
        if pnl_pct > 0.02:  # If 2% or more in profit
            # Calculate trailing stop distance
            atr_multiple = 2.0

            if direction == 'long':
                new_stop = current_price - (atr_multiple * current_atr)

                # Only update if it would move the stop higher
                if new_stop > stop_loss:
                    return {
                        "should_close": False,
                        "should_update_stop": True,
                        "new_stop": new_stop
                    }
            else:
                new_stop = current_price + (atr_multiple * current_atr)

                # Only update if it would move the stop lower
                if new_stop < stop_loss:
                    return {
                        "should_close": False,
                        "should_update_stop": True,
                        "new_stop": new_stop
                    }

        # No action needed
        return {
            "should_close": False
        }

    def adapt_to_market_conditions(self, recent_data: pd.DataFrame) -> None:
        """Adapt signal parameters to current market conditions.

        Args:
            recent_data: Recent market data
        """
        if len(recent_data) < 100:
            return

        # Detect current market regime
        if 'market_regime' in recent_data.columns:
            market_regime = recent_data['market_regime'].iloc[-20:].mean()

            if market_regime > 0.5:  # Strong uptrend
                # Bias toward bullish signals
                self.confidence_threshold = max(0.35, self.confidence_threshold * 0.9)
            elif market_regime < -0.5:  # Strong downtrend
                # Bias toward bearish signals
                self.confidence_threshold = max(0.35, self.confidence_threshold * 0.9)
            else:  # Ranging or weak trend
                # Require higher confidence
                self.confidence_threshold = min(0.45, self.confidence_threshold * 1.1)

        # Adapt to volatility conditions
        if 'volatility_regime' in recent_data.columns:
            volatility_regime = recent_data['volatility_regime'].iloc[-20:].mean()

            if volatility_regime > 0.5:  # High volatility
                # Increase stop loss multiplier
                self.atr_multiplier_sl = min(2.0, self.atr_multiplier_sl * 1.1)
                # Require higher confidence
                self.confidence_threshold = min(0.5, self.confidence_threshold * 1.1)
            elif volatility_regime < -0.5:  # Low volatility
                # Decrease stop loss multiplier
                self.atr_multiplier_sl = max(1.2, self.atr_multiplier_sl * 0.9)
                # Allow lower confidence
                self.confidence_threshold = max(0.35, self.confidence_threshold * 0.9)