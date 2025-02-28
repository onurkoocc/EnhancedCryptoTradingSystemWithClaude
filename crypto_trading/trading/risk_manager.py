"""
Risk management for cryptocurrency trading.

This module provides risk management functionality for cryptocurrency
trading, including position sizing, exposure management, and performance tracking.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any

import numpy as np

from ..utils.logging_utils import exception_handler


class AdvancedRiskManager:
    """Advanced risk management for cryptocurrency trading."""

    def __init__(self, initial_capital: float = 10000.0,
                 max_risk_per_trade: float = 0.02,
                 max_correlated_exposure: float = 0.06,
                 volatility_scaling: bool = True,
                 target_annual_vol: float = 0.2,
                 reward_risk_ratio: float = 2.5,
                 partial_close_ratio: float = 0.5,
                 max_drawdown_threshold: float = 0.20,
                 trade_frequency_limit: int = 12,  # per day
                 consecutive_loss_threshold: int = 3,
                 logger: Optional[logging.Logger] = None):
        """Initialize risk manager.

        Args:
            initial_capital: Initial capital amount
            max_risk_per_trade: Maximum risk per trade as fraction of capital
            max_correlated_exposure: Maximum correlated exposure as fraction of capital
            volatility_scaling: Whether to scale position size based on volatility
            target_annual_vol: Target annual volatility for portfolio
            reward_risk_ratio: Target reward/risk ratio for trades
            partial_close_ratio: Ratio for partial position closes
            max_drawdown_threshold: Maximum allowed drawdown
            trade_frequency_limit: Maximum number of trades per day
            consecutive_loss_threshold: Threshold for consecutive losses to reduce size
            logger: Logger to use
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_correlated_exposure = max_correlated_exposure
        self.volatility_scaling = volatility_scaling
        self.target_annual_vol = target_annual_vol
        self.reward_risk_ratio = reward_risk_ratio
        self.partial_close_ratio = partial_close_ratio
        self.max_drawdown_threshold = max_drawdown_threshold
        self.trade_frequency_limit = trade_frequency_limit
        self.consecutive_loss_threshold = consecutive_loss_threshold
        self.logger = logger or logging.getLogger('RiskManager')

        # Track positions and trade history
        self.open_positions = []
        self.trade_history = []
        self.performance_stats = self._init_performance_stats()

        # Risk state tracking
        self.current_drawdown = 0.0
        self.peak_capital = initial_capital
        self.consecutive_losses = 0
        self.risk_multiplier = 1.0
        self.last_trade_time = None
        self.trades_today = 0
        self.last_trade_day = None

    def _init_performance_stats(self) -> Dict[str, Any]:
        """Initialize performance statistics dictionary.

        Returns:
            Dictionary with empty performance statistics
        """
        return {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'drawdown': 0.0,
            'max_drawdown': 0.0,
            'returns': [],
            'equity_curve': [self.initial_capital],
            'win_count': 0,
            'loss_count': 0,
            'total_trades': 0,
            'avg_holding_time': timedelta(hours=0),
            'total_profit': 0.0,
            'total_loss': 0.0,
            'expectancy': 0.0
        }

    @exception_handler(reraise=False, fallback_value=0.0)
    def calculate_position_size(self, signal: Dict[str, Any], entry_price: float,
                                stop_loss: float, volatility_regime: float = 0) -> float:
        """Calculate optimal position size based on risk parameters.

        Args:
            signal: Signal dictionary with trade information
            entry_price: Entry price
            stop_loss: Stop loss price
            volatility_regime: Volatility regime indicator

        Returns:
            Position size (quantity)
        """
        if entry_price == stop_loss:
            self.logger.warning("Entry price equals stop loss, cannot calculate position size")
            return 0

        # Base risk amount as % of current capital
        base_risk_pct = self.max_risk_per_trade

        # Apply risk multiplier from consecutive loss logic
        base_risk_pct *= self.risk_multiplier

        # Get signal confidence and strength
        confidence = signal.get('confidence', 0.5)
        signal_type = signal.get('signal_type', '')
        strong_signal = 'Strong' in signal_type

        # Adjust risk based on signal strength/confidence
        if strong_signal:
            base_risk_pct *= min(1.25, 1.0 + confidence * 0.5)  # Increase risk for strong signals
        else:
            base_risk_pct *= min(1.0, 0.8 + confidence * 0.4)  # Decrease risk for weak signals

        # Adjust risk based on recent performance
        if len(self.trade_history) >= 10:
            recent_trades = self.trade_history[-10:]
            win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)

            # Reduce risk after consecutive losses
            if self.consecutive_losses >= self.consecutive_loss_threshold:
                base_risk_pct *= 0.5  # Cut risk by half after threshold consecutive losses
            elif win_rate < 0.4:
                base_risk_pct *= 0.7  # Reduce risk on low win rate
            elif win_rate > 0.6:
                base_risk_pct *= 1.2  # Increase risk on high win rate, but don't exceed maximum

        # Volatility scaling
        if self.volatility_scaling:
            vol_factor = 1.0
            if volatility_regime > 0:  # High volatility
                vol_factor = 0.7
            elif volatility_regime < 0:  # Low volatility
                vol_factor = 1.3
            base_risk_pct *= vol_factor

        # Limit risk during drawdowns
        if self.current_drawdown > 0.1:  # If in >10% drawdown
            drawdown_factor = 1.0 - (self.current_drawdown / self.max_drawdown_threshold)
            drawdown_factor = max(0.5, drawdown_factor)  # Don't reduce below 50%
            base_risk_pct *= drawdown_factor

        # Ensure we don't exceed max risk per trade
        risk_pct = min(base_risk_pct, self.max_risk_per_trade)

        # Calculate risk amount in currency
        risk_amount = self.current_capital * risk_pct

        # Calculate position size based on distance to stop loss
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
        else:
            self.logger.warning("Risk per unit is zero, cannot calculate position size")
            position_size = 0

        # Log position sizing decision
        self.logger.info(
            f"Position size calculation: Capital={self.current_capital:.2f}, "
            f"Risk %={risk_pct:.2%}, Risk $={risk_amount:.2f}, "
            f"Entry={entry_price:.2f}, SL={stop_loss:.2f}, "
            f"Size={position_size:.8f}"
        )

        return position_size

    @exception_handler(reraise=False, fallback_value=(False, 0.0))
    def check_correlation_risk(self, new_position: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if adding this position would exceed maximum risk for correlated assets.

        Args:
            new_position: New position dictionary

        Returns:
            Tuple of (allowed, adjusted risk fraction)
        """
        # Calculate current exposure
        current_exposure = sum(pos['risk_amount'] for pos in self.open_positions)
        current_exposure_pct = current_exposure / self.current_capital

        # Default risk amount from position
        position_risk_pct = self.max_risk_per_trade

        # If we have risk_amount in the position, convert to percentage
        if 'risk_amount' in new_position:
            position_risk_pct = new_position['risk_amount'] / self.current_capital

        # Check if new position would violate max exposure
        if current_exposure_pct + position_risk_pct > self.max_correlated_exposure:
            # Calculate how much more exposure we can add
            remaining_exposure_pct = max(0, self.max_correlated_exposure - current_exposure_pct)

            if remaining_exposure_pct <= 0:
                self.logger.warning(
                    f"Maximum exposure ({self.max_correlated_exposure:.2%}) reached. "
                    f"Current: {current_exposure_pct:.2%}"
                )
                return False, 0.0

            self.logger.info(
                f"Reduced position risk from {position_risk_pct:.2%} to {remaining_exposure_pct:.2%} "
                f"due to exposure limits"
            )
            return True, remaining_exposure_pct

        return True, position_risk_pct

    @exception_handler(reraise=False, fallback_value={"update_stop": False})
    def dynamic_exit_strategy(self, position: Dict[str, Any], current_price: float,
                              current_atr: float) -> Dict[str, Any]:
        """Implement dynamic exit strategy with trailing stops and partial exits.

        Args:
            position: Position dictionary
            current_price: Current price
            current_atr: Current ATR value

        Returns:
            Dictionary with exit instructions
        """
        # Extract position details
        entry_price = position['entry_price']
        direction = position['direction']
        initial_stop = position['stop_loss']
        take_profit = position.get('take_profit', 0)

        # Calculate current profit/loss percentage
        if direction == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Initialize response
        response = {"update_stop": False}

        # Determine trailing stop parameters
        atr_multiple = 2.0  # Base ATR multiple for trailing stop

        # Rules for trailing stop adjustment
        if direction == 'long':
            # For long positions
            if current_price > entry_price * 1.02:  # If 2% or more in profit
                # Trail with ATR-based stop
                new_stop = current_price - (atr_multiple * current_atr)

                # Only move stop if it would move it higher
                if new_stop > initial_stop:
                    response["update_stop"] = True
                    response["new_stop"] = new_stop
                    response["reason"] = "TrailingStop"

            # First profit target - exit 1/3 position
            if current_price >= entry_price * 1.02 and not position.get('partial_exit_1', False):
                response["partial_exit"] = True
                response["exit_ratio"] = self.partial_close_ratio / 2  # e.g., 0.25 if self.partial_close_ratio is 0.5
                response["reason"] = "FirstTarget"

            # Second profit target - exit another portion and move stop to breakeven
            if current_price >= entry_price * 1.04 and not position.get('partial_exit_2', False):
                response["partial_exit"] = True
                response["exit_ratio"] = self.partial_close_ratio  # e.g., 0.5
                response["update_stop"] = True
                response["new_stop"] = max(entry_price, initial_stop)  # Move to at least breakeven
                response["reason"] = "SecondTarget"
        else:
            # For short positions
            if current_price < entry_price * 0.98:  # If 2% or more in profit
                # Trail with ATR-based stop
                new_stop = current_price + (atr_multiple * current_atr)

                # Only move stop if it would move it lower
                if new_stop < initial_stop:
                    response["update_stop"] = True
                    response["new_stop"] = new_stop
                    response["reason"] = "TrailingStop"

            # First profit target - exit 1/3 position
            if current_price <= entry_price * 0.98 and not position.get('partial_exit_1', False):
                response["partial_exit"] = True
                response["exit_ratio"] = self.partial_close_ratio / 2  # e.g., 0.25 if self.partial_close_ratio is 0.5
                response["reason"] = "FirstTarget"

            # Second profit target - exit another portion and move stop to breakeven
            if current_price <= entry_price * 0.96 and not position.get('partial_exit_2', False):
                response["partial_exit"] = True
                response["exit_ratio"] = self.partial_close_ratio  # e.g., 0.5
                response["update_stop"] = True
                response["new_stop"] = min(entry_price, initial_stop)  # Move to at least breakeven
                response["reason"] = "SecondTarget"

        # Additional rule: If drawdown is high, tighten stops
        if self.current_drawdown > 0.15:  # If in >15% drawdown
            if "update_stop" in response and response["update_stop"]:
                # Make stop even tighter in drawdown
                if direction == 'long':
                    response["new_stop"] = current_price - (atr_multiple * current_atr * 0.75)  # 25% tighter
                else:
                    response["new_stop"] = current_price + (atr_multiple * current_atr * 0.75)  # 25% tighter

                response["reason"] += "_Drawdown"

        return response

    @exception_handler(reraise=True)
    def update_position(self, position_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing position with new information.

        Args:
            position_id: Position ID to update
            updates: Dictionary with updates to apply

        Returns:
            Updated position dictionary
        """
        # Find position by ID
        for i, position in enumerate(self.open_positions):
            if position.get('id') == position_id:
                # Apply updates
                for key, value in updates.items():
                    position[key] = value

                # Log the update
                self.logger.info(f"Updated position {position_id}: {updates}")

                return position

        self.logger.warning(f"Position {position_id} not found")
        return {}

    @exception_handler(reraise=True)
    def add_position(self, position: Dict[str, Any]) -> int:
        """Add a new position to the portfolio.

        Args:
            position: Position dictionary

        Returns:
            Position ID
        """
        # Check rate limiting
        if not self._check_trade_frequency_limit():
            self.logger.warning("Trade frequency limit reached")
            return -1

        # Generate position ID
        position_id = len(self.trade_history) + len(self.open_positions) + 1

        # Add id and timestamp
        position['id'] = position_id
        position['entry_time'] = position.get('entry_time', datetime.now())

        # Add risk amount if not provided
        if 'risk_amount' not in position:
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            quantity = position['quantity']

            risk_amount = quantity * abs(entry_price - stop_loss)
            position['risk_amount'] = risk_amount

        # Add position to open positions
        self.open_positions.append(position)

        # Update trade frequency tracking
        self._update_trade_frequency()

        # Log new position
        self.logger.info(f"Added position {position_id}: {position}")

        return position_id

    @exception_handler(reraise=True)
    def close_position(self, position_id: int, exit_price: float, exit_time: Optional[datetime] = None,
                       exit_reason: str = "Manual") -> Dict[str, Any]:
        """Close an open position and record the trade.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_time: Exit time (default: current time)
            exit_reason: Reason for exit

        Returns:
            Closed position dictionary with trade results
        """
        # Set exit time if not provided
        if exit_time is None:
            exit_time = datetime.now()

        # Find position by ID
        closed_position = None
        for i, position in enumerate(self.open_positions):
            if position.get('id') == position_id:
                closed_position = position

                # Calculate P&L
                entry_price = position['entry_price']
                quantity = position['quantity']
                direction = position['direction']

                if direction == 'long':
                    pnl = quantity * (exit_price - entry_price)
                else:
                    pnl = quantity * (entry_price - exit_price)

                # Add exit information
                position['exit_price'] = exit_price
                position['exit_time'] = exit_time
                position['exit_reason'] = exit_reason
                position['pnl'] = pnl
                position['pnl_percent'] = pnl / (entry_price * quantity) * 100
                position['holding_time'] = exit_time - position['entry_time']

                # Add to trade history
                self.trade_history.append(position)

                # Remove from open positions
                self.open_positions.pop(i)

                # Update capital
                self.current_capital += pnl

                # Update peak capital if appropriate
                if self.current_capital > self.peak_capital:
                    self.peak_capital = self.current_capital

                # Update drawdown
                self.current_drawdown = 1 - (self.current_capital / self.peak_capital)

                # Update consecutive win/loss count
                if pnl > 0:
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1

                # Update risk multiplier based on consecutive losses
                if self.consecutive_losses >= self.consecutive_loss_threshold:
                    self.risk_multiplier = max(0.5,
                                               1.0 - (self.consecutive_losses - self.consecutive_loss_threshold) * 0.1)
                else:
                    self.risk_multiplier = 1.0

                # Log the closed trade
                self.logger.info(
                    f"Closed position {position_id}: {direction} {quantity:.8f} units, "
                    f"Entry={entry_price:.2f}, Exit={exit_price:.2f}, "
                    f"P&L=${pnl:.2f} ({position['pnl_percent']:.2f}%), "
                    f"Reason={exit_reason}"
                )

                # Update performance stats
                self.update_performance_stats()

                return position

        self.logger.warning(f"Position {position_id} not found")
        return {}

    def _check_trade_frequency_limit(self) -> bool:
        """Check if we've exceeded the trade frequency limit.

        Returns:
            True if we can make another trade, False otherwise
        """
        now = datetime.now()
        today = now.date()

        # Reset counter if it's a new day
        if self.last_trade_day != today:
            self.trades_today = 0
            self.last_trade_day = today

        # Check if we've reached the limit
        if self.trades_today >= self.trade_frequency_limit:
            return False

        return True

    def _update_trade_frequency(self) -> None:
        """Update trade frequency tracking after a new trade."""
        now = datetime.now()
        today = now.date()

        # Initialize if first trade
        if self.last_trade_day is None:
            self.last_trade_day = today
            self.trades_today = 1
        # Reset if new day
        elif self.last_trade_day != today:
            self.last_trade_day = today
            self.trades_today = 1
        # Increment if same day
        else:
            self.trades_today += 1

        self.last_trade_time = now

    @exception_handler(reraise=False)
    def update_performance_stats(self) -> Dict[str, Any]:
        """Update performance statistics based on trade history.

        Returns:
            Dictionary with updated performance statistics
        """
        if not self.trade_history:
            return self.performance_stats

        # Reset stats
        stats = self._init_performance_stats()

        # Extract wins and losses
        wins = [t for t in self.trade_history if t['pnl'] > 0]
        losses = [t for t in self.trade_history if t['pnl'] <= 0]

        # Basic statistics
        stats['win_count'] = len(wins)
        stats['loss_count'] = len(losses)
        stats['total_trades'] = len(self.trade_history)
        stats['win_rate'] = len(wins) / len(self.trade_history) if self.trade_history else 0

        # Calculate profit/loss metrics
        if wins:
            stats['avg_win'] = sum(t['pnl'] for t in wins) / len(wins)
            stats['largest_win'] = max(t['pnl'] for t in wins)
            stats['total_profit'] = sum(t['pnl'] for t in wins)

        if losses:
            stats['avg_loss'] = sum(t['pnl'] for t in losses) / len(losses)
            stats['largest_loss'] = min(t['pnl'] for t in losses)
            stats['total_loss'] = abs(sum(t['pnl'] for t in losses))

        # Calculate profit factor
        stats['profit_factor'] = (
            stats['total_profit'] / stats['total_loss']
            if stats['total_loss'] > 0 else float('inf')
        )

        # Calculate expectancy (average P&L per trade)
        stats['expectancy'] = sum(t['pnl'] for t in self.trade_history) / len(self.trade_history)

        # Calculate average holding time
        if self.trade_history:
            holding_times = [
                (t['exit_time'] - t['entry_time']).total_seconds() / 3600  # in hours
                for t in self.trade_history
            ]
            avg_hours = sum(holding_times) / len(holding_times)
            stats['avg_holding_time'] = timedelta(hours=avg_hours)

        # Build equity curve and calculate drawdown
        equity_curve = [self.initial_capital]
        peak_equity = self.initial_capital
        drawdowns = [0]

        for trade in self.trade_history:
            equity = equity_curve[-1] + trade['pnl']
            equity_curve.append(equity)

            if equity > peak_equity:
                peak_equity = equity

            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            drawdowns.append(drawdown)

        stats['equity_curve'] = equity_curve
        stats['drawdown'] = drawdowns[-1]
        stats['max_drawdown'] = max(drawdowns)

        # Calculate returns for Sharpe ratio
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(ret)

        stats['returns'] = returns

        # Calculate Sharpe ratio (annualized)
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = np.std(returns)

            if std_return > 0:
                # Estimate average trades per day
                if len(self.trade_history) >= 2:
                    first_trade = self.trade_history[0]['entry_time']
                    last_trade = self.trade_history[-1]['exit_time']
                    days = (last_trade - first_trade).total_seconds() / (24 * 3600)
                    trades_per_day = len(self.trade_history) / max(1, days)

                    # Calculate annualization factor
                    annualization_factor = np.sqrt(max(252, trades_per_day * 252))

                    # Calculate annualized Sharpe ratio
                    stats['sharpe_ratio'] = avg_return / std_return * annualization_factor

        # Update the performance stats
        self.performance_stats = stats

        return stats

    def get_risk_analysis(self) -> Dict[str, Any]:
        """Get comprehensive risk analysis of the current portfolio.

        Returns:
            Dictionary with risk analysis
        """
        # Update performance stats first
        self.update_performance_stats()

        # Current allocation
        total_exposure = sum(p['quantity'] * p['entry_price'] for p in self.open_positions)
        exposure_pct = total_exposure / self.current_capital if self.current_capital > 0 else 0

        # Current risk exposure
        total_risk = sum(p['risk_amount'] for p in self.open_positions)
        risk_pct = total_risk / self.current_capital if self.current_capital > 0 else 0

        # Direction balance
        long_positions = [p for p in self.open_positions if p['direction'] == 'long']
        short_positions = [p for p in self.open_positions if p['direction'] == 'short']

        long_exposure = sum(p['quantity'] * p['entry_price'] for p in long_positions)
        short_exposure = sum(p['quantity'] * p['entry_price'] for p in short_positions)

        net_exposure = long_exposure - short_exposure
        net_exposure_pct = net_exposure / self.current_capital if self.current_capital > 0 else 0

        # Concentration risk
        max_position_size = max(
            [p['quantity'] * p['entry_price'] for p in self.open_positions]) if self.open_positions else 0
        concentration_pct = max_position_size / self.current_capital if self.current_capital > 0 else 0

        # Position count
        position_count = len(self.open_positions)

        # Current drawdown
        peak_capital = max(self.peak_capital, self.current_capital)
        current_drawdown = 1 - (self.current_capital / peak_capital) if peak_capital > 0 else 0

        # Return comprehensive analysis
        return {
            # Portfolio metrics
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'current_drawdown': current_drawdown,
            'current_drawdown_pct': current_drawdown * 100,
            'max_drawdown_pct': self.performance_stats['max_drawdown'] * 100,

            # Exposure metrics
            'total_exposure': total_exposure,
            'exposure_pct': exposure_pct * 100,
            'total_risk': total_risk,
            'risk_pct': risk_pct * 100,
            'position_count': position_count,

            # Directional exposure
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': net_exposure,
            'net_exposure_pct': net_exposure_pct * 100,
            'directional_bias': 'long' if net_exposure > 0 else 'short' if net_exposure < 0 else 'neutral',

            # Concentration risk
            'max_position_size': max_position_size,
            'concentration_pct': concentration_pct * 100,

            # Risk state
            'consecutive_losses': self.consecutive_losses,
            'risk_multiplier': self.risk_multiplier,
            'trades_today': self.trades_today,
            'trade_frequency_limit': self.trade_frequency_limit
        }

    def adjust_parameters_for_performance(self) -> Dict[str, Any]:
        """Automatically adjust risk parameters based on performance.

        Returns:
            Dictionary with adjustments made
        """
        # Get current performance stats
        stats = self.performance_stats
        adjustments = {}

        # Only adjust if we have enough trades
        if stats['total_trades'] < 10:
            return {'message': 'Not enough trades for adjustment'}

        # Adjust max risk per trade based on win rate and drawdown
        original_max_risk = self.max_risk_per_trade

        # Decrease risk if in drawdown or low win rate
        if stats['drawdown'] > 0.1 or stats['win_rate'] < 0.4:
            self.max_risk_per_trade = max(0.01, self.max_risk_per_trade * 0.8)
            adjustments['max_risk_per_trade'] = f"{original_max_risk:.2%} → {self.max_risk_per_trade:.2%}"

        # Increase risk if doing well
        elif stats['drawdown'] < 0.05 and stats['win_rate'] > 0.55 and stats['profit_factor'] > 1.5:
            self.max_risk_per_trade = min(0.03, self.max_risk_per_trade * 1.2)
            adjustments['max_risk_per_trade'] = f"{original_max_risk:.2%} → {self.max_risk_per_trade:.2%}"

        # Adjust reward/risk ratio based on avg win/loss ratio
        original_reward_risk = self.reward_risk_ratio

        if stats['avg_win'] > 0 and stats['avg_loss'] < 0:
            win_loss_ratio = abs(stats['avg_win'] / stats['avg_loss']) if stats['avg_loss'] != 0 else 2.0

            # If our targets are too aggressive compared to reality
            if win_loss_ratio < self.reward_risk_ratio * 0.7:
                self.reward_risk_ratio = max(1.5, win_loss_ratio * 1.1)
                adjustments['reward_risk_ratio'] = f"{original_reward_risk:.1f} → {self.reward_risk_ratio:.1f}"

            # If we're consistently doing better than our targets
            elif win_loss_ratio > self.reward_risk_ratio * 1.3:
                self.reward_risk_ratio = min(4.0, win_loss_ratio * 0.9)
                adjustments['reward_risk_ratio'] = f"{original_reward_risk:.1f} → {self.reward_risk_ratio:.1f}"

        # Adjust partial close ratio based on holding time
        original_partial_close = self.partial_close_ratio

        if isinstance(stats['avg_holding_time'], timedelta):
            avg_hours = stats['avg_holding_time'].total_seconds() / 3600

            # If trades are closing too quickly, increase partial close ratio
            if avg_hours < 12 and self.partial_close_ratio < 0.6:
                self.partial_close_ratio = min(0.75, self.partial_close_ratio * 1.2)
                adjustments['partial_close_ratio'] = f"{original_partial_close:.2f} → {self.partial_close_ratio:.2f}"

            # If trades are staying open too long, decrease partial close ratio
            elif avg_hours > 48 and self.partial_close_ratio > 0.4:
                self.partial_close_ratio = max(0.25, self.partial_close_ratio * 0.8)
                adjustments['partial_close_ratio'] = f"{original_partial_close:.2f} → {self.partial_close_ratio:.2f}"

        if not adjustments:
            adjustments['message'] = 'No parameter adjustments needed'

        return adjustments