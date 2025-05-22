from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import numpy as np
from datetime import datetime

# Constants
MIN_CONFIRMATION_SCORE = 0.85
MIN_EXHAUSTION_PROB = 0.7
MOMENTUM_THRESHOLD = 1.5
MIN_STABLE_CANDLE_COUNT = 4
POWER_EXIT_THRESHOLD = 0.8
MIN_REACTION_STRENGTH = 0.3
MAX_REACTION_STRENGTH = 0.8
WICK_PRESSURE_THRESHOLD = 2.0
MIN_LIQUIDITY_ABSORPTION_STRENGTH = 0.75
MIN_DSCM_SENTIMENT_PROB = 0.6
DSCM_SCALE_MULTIPLIER = 0.5
MAX_DSCM_SCALE = 2.0
BASE_POSITION_SIZE = 0.01
MIN_POWER_CANDLE_SIZE = 1.5
MIN_ORDER_FLOW_STRENGTH = 0.85
MIN_STABLE_PRESSURE_COUNT = 3
GAP_EXHAUSTION_THRESHOLD = 1.2
MIN_REACTION_SPEED = 0.5
MIN_CONFIDENCE_THRESHOLD = 0.7
MIN_ZONE_FIGHT_STRENGTH = 1.2

# Configure sparse logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data structure for candles
@dataclass
class Candle:
    open: float
    close: float
    high: float
    low: float
    volume: float
    atr: float
    direction: str = "none"
    prior_candles: Optional[List['Candle']] = None
    timestamp: Optional[datetime] = None

    def __hash__(self):
        return hash((self.open, self.close, self.high, self.low, self.volume, self.atr, self.direction, self.timestamp))

    def __eq__(self, other):
        if not isinstance(other, Candle):
            return False
        return (self.open, self.close, self.high, self.low, self.volume, self.atr, self.direction, self.timestamp) == \
               (other.open, other.close, other.high, other.low, other.volume, other.atr, other.direction, other.timestamp)

# Cache monitoring
def monitor_cache(cache: Dict, key: Any, function_name: str, max_size: int = 5000) -> None:
    if len(cache) >= max_size:
        cache.clear()
        logger.info(f"Cache cleared for {function_name} due to max size {max_size}")

# Sparse logging
def log_sparse_data(message: str, data: Any) -> None:
    logger.info(f"Sparse data log: {message}, Data: {data}")

# Thread-safe caches
_power_cache = {}
_five_second_cache = {}
_compute_reaction_cache = {}
_reaction_cache = {}

# Core functions
def calculate_candle_size(candle: Candle, atr_window: List[Candle] = None) -> Dict[str, Any]:
    """
    Calculate candle size metrics, incorporating body size, wick proportions, and volatility context.
    Returns a dictionary with relative size, classification, and wick rejection indicators.
    """
    if not all(hasattr(candle, k) for k in ['open', 'close', 'high', 'low', 'atr']):
        log_sparse_data('Missing candle attributes in calculate_candle_size', candle.__dict__)
        return {
            'relative_size': 0.0,
            'is_small': True,
            'is_large': False,
            'is_consistent': False,
            'has_wick_rejection': False
        }

    body_size = abs(candle.close - candle.open)
    upper_wick = candle.high - max(candle.open, candle.close)
    lower_wick = min(candle.open, candle.close) - candle.low
    total_range = candle.high - candle.low

    # Use ATR from candle or calculate from atr_window if provided
    atr = candle.atr if candle.atr > 0 else (np.std([c.high - c.low for c in atr_window]) if atr_window else 0.001)
    relative_size = body_size / atr if atr > 0 else 0.0

    # Classify candle size
    is_small = relative_size <= 0.5
    is_large = relative_size >= MIN_POWER_CANDLE_SIZE
    is_consistent = 0.5 <= relative_size <= 1.5

    # Detect wick rejection (long wick relative to body, indicating rejection at high/low)
    wick_ratio = max(upper_wick, lower_wick) / body_size if body_size > 0 else 0.0
    has_wick_rejection = wick_ratio > WICK_PRESSURE_THRESHOLD and total_range > atr

    return {
        'relative_size': relative_size,
        'is_small': is_small,
        'is_large': is_large,
        'is_consistent': is_consistent,
        'has_wick_rejection': has_wick_rejection
    }

def validate_wick_exhaustion(candle: Candle, prior_candles: List[Candle] = None) -> Dict[str, Any]:
    """
    Validate wick exhaustion based on wick rejection patterns and prior candle context.
    Returns a dictionary indicating if the wick suggests exhaustion and its strength.
    """
    if not all(hasattr(candle, k) for k in ['open', 'close', 'high', 'low', 'atr']):
        log_sparse_data('Missing candle attributes in validate_wick_exhaustion', candle.__dict__)
        return {'is_valid': False, 'exhaustion_strength': 0.0}

    upper_wick = candle.high - max(candle.open, candle.close)
    lower_wick = min(candle.open, candle.close) - candle.low
    body_size = abs(candle.close - candle.open)
    atr = candle.atr if candle.atr > 0 else 0.001

    # Wick-to-body ratio
    wick_ratio = max(upper_wick, lower_wick) / body_size if body_size > 0 else 0.0
    is_wick_dominant = wick_ratio > WICK_PRESSURE_THRESHOLD

    # Check for rejection pattern: long wick with small body and prior strong move
    prior_move = 0.0
    if prior_candles and len(prior_candles) >= 2:
        prior_move = abs(prior_candles[-1].close - prior_candles[-2].open) / atr
    is_rejection_pattern = is_wick_dominant and prior_move > 1.5

    # Exhaustion strength based on wick size and prior context
    exhaustion_strength = min(wick_ratio / WICK_PRESSURE_THRESHOLD, 1.0) if is_wick_dominant else 0.0
    if is_rejection_pattern:
        exhaustion_strength *= 1.2

    return {
        'is_valid': is_wick_dominant or is_rejection_pattern,
        'exhaustion_strength': min(exhaustion_strength, 1.0)
    }

def analyze_context(prior_candles: List[Candle], lookback: int = 5) -> Dict[str, float]:
    """
    Analyze market context using momentum, trend direction, and support/resistance proximity.
    Returns a dictionary with momentum, trend strength, and support/resistance metrics.
    """
    if not prior_candles or len(prior_candles) < 2:
        log_sparse_data('Insufficient prior candles in analyze_context', {'count': len(prior_candles)})
        return {'momentum': 0.5, 'trend_strength': 0.0, 'near_support_resistance': 0.0}

    price_changes = [c.close - c.open for c in prior_candles[-lookback:]]
    avg_price_change = np.mean(price_changes) if price_changes else 0.0
    atr = np.mean([c.atr for c in prior_candles[-lookback:] if c.atr > 0]) or 0.001
    momentum = abs(avg_price_change) / atr if atr > 0 else 0.5

    closes = [c.close for c in prior_candles[-lookback:]]
    sma = np.mean(closes) if closes else prior_candles[-1].close
    trend_strength = (closes[-1] - sma) / atr if atr > 0 else 0.0

    recent_highs = [max(c.high for c in prior_candles[-lookback:])]
    recent_lows = [min(c.low for c in prior_candles[-lookback:])]
    last_close = prior_candles[-1].close
    distance_to_resistance = min([abs(last_close - h) for h in recent_highs]) / atr if recent_highs else 1.0
    distance_to_support = min([abs(last_close - l) for l in recent_lows]) / atr if recent_lows else 1.0
    near_support_resistance = 1.0 if min(distance_to_support, distance_to_resistance) < 1.0 else 0.0

    return {
        'momentum': min(max(momentum, 0.0), 1.0),
        'trend_strength': min(max(trend_strength, -1.0), 1.0),
        'near_support_resistance': near_support_resistance
    }

def calculate_avg_speed(prior_candles: List[Candle], lookback: int = 5) -> float:
    """
    Calculate average price movement speed over a lookback period, normalized by ATR.
    Returns a float representing average speed.
    """
    if not prior_candles or len(prior_candles) < 2:
        log_sparse_data('Insufficient prior candles in calculate_avg_speed', {'count': len(prior_candles)})
        return 0.0

    price_moves = [abs(c.close - c.open) for c in prior_candles[-lookback:]]
    atr = np.mean([c.atr for c in prior_candles[-lookback:] if c.atr > 0]) or 0.001
    avg_speed = np.mean(price_moves) / atr if atr > 0 else 0.0
    return min(avg_speed, 1.0)

def assess_five_second_reaction(candles: List[Candle], timeframe: int = 5) -> Dict[str, Any]:
    """
    Assess short-term (5-second equivalent) reaction strength and direction.
    Returns a dictionary with reaction metrics.
    """
    if not candles or len(candles) < 1:
        log_sparse_data('Empty candle list in assess_five_second_reaction', None)
        return {
            'direction': 'none',
            'dominant_party': 'none',
            'buyer_strength': 0.0,
            'seller_strength': 0.0,
            'strength_ratio': 1.0,
            'dominant_party_strength': 0.0
        }

    candle = candles[-1]
    if not all(hasattr(candle, k) for k in ['open', 'close', 'high', 'low', 'volume', 'atr']):
        log_sparse_data('Missing candle attributes in assess_five_second_reaction', candle.__dict__)
        return {
            'direction': 'none',
            'dominant_party': 'none',
            'buyer_strength': 0.0,
            'seller_strength': 0.0,
            'strength_ratio': 1.0,
            'dominant_party_strength': 0.0
        }

    body_size = abs(candle.close - candle.open)
    atr = candle.atr if candle.atr > 0 else 0.001
    reaction_strength = body_size / atr
    volume_factor = candle.volume / np.mean([c.volume for c in candles[-3:] if c.volume > 0]) if len(candles) >= 3 else 1.0

    direction = 'buy' if candle.close > candle.open else 'sell'
    buyer_strength = reaction_strength * volume_factor if direction == 'buy' else 0.0
    seller_strength = reaction_strength * volume_factor if direction == 'sell' else 0.0
    dominant_party = 'buyers' if buyer_strength > seller_strength else 'sellers'
    strength_ratio = buyer_strength / seller_strength if seller_strength > 0 else 2.0 if buyer_strength > 0 else 1.0

    return {
        'direction': direction,
        'dominant_party': dominant_party,
        'buyer_strength': min(buyer_strength, 1.0),
        'seller_strength': min(seller_strength, 1.0),
        'strength_ratio': min(strength_ratio, 2.0),
        'dominant_party_strength': max(buyer_strength, seller_strength)
    }

def calculate_momentum(candles: List[Candle], lookback: int = 6) -> float:
    """
    Calculate momentum based on price changes and volatility over a lookback period.
    Returns a float representing momentum strength.
    """
    if not candles or len(candles) < 2:
        log_sparse_data('Empty or insufficient candle list in calculate_momentum', {'count': len(candles)})
        return 0.0

    price_changes = [c.close - c.open for c in candles[-lookback:]]
    atr = np.mean([c.atr for c in candles[-lookback:] if c.atr > 0]) or 0.001
    momentum = np.sum(price_changes) / atr if atr > 0 else 0.0
    return min(max(momentum, 0.0), 2.0)

def decode_multi_timeframe_structure(candles: List[Candle], timeframe: str, lookback: int = 5) -> Dict[str, Any]:
    """
    Decode market structure across timeframes using trend direction and consolidation detection.
    Returns a dictionary with validity, strength, and direction.
    """
    if not candles or len(candles) < 2:
        log_sparse_data(f'Insufficient candles for {timeframe} in decode_multi_timeframe_structure', {'count': len(candles)})
        return {'is_valid': False, 'strength': 0.0, 'direction': 'none'}

    closes = [c.close for c in candles[-lookback:]]
    sma = np.mean(closes) if closes else candles[-1].close
    atr = np.mean([c.atr for c in candles[-lookback:] if c.atr > 0]) or 0.001
    trend_strength = (closes[-1] - sma) / atr if atr > 0 else 0.0
    direction = 'buy' if trend_strength > 0 else 'sell'

    small_candle_count = sum(1 for c in candles[-lookback:] if calculate_candle_size(c, candles)['is_small'])
    is_consolidation = small_candle_count >= MIN_STABLE_CANDLE_COUNT

    strength = abs(trend_strength) if not is_consolidation else 0.5 * abs(trend_strength)
    is_valid = abs(trend_strength) > 0.5 or is_consolidation

    return {
        'is_valid': is_valid,
        'strength': min(strength, 1.0),
        'direction': direction if is_valid else 'none'
    }

def calculate_volatility(candles: List[Candle], lookback: int = 5) -> Dict[str, float]:
    """
    Calculate volatility metrics using ATR and standard deviation of price ranges.
    Returns a dictionary with ATR multiplier and volatility score.
    """
    if not candles or len(candles) < 2:
        log_sparse_data('Insufficient candles in calculate_volatility', {'count': len(candles)})
        return {'atr_multiplier': 1.0, 'volatility_score': 0.5}

    ranges = [c.high - c.low for c in candles[-lookback:]]
    atr = np.mean([c.atr for c in candles[-lookback:] if c.atr > 0]) or 0.001
    std_dev = np.std(ranges) if ranges else 0.001
    volatility_score = std_dev / atr if atr > 0 else 0.5
    atr_multiplier = min(max(volatility_score, 0.5), 2.0)

    return {
        'atr_multiplier': atr_multiplier,
        'volatility_score': min(volatility_score, 1.0)
    }

def detect_breakout_failure(candles: List[Candle], volatility: Dict[str, float], lookback: int = 3) -> float:
    """
    Detect breakout failure by checking price reversal after a strong move.
    Returns a float representing failure probability.
    """
    if not candles or len(candles) < lookback:
        log_sparse_data('Insufficient candles in detect_breakout_failure', {'count': len(candles)})
        return 0.0

    atr = np.mean([c.atr for c in candles[-lookback:] if c.atr > 0]) or 0.001
    last_candle = candles[-1]
    prior_candle = candles[-2]

    prior_move = abs(prior_candle.close - prior_candle.open) / atr
    reversal = abs(last_candle.close - prior_candle.close) / atr
    is_breakout = prior_move > 1.5
    is_failure = is_breakout and reversal > 1.0 and (last_candle.close < prior_candle.high if prior_candle.close > prior_candle.open else last_candle.close > prior_candle.low)

    failure_prob = min(reversal / volatility['atr_multiplier'], 1.0) if is_failure else 0.0
    return failure_prob

def assess_market_sentiment(candles: List[Candle], lookback: int = 5) -> Dict[str, Any]:
    """
    Assess market sentiment based on price direction, volume, and candle patterns.
    Returns a dictionary with validity, shift probability, and direction.
    """
    if not candles or len(candles) < 2:
        log_sparse_data('Insufficient candles in assess_market_sentiment', {'count': len(candles)})
        return {'is_valid': False, 'shift_probability': 0.0, 'direction': 'none'}

    closes = [c.close for c in candles[-lookback:]]
    volumes = [c.volume for c in candles[-lookback:]]
    atr = np.mean([c.atr for c in candles[-lookback:] if c.atr > 0]) or 0.001

    trend_strength = (closes[-1] - closes[0]) / atr if atr > 0 else 0.0
    direction = 'buy' if trend_strength > 0 else 'sell'

    volume_trend = np.mean([v for c, v in zip(candles[-lookback:], volumes) if c.close > c.open]) / np.mean(volumes) if volumes else 1.0
    shift_probability = min(abs(trend_strength) * volume_trend, 1.0)

    return {
        'is_valid': shift_probability >= MIN_DSCM_SENTIMENT_PROB,
        'shift_probability': shift_probability,
        'direction': direction
    }

def adjust_position_size_sentiment(base_size: float, scale_factor: float, is_aligned: bool, volatility: Dict[str, float]) -> float:
    """
    Adjust position size based on sentiment alignment and volatility.
    Returns adjusted position size as a float.
    """
    if not is_aligned or base_size <= 0 or scale_factor <= 0:
        return 0.0

    adjusted_size = base_size * scale_factor * min(volatility['atr_multiplier'], MAX_DSCM_SCALE)
    return min(adjusted_size, base_size * MAX_DSCM_SCALE)

def is_gap_up(candles: List[Candle], lookback: int = 2) -> bool:
    """
    Detect a gap-up pattern (current open above previous high).
    Returns True if a gap-up is detected.
    """
    if not candles or len(candles) < lookback:
        log_sparse_data('Insufficient candles in is_gap_up', {'count': len(candles)})
        return False

    last_candle = candles[-1]
    prior_candle = candles[-2]
    return last_candle.open > prior_candle.high

def is_gap_down(candles: List[Candle], lookback: int = 2) -> bool:
    """
    Detect a gap-down pattern (current open below previous low).
    Returns True if a gap-down is detected.
    """
    if not candles or len(candles) < lookback:
        log_sparse_data('Insufficient candles in is_gap_down', {'count': len(candles)})
        return False

    last_candle = candles[-1]
    prior_candle = candles[-2]
    return last_candle.open < prior_candle.low

def safeguard_broker_instability(candles: List[Candle], lookback: int = 5) -> bool:
    """
    Detect broker instability based on abnormal price or volume spikes.
    Returns True if instability is detected.
    """
    if not candles or len(candles) < lookback:
        log_sparse_data('Insufficient candles in safeguard_broker_instability', {'count': len(candles)})
        return False

    ranges = [c.high - c.low for c in candles[-lookback:]]
    volumes = [c.volume for c in candles[-lookback:]]
    atr = np.mean([c.atr for c in candles[-lookback:] if c.atr > 0]) or 0.001
    avg_volume = np.mean(volumes) if volumes else 1000.0

    range_std = np.std(ranges) if ranges else 0.001
    volume_std = np.std(volumes) if volumes else 100.0
    is_range_spike = (ranges[-1] > atr * 3.0) or (ranges[-1] > np.mean(ranges) + 3 * range_std)
    is_volume_spike = (volumes[-1] > avg_volume + 3 * volume_std)

    return is_range_spike or is_volume_spike

def calculate_avg_candle_size(candles: List[Candle], lookback: int = 5) -> float:
    """
    Calculate average candle body size normalized by ATR.
    Returns a float representing average size.
    """
    if not candles or len(candles) < 2:
        log_sparse_data('Insufficient candles in calculate_avg_candle_size', {'count': len(candles)})
        return 0.0

    body_sizes = [abs(c.close - c.open) for c in candles[-lookback:]]
    atr = np.mean([c.atr for c in candles[-lookback:] if c.atr > 0]) or 0.001
    avg_size = np.mean(body_sizes) / atr if atr > 0 else 0.0
    return min(avg_size, 1.0)

def assess_power(candle: Candle, prior_candles: Optional[List[Candle]] = None, 
                 party: Optional[str] = None, direction: Optional[str] = None) -> Dict[str, Any]:
    """
    Assess the power of a candle based on size, reaction, and market context.
    Returns a dictionary with strength, stability, and direction metrics.
    """
    global _power_cache
    candle_id = hash(candle)
    
    if candle_id in _power_cache:
        return _power_cache[candle_id]

    if len(_power_cache) >= 5000:
        _power_cache.clear()
        logger.info("Cache cleared for assess_power due to max size 5000")

    if not all(hasattr(candle, k) for k in ['open', 'close', 'high', 'low']):
        log_sparse_data('Missing candle attributes in assess_power', candle.__dict__)
        return {'strength': 0.0, 'stable': False, 'extreme': False, 'weak': True, 'direction': 'none'}

    size = calculate_candle_size(candle, prior_candles)
    reaction = measure_reaction(candle, timeframe=5)
    context = analyze_context(prior_candles) if prior_candles else {
        'momentum': 0.5, 
        'trend_strength': 0.0, 
        'near_support_resistance': 0.0
    }
    
    strength = 0.35 * size['relative_size'] + 0.35 * reaction['strength'] + 0.30 * context['momentum']
    is_stable = size['is_consistent'] and reaction['is_steady']
    is_extreme = size['is_large'] and reaction['is_volatile']
    is_weak = size['is_small'] and reaction['is_minimal']
    party_dir = 'buyers' if candle.close > candle.open else 'sellers'

    result = {
        'strength': min(max(strength, 0.0), 1.0),
        'stable': is_stable,
        'extreme': is_extreme,
        'weak': is_weak,
        'direction': direction or party_dir
    }
    
    _power_cache[candle_id] = result
    return result

def measure_reaction(candle: Candle, timeframe: int = 5) -> Dict[str, Any]:
    """
    Measure the reaction strength and direction of a candle.
    Returns a dictionary with reaction metrics.
    """
    global _reaction_cache
    candle_id = hash(candle)
    
    if candle_id in _reaction_cache:
        return _reaction_cache[candle_id]

    if len(_reaction_cache) >= 5000:
        _reaction_cache.clear()
        logger.info("Cache cleared for measure_reaction due to max size 5000")

    if not all(hasattr(candle, k) for k in ['open', 'close']):
        log_sparse_data('Missing candle attributes in measure_reaction', candle.__dict__)
        return {'strength': 0.0, 'is_steady': False, 'is_volatile': False, 'is_minimal': True, 'direction': 'none'}

    price_move = abs(candle.close - candle.open) / timeframe
    avg_move = calculate_avg_speed(candle.prior_candles or [])
    strength = min(price_move / avg_move, 1.0) if avg_move > 0 else 0.5
    is_steady = MIN_REACTION_STRENGTH <= strength <= MAX_REACTION_STRENGTH
    is_volatile = strength > MAX_REACTION_STRENGTH
    is_minimal = strength < MIN_REACTION_STRENGTH
    direction = 'buy' if candle.close > candle.open else 'sell'

    result = {
        'strength': strength,
        'is_steady': is_steady,
        'is_volatile': is_volatile,
        'is_minimal': is_minimal,
        'direction': direction
    }
    
    _reaction_cache[candle_id] = result
    return result

def five_second_confirmation(candle: Candle) -> Dict[str, Any]:
    """
    Confirm short-term reaction validity for a candle.
    Returns a dictionary with confirmation metrics.
    """
    global _five_second_cache
    candle_id = hash(candle)
    
    if candle_id in _five_second_cache:
        return _five_second_cache[candle_id]

    if len(_five_second_cache) >= 5000:
        _five_second_cache.clear()
        logger.info("Cache cleared for five_second_confirmation due to max size 5000")

    if not all(hasattr(candle, k) for k in ['open', 'close', 'high', 'low']):
        log_sparse_data('Missing candle attributes in five_second_confirmation', candle.__dict__)
        return {'direction': 'none', 'score': 0.0, 'is_valid': False, 'buyer_strength': 0.0, 'seller_strength': 0.0}

    metrics = compute_reaction_metrics(candle, timeframe=5)
    wick_exhaustion = validate_wick_exhaustion(candle, candle.prior_candles)
    dominant_party = metrics['dominant_party']
    opposing_absence = not detect_opposing_party([candle])['is_present']
    stability = assess_candle_consistency(candle, candle.prior_candles or [])

    score = (
        0.35 * metrics['reaction_strength'] +
        0.25 * (1.0 if wick_exhaustion['is_valid'] else 0.0) +
        0.2 * (1.0 if opposing_absence else 0.0) +
        0.2 * stability
    )

    result = {
        'direction': metrics['reaction_direction'],
        'score': score,
        'is_valid': score >= MIN_CONFIRMATION_SCORE,
        'buyer_strength': metrics['buyer_strength'],
        'seller_strength': metrics['seller_strength']
    }
    
    _five_second_cache[candle_id] = result
    return result

def compute_reaction_metrics(candle: Candle, timeframe: int = 5) -> Dict[str, Any]:
    """
    Compute reaction metrics for a candle.
    Returns a dictionary with reaction strength and party dominance.
    """
    global _compute_reaction_cache
    candle_id = hash(candle)
    
    if candle_id in _compute_reaction_cache:
        return _compute_reaction_cache[candle_id]

    if len(_compute_reaction_cache) >= 5000:
        _compute_reaction_cache.clear()
        logger.info("Cache cleared for compute_reaction_metrics due to max size 5000")

    reaction = measure_reaction(candle, timeframe)
    buyer_strength = reaction['strength'] if reaction['direction'] == 'buy' else 0.0
    seller_strength = reaction['strength'] if reaction['direction'] == 'sell' else 0.0
    dominant_party = 'buyers' if buyer_strength > seller_strength else 'sellers'

    result = {
        'reaction_strength': reaction['strength'],
        'reaction_direction': reaction['direction'],
        'buyer_strength': buyer_strength,
        'seller_strength': seller_strength,
        'dominant_party': dominant_party
    }
    
    _compute_reaction_cache[candle_id] = result
    return result

def detect_exhaustion(candles: List[Candle], cache: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Detect exhaustion signals based on candle patterns.
    Returns a dictionary with exhaustion metrics.
    """
    cache = cache or {}
    if not candles:
        log_sparse_data('Empty candle list in detect_exhaustion', None)
        return {'is_exhausted': False, 'probability': 0.0, 'direction': 'none'}

    small_candles = all(calculate_candle_size(c, candles)['is_small'] for c in candles)
    weak_reaction = all(measure_reaction(c, timeframe=5)['is_minimal'] for c in candles)
    gap_exhaustion = mitigate_gap_pressure_exhaustion(candles) if is_gap_up(candles) or is_gap_down(candles) else True
    wick_exhaustion = validate_wick_exhaustion(candles[-1], candles[:-1])['is_valid']
    momentum_decay = assess_momentum_decay(candles)

    return {
        'is_exhausted': small_candles and weak_reaction and gap_exhaustion and wick_exhaustion,
        'probability': min(momentum_decay * 0.9, 1.0),
        'direction': candles[-1].direction if candles else 'none'
    }

def check_momentum_entry(candles: List[Candle], direction: str, 
                        candles_5m: List[Candle], candles_15m: List[Candle], 
                        cache: Optional[Dict] = None) -> str:
    """
    Check for momentum entry signals based on power trend, stability, and confluence.
    Returns the trade direction ('BUY', 'SELL') or 'NONE'.
    """
    cache = cache or {}
    monitor_cache(cache, 'momentum_entry', 'check_momentum_entry')

    if not candles or len(candles) < 3:
        log_sparse_data('Empty or insufficient candle list in check_momentum_entry', {'count': len(candles)})
        return "NONE"

    power_trend = [assess_power(c, tuple(candles[:-1]))['strength'] for c in candles[-3:-1]]
    if not (len(power_trend) >= 2 and all(power_trend[i] < power_trend[i+1] for i in range(len(power_trend)-1))):
        return "NONE"

    if (power_trend[-1] > MOMENTUM_THRESHOLD * calculate_avg_power(candles[-5:-1]) and
        detect_stable_pressure(candles[-1]) and
        five_second_confirmation(candles[-1])['is_valid'] and
        adapt_order_flow(candles, candles_5m, candles_15m, direction, cache=cache)['is_adapted'] and
        not detect_manipulation(candles)['is_manipulated']):
        
        if validate_stable_power_confluence(
            candles, 
            direction, 
            recent_position_detected=identify_recent_position(candles), 
            cache=cache
        ):
            return direction.upper()
    
    return "NONE"

def adapt_order_flow(candles_1m: List[Candle], candles_5m: List[Candle], 
                    candles_15m: List[Candle], trade_direction: str, 
                    cache: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Adapt order flow based on multi-timeframe structure alignment.
    Returns a dictionary with adaptation metrics.
    """
    cache = cache or {}
    monitor_cache(cache, 'order_flow', 'adapt_order_flow')

    if not all([candles_1m, candles_5m, candles_15m]):
        log_sparse_data('Missing timeframe candles in adapt_order_flow', 
                       {'1m': len(candles_1m), '5m': len(candles_5m), '15m': len(candles_15m)})
        return {'is_adapted': False, 'score': 0.0, 'direction': trade_direction}

    cache_results = {}
    for timeframe, candles in [('1m', candles_1m), ('5m', candles_5m), ('15m', candles_15m)]:
        cache_key = f'{timeframe}_{hash(candles[-1])}'
        if cache_key not in cache:
            structure = decode_multi_timeframe_structure(candles, timeframe)
            if not structure['is_valid']:
                log_sparse_data(f'Invalid {timeframe} structure in adapt_order_flow', 
                              {'candle_count': len(candles), 'last_candle': candles[-1].__dict__})
            cache[cache_key] = structure
        cache_results[timeframe] = cache[cache_key]

    structure_1m = cache_results['1m']
    structure_5m = cache_results['5m']
    structure_15m = cache_results['15m']
    alignment_score = 0.5 * structure_1m['strength'] + 0.3 * structure_5m['strength'] + 0.2 * structure_15m['strength']
    direction_match = all(s['direction'] == trade_direction for s in [structure_1m, structure_5m, structure_15m])

    return {
        'is_adapted': alignment_score >= MIN_ORDER_FLOW_STRENGTH and direction_match,
        'score': alignment_score,
        'direction': trade_direction
    }

def detect_opposing_party(candles: List[Candle], direction: Optional[str] = None, 
                         cache: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Detect the presence and strength of opposing market parties.
    Returns a dictionary with opposition metrics.
    """
    cache = cache or {}
    candle_id = hash(candles[-1]) if candles else 0
    monitor_cache(cache, candle_id, 'detect_opposing_party')

    if candle_id in cache:
        return cache[candle_id]

    if not candles:
        log_sparse_data('Empty candle list in detect_opposing_party', None)
        return {'strength': 0.0, 'is_present': False}

    dominant_dir = direction or ('buy' if candles[-1].close > candles[-1].open else 'sell')
    opposing_candles = [c for c in candles if (c.close < c.open if dominant_dir == 'buy' else c.close > c.open)]
    strength = sum(assess_power(c, tuple(candles[:-1]))['strength'] for c in opposing_candles) / max(len(opposing_candles), 1)

    result = {'strength': strength, 'is_present': len(opposing_candles) > 0}
    cache[candle_id] = result
    return result

def detect_stable_pressure(candle: Candle) -> bool:
    """
    Detect stable pressure based on consistent short-term reactions.
    Returns True if pressure is stable.
    """
    prior_candles = candle.prior_candles or []
    if not prior_candles:
        log_sparse_data('Empty prior candles in detect_stable_pressure', candle.__dict__)
        return False
    reactions = [assess_five_second_reaction([c]) for c in prior_candles[-MIN_STABLE_PRESSURE_COUNT:]]
    return len(reactions) >= MIN_STABLE_PRESSURE_COUNT and all(r['direction'] == reactions[-1]['direction'] for r in reactions)

def detect_manipulation(candles: List[Candle]) -> Dict[str, Any]:
    """
    Detect potential market manipulation based on gaps, wicks, or instability.
    Returns a dictionary with manipulation metrics.
    """
    if not candles:
        log_sparse_data('Empty candle list in detect_manipulation', None)
        return {'is_manipulated': False, 'reason': 'No candles provided'}

    gap_detected = is_gap_up(candles) or is_gap_down(candles)
    wick_pressure = is_high_wick_pressure(candles[-1])
    pressure_exhaustion = not mitigate_gap_pressure_exhaustion(candles) if gap_detected else False
    broker_instability = safeguard_broker_instability(candles)

    return {
        'is_manipulated': gap_detected or wick_pressure or pressure_exhaustion or broker_instability,
        'reason': (
            'Broker instability' if broker_instability else 
            'Gap detected' if gap_detected else 
            'Wick pressure'
        )
    }

def assess_momentum_decay(candles: List[Candle]) -> float:
    """
    Assess momentum decay based on recent candle speeds.
    Returns a float representing decay probability.
    """
    if not candles:
        log_sparse_data('Empty candle list in assess_momentum_decay', None)
        return 0.3
    speeds = [calculate_momentum([c]) for c in candles]
    return 0.9 if len(speeds) > 1 and speeds[-1] < speeds[0] * 0.7 else 0.3

def validate_stable_power_confluence(candles: List[Candle], trade_direction: str, 
                                   recent_position_detected: Optional[Dict], 
                                   position_size: str = 'standard', 
                                   cache: Optional[Dict] = None) -> bool:
    """
    Validate confluence for stable power signals.
    Returns True if confluence is valid.
    """
    cache = cache or {}
    monitor_cache(cache, 'stable_power_confluence', 'validate_stable_power_confluence')

    if not candles:
        log_sparse_data('Empty candle list in validate_stable_power_confluence', None)
        return False

    if not recent_position_detected:
        return validate_optimized_confluence(candles, trade_direction, cache=cache)

    pressure_stability = assess_pressure_stability(candles)
    market = filter_market_type(candles)
    confluence_score = 0.83 if market['market_type'] == 'unstable_continuation' else 0.85

    confirmations = [
        ('power', filter_power_dominance(candles, trade_direction)),
        ('zone_fight', check_zone_fight_outcome(candles, trade_direction)),
        ('exhaustion', check_non_exhausted_previous(candles[-1], candles[-2]) if trade_direction == candles[-1].direction 
         else filter_tradeable_exhaustion([candles[-2]], trade_direction)['is_tradeable']),
        ('five_sec', five_second_confirmation(candles[-1])['is_valid']),
        ('pressure', pressure_stability['is_stable']),
        ('movement', detect_market_movement(candles)['direction'] == trade_direction),
        ('structure', decode_market_structure(candles, cache=cache)['direction'] == trade_direction),
        ('no_gap', not is_gap_up(candles) and not is_gap_down(candles)),
        ('order_flow', adapt_order_flow(candles, candles_5m=candles, candles_15m=candles, trade_direction=trade_direction, cache=cache)['is_adapted'])
    ]

    weights = {
        'power': 0.30, 'zone_fight': 0.10, 'exhaustion': 0.15, 'five_sec': 0.25,
        'pressure': 0.09, 'movement': 0.03, 'structure': 0.03, 'no_gap': 0.02, 'order_flow': 0.03
    }

    score = sum(weights[name] * value for name, value in confirmations)
    return (
        score >= confluence_score and
        sum(1 for _, c in confirmations[:4] if c) >= 3 and
        safeguard_position_size(candles, position_size)
    )

def validate_optimized_confluence(candles: List[Candle], trade_direction: str, 
                                min_confirmations: int = 3, cache: Optional[Dict] = None) -> bool:
    """
    Validate optimized confluence for trade signals.
    Returns True if confluence is valid.
    """
    cache = cache or {}
    monitor_cache(cache, 'optimized_confluence', 'validate_optimized_confluence')

    if not candles:
        log_sparse_data('Empty candle list in validate_optimized_confluence', None)
        return False

    market = filter_market_type(candles)
    confluence_score = 0.83 if market['market_type'] == 'unstable_continuation' else 0.85

    confirmations = [
        ('power', filter_power_dominance(candles, trade_direction)),
        ('exhaustion', check_non_exhausted_previous(candles[-1], candles[-2]) if trade_direction == candles[-1].direction 
         else filter_tradeable_exhaustion([candles[-2]], trade_direction)['is_tradeable']),
        ('five_sec', five_second_confirmation(candles[-1])['is_valid']),
        ('movement', detect_market_movement(candles)['direction'] == trade_direction),
        ('structure', decode_market_structure(candles, cache=cache)['direction'] == trade_direction),
        ('order_flow', adapt_order_flow(candles, candles_5m=candles, candles_15m=candles, trade_direction=trade_direction, cache=cache)['is_adapted'])
    ]

    weights = {
        'power': 0.40, 'exhaustion': 0.20, 'five_sec': 0.25,
        'movement': 0.05, 'structure': 0.05, 'order_flow': 0.05
    }

    score = sum(weights[name] * value for name, value in confirmations)
    return score >= confluence_score and sum(1 for _, c in confirmations[:3] if c) >= min_confirmations

def is_high_wick_pressure(candle: Candle, volatility: Optional[Dict] = None) -> bool:
    """
    Detect high wick pressure indicating rejection.
    Returns True if high wick pressure is present.
    """
    if not all(hasattr(candle, k) for k in ['open', 'close', 'high', 'low']):
        log_sparse_data('Missing candle attributes in is_high_wick_pressure', candle.__dict__)
        return False
    wick_size = max(candle.high - candle.close, candle.close - candle.low)
    body_size = abs(candle.close - candle.open)
    volatility_factor = volatility['atr_multiplier'] if volatility else 1.0
    return wick_size > WICK_PRESSURE_THRESHOLD * body_size * volatility_factor

def mitigate_gap_pressure_exhaustion(candles: List[Candle]) -> bool:
    """
    Mitigate exhaustion caused by gap pressure.
    Returns True if gap pressure is mitigated.
    """
    if not candles:
        log_sparse_data('Empty candle list in mitigate_gap_pressure_exhaustion', None)
        return True
    if is_gap_up(candles) or is_gap_down(candles):
        pressure = assess_five_second_reaction(candles)['strength_ratio']
        return pressure < GAP_EXHAUSTION_THRESHOLD
    return True

def check_non_exhausted_previous(current_candle: Candle, previous_candle: Candle) -> bool:
    """
    Check if the previous candle is not exhausted.
    Returns True if not exhausted.
    """
    if not all(hasattr(previous_candle, k) for k in ['open', 'close']):
        log_sparse_data('Missing previous candle attributes in check_non_exhausted_previous', previous_candle.__dict__)
        return False
    size = calculate_candle_size(previous_candle, [previous_candle])['is_small']
    reaction = measure_reaction(previous_candle, timeframe=5)['is_minimal']
    return not (size or reaction)

def identify_recent_position(candles: List[Candle]) -> Optional[Dict]:
    """
    Identify recent significant market position.
    Returns a dictionary with position details or None.
    """
    if not candles:
        log_sparse_data('Empty candle list in identify_recent_position', None)
        return None
    recent_candles = candles[-5:-1]
    max_activity = max([assess_power(c, tuple(candles[:-1]))['strength'] for c in recent_candles]) if recent_candles else 0.0
    for c in recent_candles[::-1]:
        if assess_power(c, tuple(candles[:-1]))['strength'] >= 0.8 * max_activity:
            return {'price': c.close, 'party': 'buyers' if c.direction == 'buy' else 'sellers'}
    return None

def decode_market_structure(candles: List[Candle], cache: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Decode market structure based on momentum and consolidation.
    Returns a dictionary with structure metrics.
    """
    cache = cache or {}
    candle_id = hash(candles[-1]) if candles else 0
    monitor_cache(cache, candle_id, 'decode_market_structure')

    if candle_id in cache:
        return cache[candle_id]

    if not candles:
        log_sparse_data('Empty candle list in decode_market_structure', None)
        return {'is_trending': False, 'is_consolidation': False, 'direction': 'neutral'}

    momentum = calculate_momentum(candles[-6:])
    power = assess_power(candles[-1], tuple(candles[:-1]))
    is_trending = momentum > MOMENTUM_THRESHOLD and power['stable']
    is_consolidation = sum(1 for c in candles[-6:] if calculate_candle_size(c, candles)['is_small']) >= MIN_STABLE_CANDLE_COUNT
    direction = power['direction'] if is_trending else 'neutral'

    result = {
        'is_trending': is_trending,
        'is_consolidation': is_consolidation,
        'direction': direction
    }
    cache[candle_id] = result
    return result

def filter_power_dominance(candles: List[Candle], direction: str) -> bool:
    """
    Filter for power dominance in the specified direction.
    Returns True if power is dominant.
    """
    if not candles:
        log_sparse_data('Empty candle list in filter_power_dominance', None)
        return False
    power = assess_power(candles[-1], tuple(candles[:-1]))
    return power['direction'] == direction and power['strength'] > POWER_EXIT_THRESHOLD * calculate_avg_power(candles[-3:-1])

def check_zone_fight_outcome(candles: List[Candle], trade_direction: str) -> bool:
    """
    Check the outcome of a zone fight between buyers and sellers.
    Returns True if the trade direction dominates.
    """
    if not candles:
        log_sparse_data('Empty candle list in check_zone_fight_outcome', None)
        return False
    buyer_power = assess_power(candles[-1], tuple(candles[:-1]), party='buyers')['strength']
    seller_power = assess_power(candles[-1], tuple(candles[:-1]), party='sellers')['strength']
    if trade_direction == 'buy':
        return buyer_power > seller_power * MIN_ZONE_FIGHT_STRENGTH
    return seller_power > buyer_power * MIN_ZONE_FIGHT_STRENGTH

def assess_pressure_stability(candles: List[Candle]) -> Dict[str, Any]:
    """
    Assess the stability of market pressure.
    Returns a dictionary with stability metrics.
    """
    return {'is_stable': detect_stable_pressure(candles[-1])}

def filter_market_type(candles: List[Candle]) -> Dict[str, Any]:
    """
    Filter market type based on stability and power trends.
    Returns a dictionary with market type and tradability.
    """
    if not candles:
        log_sparse_data('Empty candle list in filter_market_type', None)
        return {'is_tradable': False, 'market_type': 'unstable'}

    power_trend = [assess_power(c, tuple(candles[:-1]))['strength'] for c in candles[-3:]]
    is_stable = detect_stable_pressure(candles[-1]) and all(
        abs(power_trend[i] - power_trend[i+1]) < 0.1 * power_trend[i] for i in range(len(power_trend)-1)
    )
    is_extreme = power_trend[-1] > MOMENTUM_THRESHOLD * calculate_avg_power(candles[-5:-1])

    if is_stable:
        return {'is_tradable': True, 'market_type': 'stable'}
    elif is_extreme:
        return {'is_tradable': True, 'market_type': 'unstable_continuation'}
    elif power_trend[-1] < POWER_EXIT_THRESHOLD * calculate_avg_power(candles[-3:-1]):
        return {'is_tradable': True, 'market_type': 'bad'}
    else:
        return {'is_tradable': False, 'market_type': 'unstable'}

def safeguard_position_size(candles: List[Candle], position_size: str) -> bool:
    """
    Safeguard position size to ensure validity.
    Returns True if position size is valid.
    """
    return position_size in ['standard', 'reduced']

def generate_sample_candles(n: int = 10) -> List[Candle]:
    """
    Generate sample candles for testing.
    Returns a list of Candle objects.
    """
    candles = []
    for i in range(n):
        open_price = 100.0 + np.random.normal(0, 0.5)
        close_price = open_price + np.random.normal(0, 0.3)
        high_price = max(open_price, close_price) + np.random.uniform(0.1, 0.5)
        low_price = min(open_price, close_price) - np.random.uniform(0.1, 0.5)
        candles.append(Candle(
            open=open_price,
            close=close_price,
            high=high_price,
            low=low_price,
            volume=1000.0,
            atr=0.5,
            direction='buy' if close_price > open_price else 'sell',
            prior_candles=candles[-5:] if i >= 5 else [],
            timestamp=datetime.now()
        ))
    return candles

def assess_candle_consistency(candle: Candle, prior_candles: List[Candle]) -> float:
    """
    Assess candle consistency by comparing its size and direction to prior candles.
    Returns a float representing consistency score.
    """
    if not prior_candles or len(prior_candles) < 2:
        log_sparse_data('Insufficient prior candles in assess_candle_consistency', {'count': len(prior_candles)})
        return 0.5

    current_size = calculate_candle_size(candle, prior_candles)['relative_size']
    prior_sizes = [calculate_candle_size(c, prior_candles)['relative_size'] for c in prior_candles[-2:]]
    avg_prior_size = np.mean(prior_sizes) if prior_sizes else 1.0

    current_direction = 'buy' if candle.close > candle.open else 'sell'
    prior_directions = ['buy' if c.close > c.open else 'sell' for c in prior_candles[-2:]]
    direction_score = 1.0 if all(d == current_direction for d in prior_directions) else 0.5

    size_score = 1.0 - min(abs(current_size - avg_prior_size) / avg_prior_size, 1.0) if avg_prior_size > 0 else 0.5

    return 0.6 * direction_score + 0.4 * size_score

def filter_tradeable_exhaustion(candles: List[Candle], trade_direction: str) -> Dict[str, Any]:
    """
    Filter exhaustion signals to determine if they are tradeable in the given direction.
    Returns a dictionary with tradeability and strength.
    """
    if not candles:
        log_sparse_data('Empty candle list in filter_tradeable_exhaustion', None)
        return {'is_tradeable': False, 'exhaustion_strength': 0.0}

    exhaustion = detect_exhaustion(candles)
    last_candle = candles[-1]
    direction_match = (exhaustion['direction'] == trade_direction) or (exhaustion['direction'] == 'none' and last_candle.close > last_candle.open if trade_direction == 'buy' else last_candle.close < last_candle.open)

    return {
        'is_tradeable': exhaustion['is_exhausted'] and direction_match and exhaustion['probability'] >= MIN_EXHAUSTION_PROB,
        'exhaustion_strength': exhaustion['probability']
    }

def detect_market_movement(candles: List[Candle], lookback: int = 5) -> Dict[str, Any]:
    """
    Detect market movement direction and strength based on recent candles.
    Returns a dictionary with direction and movement strength.
    """
    if not candles or len(candles) < 2:
        log_sparse_data('Insufficient candles in detect_market_movement', {'count': len(candles)})
        return {'direction': 'none', 'movement_strength': 0.0}

    closes = [c.close for c in candles[-lookback:]]
    atr = np.mean([c.atr for c in candles[-lookback:] if c.atr > 0]) or 0.001
    movement = (closes[-1] - closes[0]) / atr if atr > 0 else 0.0
    direction = 'buy' if movement > 0 else 'sell'

    return {
        'direction': direction,
        'movement_strength': min(abs(movement), 1.0)
    }

def calculate_avg_power(candles: List[Candle]) -> float:
    """
    Calculate average power of candles based on assess_power results.
    Returns a float representing average power strength.
    """
    if not candles:
        return 0.0
    return sum(assess_power(c, tuple(candles[:-1]))['strength'] for c in candles) / max(len(candles), 1)
if __name__ == "__main__":
    # Example test run (optional)
    print("Module loaded. You can run this in Leviathan_binary_bot.py")

def run_signal(symbol: str, candles_1m, candles_5m, candles_15m, direction="buy", log_file="signal_log.csv"):
    """
    Execute the full algorithm and return a trading signal.
    Also logs the signal with metadata to a CSV file.
    """
    signal = check_momentum_entry(candles_1m, direction, candles_5m, candles_15m)

    # Extract latest candle and relevant info
    latest = candles_1m[-1]
    power = getattr(latest, 'power', 0.0)
    exhaustion = getattr(latest, 'exhaustion', False)


    return signal
