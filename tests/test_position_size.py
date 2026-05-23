from tape.risk import calculate_position_size


def test_one_percent_rule_with_a_clean_round_number():
    # $100k account, 1% risk, $1/share stop distance → $1000 dollar-risk →
    # 1000 shares.
    assert calculate_position_size(100_000, 100.0, 99.0, risk_pct=0.01) == 1000


def test_tighter_stop_means_larger_share_count_same_dollar_risk():
    # Same account + risk %, but a 50¢ stop instead of $1 should double the
    # share count because each share now risks half as much.
    assert calculate_position_size(100_000, 100.0, 99.5, risk_pct=0.01) == 2000


def test_floors_to_a_whole_share():
    # $10k account, 1% risk → $100 dollar risk. $33/share stop distance →
    # 3.03 shares → floor to 3.
    assert calculate_position_size(10_000, 100.0, 67.0, risk_pct=0.01) == 3


def test_returns_zero_when_stop_equals_entry():
    # No defined risk per share — refuse the trade rather than divide by zero.
    assert calculate_position_size(100_000, 100.0, 100.0) == 0


def test_returns_zero_for_non_positive_inputs():
    assert calculate_position_size(0, 100.0, 99.0) == 0
    assert calculate_position_size(100_000, 0.0, -1.0) == 0
    assert calculate_position_size(100_000, 100.0, 99.0, risk_pct=0.0) == 0


def test_returns_zero_when_one_share_already_exceeds_dollar_risk():
    # $1k account, 1% risk → $10 dollar risk. $50 stop distance per share.
    # Even one share would risk $50 > $10, so we skip the trade.
    assert calculate_position_size(1_000, 200.0, 150.0, risk_pct=0.01) == 0


def test_stop_above_entry_is_a_short_with_the_same_math():
    # Risk is the absolute distance, so a short trade with stop above entry
    # sizes the same way as a long with stop below.
    assert calculate_position_size(100_000, 100.0, 101.0, risk_pct=0.01) == 1000
