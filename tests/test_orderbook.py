import pytest

from quantsim import OrderBook


def test_quotes_spread_midpoint():
    book = OrderBook()
    book.limit("buy", 99, 10)
    book.limit("buy", 98, 5)
    book.limit("sell", 101, 7)
    book.limit("sell", 102, 3)
    assert book.best_bid() == 99
    assert book.best_ask() == 101
    assert book.spread() == 2
    assert book.midpoint() == 100
    assert book.open_orders == 4


def test_empty_book_quotes_are_none():
    book = OrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.spread() is None


def test_depth_aggregates_levels_best_first():
    book = OrderBook()
    book.limit("buy", 99, 10)
    book.limit("buy", 99, 15)
    book.limit("buy", 98, 20)
    book.limit("sell", 101, 5)
    depth = book.depth(2)
    assert [(l.price, l.qty, l.orders) for l in depth["bids"]] == [(99, 25, 2), (98, 20, 1)]
    assert [(l.price, l.qty) for l in depth["asks"]] == [(101, 5)]


def test_executes_at_maker_price():
    book = OrderBook()
    book.limit("sell", 101, 10, id="maker")
    trades = book.limit("buy", 105, 10, id="taker")
    assert len(trades) == 1
    assert (trades[0].price, trades[0].qty) == (101, 10)
    assert (trades[0].maker_id, trades[0].taker_id) == ("maker", "taker")
    assert book.open_orders == 0


def test_price_priority_then_time_priority():
    book = OrderBook()
    book.limit("sell", 102, 5, id="worse")
    book.limit("sell", 101, 5, id="better")
    trades = book.limit("buy", 102, 10)
    assert [t.maker_id for t in trades] == ["better", "worse"]

    book2 = OrderBook()
    book2.limit("sell", 101, 5, id="first")
    book2.limit("sell", 101, 5, id="second")
    trades2 = book2.limit("buy", 101, 7)
    assert [(t.maker_id, t.qty) for t in trades2] == [("first", 5), ("second", 2)]


def test_partial_fill_rests_remainder():
    book = OrderBook()
    book.limit("sell", 101, 4)
    trades = book.limit("buy", 101, 10)
    assert trades[0].qty == 4
    assert book.best_bid() == 101
    assert book.depth(1)["bids"][0].qty == 6


def test_non_crossing_orders_rest():
    book = OrderBook()
    book.limit("sell", 101, 10)
    assert book.limit("buy", 100, 10) == []
    assert (book.best_bid(), book.best_ask()) == (100, 101)


def test_market_order_is_ioc():
    book = OrderBook()
    book.limit("sell", 101, 5)
    book.limit("sell", 103, 5)
    trades = book.market("buy", 50)
    assert sum(t.qty for t in trades) == 10
    assert [t.price for t in trades] == [101, 103]
    assert book.best_bid() is None  # remainder discarded, not rested
    assert book.open_orders == 0


def test_cancel_and_level_cleanup():
    book = OrderBook()
    book.limit("buy", 99, 10, id="a")
    book.limit("buy", 99, 5, id="b")
    assert book.cancel("a") is True
    assert book.depth(1)["bids"][0].qty == 5
    assert book.cancel("b") is True
    assert book.best_bid() is None
    assert book.cancel("ghost") is False
    # filled orders can't be cancelled
    book.limit("sell", 101, 5, id="maker")
    book.market("buy", 5)
    assert book.cancel("maker") is False


def test_validation():
    book = OrderBook()
    with pytest.raises(ValueError):
        book.limit("buy", 0, 1)
    with pytest.raises(ValueError):
        book.limit("buy", 100, -1)
    with pytest.raises(ValueError):
        book.market("hold", 1)
    book.limit("buy", 99, 1, id="dup")
    with pytest.raises(ValueError):
        book.limit("buy", 98, 1, id="dup")


def _lcg(state=42):
    while True:
        state = (state * 1664525 + 1013904223) % 2**32
        yield state / 2**32


def test_book_never_crosses_itself_under_random_storm():
    book = OrderBook()
    rand = _lcg()
    for _ in range(5_000):
        side = "buy" if next(rand) < 0.5 else "sell"
        price = 90 + int(next(rand) * 21)
        book.limit(side, price, 1 + int(next(rand) * 10))
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None and ask is not None:
            assert bid < ask


def test_quantity_conservation():
    book = OrderBook()
    rand = _lcg(7)
    submitted = filled = 0
    for _ in range(2_000):
        qty = 1 + int(next(rand) * 5)
        submitted += qty
        trades = book.limit("buy" if next(rand) < 0.5 else "sell",
                            95 + int(next(rand) * 11), qty)
        filled += 2 * sum(t.qty for t in trades)  # each trade consumes maker+taker
    depth = book.depth(1_000)
    resting = sum(l.qty for l in depth["bids"]) + sum(l.qty for l in depth["asks"])
    assert filled + resting == submitted
