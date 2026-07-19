"""Order book throughput benchmark: a random storm of limit orders (which both
rest and match) plus periodic cancels — a realistic mixed workload.

Run: python scripts/bench_orderbook.py
"""
import time

from quantsim import OrderBook

N = 200_000


def lcg(state=12345):
    while True:
        state = (state * 1664525 + 1013904223) % 2**32
        yield state / 2**32


def main() -> None:
    book = OrderBook()
    rand = lcg()
    ids = []

    start = time.perf_counter()
    for i in range(N):
        roll = next(rand)
        if roll < 0.05 and ids:
            book.cancel(ids[int(next(rand) * len(ids))])
        else:
            side = "buy" if roll < 0.525 else "sell"
            price = 9_000 + int(next(rand) * 201)  # 201 price levels
            order_id = f"b{i}"
            book.limit(side, price, 1 + int(next(rand) * 100), id=order_id)
            if len(ids) < 10_000:
                ids.append(order_id)
    elapsed = time.perf_counter() - start

    print(f"orders processed : {N:,}")
    print(f"elapsed          : {elapsed * 1000:.0f} ms")
    print(f"throughput       : {N / elapsed:,.0f} ops/sec")
    print(f"resting orders   : {book.open_orders:,}")
    print(f"best bid/ask     : {book.best_bid()} / {book.best_ask()}")


if __name__ == "__main__":
    main()
