class TrendVolumeComposite(Factor):
    """Composite from pipeline/strategy.py, as a Qube factor (Python mode, US market).

    0.5 * zscore(trend) + 0.5 * zscore(volume ratio), cross-sectional per date:
      trend   mean of the 20- and 60-day price return, measured to the PRIOR close
      volume  mean(volume, 5) / mean(volume, 20) over sessions BEFORE the current one
    Both legs use only completed bars (shift 1), same rule as the strategy, so the value on
    date t is knowable before t's fills.

    Not included: the earnings catalyst gate. It needs event dates, which a factor can't
    read -- it stays in the strategy layer.
    """

    def calculate(self, factors):
        close = factors["close"].groupby(level="symbol").shift(1)
        volume = factors["volume"].groupby(level="symbol").shift(1)

        c20 = close.groupby(level="symbol").shift(20)
        c60 = close.groupby(level="symbol").shift(60)
        trend = 0.5 * (close / c20 - 1) + 0.5 * (close / c60 - 1)

        v5 = volume.groupby(level="symbol").transform(lambda s: s.rolling(5).mean())
        v20 = volume.groupby(level="symbol").transform(lambda s: s.rolling(20).mean())
        vratio = v5 / v20

        def zscore(x):
            return x.groupby(level="date").transform(lambda s: (s - s.mean()) / s.std())

        return 0.5 * zscore(trend) + 0.5 * zscore(vratio)
