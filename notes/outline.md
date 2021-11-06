# Some Details on Building Equity Index Option Analytics

## The Problem

Build a risk graph and compute greeks for a portfolio of index options.

- Not covering options on ETF, futures or single stocks.
- Focus on what can be extracted from a single snapshot of an option chain
  - All expirations, strikes, calls and puts have bid/ask sampled at the same time.
  - Not assuming spot index or futures quotes available.
  - Not assuming interest rates or dividend yields are available.

## Procedure Outline:

1. Start with a table of option prices:  
  ```Expiration, Strike, PutOrCall, Bid, Ask```
2. For each expiration, assign a `dte` and set `YearFrac` $\tau = \text{dte}/365.$
3. For each expiration, assign an interest rate, $r$, and discount factor:
  $D = \exp(-r\tau)$.
4. For each expiration, assign a forward index price, $F$.  
  Assign a spot price, $S$, for all expirations, and 
  a dividend yield, $\delta$, for each expiration and use
  $F = S \exp( (r-\delta)\tau)$.
5. For each call and put option, compute a mark between the bid and ask prices.
6. Compute an implied volatility for each call and put option.
7. Compute greeks for each call and put option.  Aggregate to get portfolio greeks.
8. Build a "sticky strike" risk graph with ```T+x``` lines by defining a two-dimensional 
grid of "scenario" year fractions and spot prices $\tau_i \times X_j$.
Evaluate each option price (and greeks if desired) at all grid points. 
Aggregate scenario values by portfolio weights to build porfolio value as 
function of spot and year-fraction.  Subtract initial value or cost basis
to make an PnL graph.


