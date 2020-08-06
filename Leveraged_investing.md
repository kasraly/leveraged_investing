# Leveraged Investing

## Margin account

Let's consider the simple approach of leveraged investing with a margin account.
Let's assume we have $50 and we buy $100 of the US stock market with the other $50 borrowed from the margin.
We would have $100 of assets and $50 debt, with a 2x leverage.
If the us stock market goes up by 10%, our $100 asset will grow to $110 but the $50 debt will not change, resulting in a total worth of $60.
This yields a $10 gain over our initial $50, with is 20% gain, hence the 2x leverage.
Similarly, if the stock market drops by 10%, our total worth will drop to $40 resulting in 20% drop.

Historically, the stock market have had about 7% return.
As long as the cost of borrowing on the margin is considerably lower than the expected stock market return, in my opinion using leverage to boost the returns is worth considering.

### Margin call

So far everything seemed positive, so let's talk about the risks and trade-offs.
The main risk is a related to the possibility of margin call.
The margin call happens when the margin in the account goes negative.
In other words, it happens when the debt to asset ratio exceeds beyond certain threshold.
The stock market is a very liquid asset with real-time pricing.
Buying stock on leverage, is different than buying a house on leverage.
If you buy a house with 80% mortgage, and the house prices drop by more than 20%, the bank will not foreclose the house as long as the mortgage is payment every month.
Even though the client's net worth is negative, you can still keep the house by paying the monthly payment.
However, when the stocks are bought on leverage, the debt to asset ratio should always remain below certain threshold.
Since the debt is fixed and stock prices change in real-time this can happen at any moment.

### Max leverage ratio

We are going to limit the discussion to highly diversified index funds.
Historically, the maximum drawdown (drop from the high) for the US stock market has been around 50% (happened in the 2009 crash).
The margin ratio (maximum debt to asset ratio) for liquid ETFs (ETFs with more than 50k daily volume) is 70%.
When taking these two ratios into account, we can find the maximum leverage that would almost never cause a margin call.
We can easily calculate the maximum leverage ratio in reverse.
Assume our assets are at margin call limit of 70%.
Let's consider $100 worth of ETF with $70 debt.
This $100 is after 50% drop in price, so the original value of the ETF has been $200.
Meaning we had $130 dollar and borrowed another $70 to buy $200 worth of ETF.
This gives us a leverage ratio of 200/130 = 1.538x.
So if we limit the leverage ratio to 1.538x, and buy highly diversified ETF (VUN, VEQT, XIC, etc.), the chance of a margin call happening would be almost zero.

## Leveraged ETF

In registered accounts, we do not have access to margin and borrowed money, so we need to use leveraged ETFs.
However, leveraged ETFs are different than borrowing on the margin, mainly due to their daily reset of the leverage.
This daily reset along with volatility in the price result in a decay over time.
The higher the volatility, the higher the decay.
The numbers can be found in the ProShares ETFs prospectus.
With the average volatility of 13% for the US stock market, the yearly decay for a 3x leveraged ETF would be around 5-6%.

Although the leveraged ETFs have daily leverage reset, we can counteract this daily reset by buying them when they drop and selling the excess when they go up.
I ETF related to S&P500 as example.
The cheapest index ETF that follows S&P500 is VOO, and the 3x Leveraged ETF of S&P500 is UPRO.
Let's assume we buy $100 UPRO, which is almost equivalent to buying $300 worth of VOO with $200 borrowed money.
If the VOO drops by 10%, our $300 would drop to $270 while the $200 debt remains.
In the case of margin account, we would maintain the $200 debt, and keep the $270 VOO, effectively increasing the leverage to 270/70 = 3.857.
However, the UPRO resets the leverage daily to 3x.
After 10% drop in VOO, the UPRO price has dropped to $70.
With 3x leverage, its total debt would be $140, so it effectively sells the assets internally to reduce the debt to $140.

After the UPRO is dropped to $70, if we buy another $20 of UPRO, our total UPRO value would be $90 and our equivalent VOO exposure would $270.
We effectively have compensated for the daily reset of the UPRO and simulated the behaviour of a margin account with leveraged ETF.
Note that if the VOO gains 10%, we have to do the same.
In this case UPRO would increase from $100 to $130.
But this will result in $390 equivalent VOO, and we would need to sell $20 of the UPRO to reduce it to $110, and bring it down to equivalent $330 of VOO.

Now the question is where did the $20 that we used to buy the UPRO came from?
Since this is a registered account, it is not possible to add cash to it at will.
So we effectively need to keep some cash in the account to buy UPRO when needed.
Let's assume that we have $200 and use $100 to buy UPRO and leave the other $100 as reserve.
We would buy UPRO from the reserve when it drops and sell UPRO and add to the reserve when UPRO gains.
Of course the reserve does not have to be cash, and we can use long-term treasury ETF instead.
It has two benefits.
Firsts, the treasury ETF gradually grows instead of sitting idle, and compensate for the higher MER of UPRO.
Second, the value of the long-term treasury ETFs usually move opposite of the stock market in the sort term.
This provides some cushion for volatility of the stock market.
A good choice for the long-term treasury ETF is EDV.
Another option is TMF (which is a 3x leveraged treasury ETF).
TMF can provide more cushion, but has more risk due to its leveraged nature, and is not ideal as a reserve.

### Rebalancing 1.0

The simple approach on how to maintaining the reserve is to use a fixed asset allocation, for example 50% UPRO and 50% EDV.
Say we have $100, and buy $50 of UPRO and $50 of EDV.
This gives $150 of exposure to VOO, or 1.5x leverage with $50 reserve.
If VOO gain 10%, the %50 UPRO increase to $65.
We sell $7.5 of the UPRO and buy EDV with that resulting in $57.5 UPRO and $57.5 EDV.
This results in VOO equivalent exposure of $172.5, which is slightly higher than the $165 (1.5x * 110) that we had in mind.
Similarly, if the VOO drop 10%, the UPRO value would be $35.
After rebalancing we would have $42.5 of UPRO and $42.5 of EDV.
Here again our exposure to VOO would be $127.5 instead of $135.

We assumed that the EDV value would not change when VOO price changes.
But in fact EDV have a negative correlation or around -0.5 with stock market and would move opposite the VOO in short term.
Therefore, the rebalanced values would be closer to desired VOO exposure than what we calculated above.

Personally, I would also throw in some VOO into the mix to be safe in case the leveraged ETF was collapsed for any reason.
We can use asset allocation of 40% UPRO, 30% VOO, 30% EDV.
This will result in the same 1.5x leverage, while reducing the 3x leverage portions.
However, since the EDV ratio is reduced from 50% to 30%, the cushion from the treasury's negative correlation would reduce and short term volatility would increase.
Over the long term, the new mix would have very close performance to the 50% UPRO + 50% EDV.

### Rebalancing 2.0

In the previous rebalancing approach, we used simple asset allocation which can result in the equivalent exposure to VOO increase or decrease.
While the overall impact is not that significant, if the VOO drops and we lose the exposure to it, when it eventually recovers, we cannot regain the losses and end up with some decay.
This decay would be less than 1% over a year and much lower than the decay of a 3x leverage with daily reset, but still not desirable.
In this approach we try to keep the leverage ratio fixed and eliminate the decay.

As discussed in the margin account section, we assume the maximum drawdown of the VOO to be 50%.
Initially, we also use cash as reserve and do assume correlation between reserve and stock price.
Also, we assume the rebalancing approach is going to maintain a fixed leverage ratio.

#### rebalancing with cash reserve

With these assumptions in mind, we try to find the maximum leverage that we can have.
If our leverage ratio is `x`, when VOO drops by 50%, our total asset (stock + reserve) would drop by `x * 50%`.
If we solve this, we can see that the limit for `x` is 1.5x, meaning at max we can have 50% UPRO and 50% reserve.
Let's assume we buy $50 of UPRO and keep $50 cash with our initial $100 investment.
If the VOO drop by 50%, the UPRO will drop by 150% (note we are maintaining the leverage with rebalancing), and the value of UPRO would be -$25.
We buy another $50 of UPRO with the reserve and will have $25 of UPRO and no reserve.
This is equivalent to $75 of VOO which is the 1.5x leverage when compared to the case that we had bought $100 of VOO in the first place and it had dropped 50%.
Note that in the margin account, at the limit we could go as high as 70% debt to asset ratio.
However, here we are limited to the 3x leveraged ETF.
Therefore, the maximum leverage that can sustain 50% drop is slightly lower.

Now onto the rebalancing algorithm.
Let's define `d` as the drawdown of the VOO, defined by price of VOO divided by its 52-week high price.
If `d` is zero, the asset ratio of cash and UPRO should be 50%-50%, so that we have enough reserve for a 50% drawdown.
If VOO drops by `d` percentage, the UPRO drops by `3*d`.
So we need to buy `2*d` of UPRO to bring back the exposure relative to VOO.
Assuming that before rebalancing the UPRO is `1-3d` and the cash reserve is `1`.
After the rebalance, the UPRO would be `1-d` and the cash reserve would be `1-2d`.
The resulting ratio of UPRO as a function of `d` would be: `(UPRO) / (UPRO + CASH) = (1-d) / (1-d + 1-2d)`.
When `d` is 50%, the asset allocation of UPRO becomes 1.

#### rebalancing with EDV reserve

We can replace the cash reserve with EDV reserve.
In this case, when VOO drops by `d`, the EDV will increase by `0.5*d`.
Therefore, before rebalancing the UPRO would be `1-3d` and the EDV would be `1+0.5d`.
After the rebalance, the UPRO would be `1-d` and the EDV reserve would be `1-1.5d`.
The resulting ratio of UPRO as a function of `d` would be: `(UPRO) / (UPRO + EDV) = (1-d) / (1-d + 1-1.5d)`.
Note that when `d` is 50%, the asset allocation of UPRO would not reach 1.
Meaning, we can increase the leverage to around 1.7x and still be able to recover from a 50% drop in VOO.
In this case the new formual for UPRO asset allocation would be `(UPRO) / (UPRO + EDV) = (1-d) / (1-d + 0.75-1.5d)`.

In the same way that we introduced VOO to the mix in the Rebalancing 1.0, we can also include VOO in the mix.

#### Profiting from market volatility

In the previous approach we tried to maintain a fixed leverage ratio.
Usually when the markets drop, it is followed by a recovery.
Here the goal is to take advantage of the oscillations and profit from them.
Using the EDV as reserve, we rewrite the rebalancing as `(UPRO) / (UPRO + EDV) = (1) / (1 + 1.0-2.5d)`.
Here we buy UPRO to bring it back to its original value, instead of equivalent leverage of VOO.
As VOO drops, we buy even more UPRO, and when the VOO gains its losses, we sell the UPRO.
With the above equation, we can only fully recover from up to 40% drawdown.
Also the initial leverage is only 1.5x compared the 1.7x.

### Notes

It is worth mentioning that each approach has its benefits and disadvantages and should be chosen based on the individuals risk tolerance.
I have back-tested all of the above algorithms.
While usually the last approach results in highest average returns, it also has the largest drawdown.
Similarly, the simple rebalancing approach has the lowest drawdown and volatility.
