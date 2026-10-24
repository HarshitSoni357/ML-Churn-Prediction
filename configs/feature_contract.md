
# Feature Engineering Contract

## Prediction timestamp
T = Monday 00:00.

## Feature boundary
Only information available at or before T may be used.

## Rolling windows
- 7 days
- 30 days
- 90 days

## Sources
- Customers: profile and tenure
- Orders: counts, spend, frequency, recency and category mix
- Payments: failure counts, failure rate and payment methods
- Support: ticket volume, priority and resolution time known by T
- Subscriptions: active plan, billing cycle and tenure
- Marketing: event counts and engagement ratios
- Feedback: mean rating, negative-feedback rate and feedback count

## Missingness
No feedback means an unknown rating, not a rating of zero.
No payment failures means a failure count of zero.
No observed support tickets means a ticket count of zero.

## Target
inactive_next_30d = 1 when an eligible customer has
no qualifying orders during (T, T + 30 days].

The target is an inactivity proxy, not formal subscription cancellation.
