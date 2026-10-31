# ExamplePay integration handbook (fictional)

## Retries
Retry failed payment requests at most three times. Use exponential backoff with delays of 1, 2, and 4 seconds. Retry only HTTP 429 and HTTP 503. Never retry HTTP 400 or HTTP 401 without correcting the request.

## Idempotency
Send an Idempotency-Key header on every payment creation request. Reuse the same key when retrying the same payment. Keys expire after 24 hours. A new key can create a second payment.

## Timeouts
The payment request timeout is 10 seconds. A timeout does not prove the payment failed. Check the payment status before attempting another payment.

