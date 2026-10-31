# ExamplePay webhooks (fictional)

## Delivery
Webhook delivery uses HTTPS POST requests. Return HTTP 200 within five seconds to acknowledge delivery. Failed webhook deliveries are retried up to eight times over 24 hours.

## Verification
Verify the X-ExamplePay-Signature header using the webhook signing secret and the raw request body. Reject timestamps older than five minutes to reduce replay attacks. The signing secret differs from your API token.

## Deduplication
Each webhook has an event_id. Store processed event identifiers and ignore duplicate events. Delivery order is not guaranteed; fetch the current payment state when event order matters.

