# ExamplePay authentication (fictional)

## Requests
Authenticate API requests with an Authorization header containing Bearer followed by your API token. Create tokens in the developer dashboard. Keep tokens on the server and never include them in browser code.

## Rotation
Rotate API tokens every 90 days. Revoke a compromised token immediately. A revoked token returns HTTP 401 on subsequent requests. The dashboard supports two active tokens so you can rotate without downtime.

## Environments
Sandbox tokens start with test_. Production tokens start with live_. Sandbox tokens cannot access production payments. Store tokens in environment variables or a secret manager.

