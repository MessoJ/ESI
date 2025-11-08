# ESI
A composite indicator that tracks macroeconomic and financial stress in near‑real time

## Secrets management (production)

- Do not use .env for production secrets. Use your platform's secret store:
  - Fly.io: `fly secrets set API_KEY_SECRET=... SENTRY_DSN=... REDIS_URL=...`
  - AWS: AWS Secrets Manager or SSM Parameter Store; inject into task env
  - Render/Heroku: config vars

## Security hardening

- Configure CORS via `ALLOW_ORIGINS` environment variable (comma-separated). In production, set it to your frontend origin(s).
- HTTPS-only: terminate TLS at load balancer/ingress and force HTTPS; if running directly, put behind a reverse proxy that redirects HTTP→HTTPS.
- Protected endpoints use `X-API-KEY` header. Set `API_KEY_SECRET` and include the header when calling admin/alerts endpoints.
- CSV upload sanitization protects against formula injection and invalid content.

## Rate limiting & caching

- Redis-backed caching: latest (10 min), history (1 hour), components (10 min).
- On upstream failure, API returns last cached body with `X-Data-Stale: true`.

## Observability

- JSON logs to stdout
- Prometheus `/metrics` exposes `esi_last_compute_timestamp` gauge
- Sentry (set `SENTRY_DSN`) for error tracking

## CI

- Configure GitHub Actions to run tests and coverage on push/PR. See `.github/workflows/ci.yml`.

## Production Docker

- Build API with `infra/docker/api.prod.Dockerfile` (gunicorn+uvicorn, non-root)
- Staging compose: `docker-compose -f docker-compose.prod.yml up --build`