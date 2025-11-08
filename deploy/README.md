# Deployment Notes

## Fly.io
- `fly launch` (choose Dockerfile: `infra/docker/api.prod.Dockerfile`)
- `fly secrets set API_KEY_SECRET=... FRED_API_KEY=... REDIS_URL=... SENTRY_DSN=... ALLOW_ORIGINS=https://yourapp.example`
- `fly deploy`

## Render
- Create a Web Service; Dockerfile path `infra/docker/api.prod.Dockerfile`
- Add env vars: `API_KEY_SECRET`, `FRED_API_KEY`, `REDIS_URL`, `SENTRY_DSN`, `ALLOW_ORIGINS`

## AWS Fargate (ECS)
- Build & push image to ECR
- Create ECS Task with env vars:
  - `API_KEY_SECRET`, `FRED_API_KEY`, `REDIS_URL`, `SENTRY_DSN`, `ALLOW_ORIGINS`
- Service behind ALB with HTTPS
- Use AWS Secrets Manager/SSM Parameter Store for secrets



