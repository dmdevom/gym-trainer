# trAIner web

Responsive Next.js frontend for the trAIner API.

**Live:** https://gym-trainer-web-ai.vercel.app

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000. By default, `/backend-api/*` is rewritten to the
Railway API so local development works even before the latest backend CORS build
is deployed.

## Environment

- `BACKEND_API_URL`: server-side destination for the same-origin rewrite.
- `NEXT_PUBLIC_API_BASE_URL`: optional CORS-enabled API URL for direct browser
  requests. Leave unset to use the rewrite.

For direct requests, deploy the FastAPI change and set `CORS_ORIGINS` to a
comma-separated list containing the frontend's exact origins.

## Checks

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

Camera capture requires HTTPS in production (localhost is allowed by browsers).
Demo-video credits and modifications are documented in
`public/samples/ATTRIBUTION.md`.
