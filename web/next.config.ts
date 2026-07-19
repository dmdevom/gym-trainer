import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The UI caps video files at 15 MB. Leave room for multipart form headers so
  // valid uploads are not truncated while Next.js proxies them to Railway.
  experimental: {
    proxyClientMaxBodySize: "20mb",
  },
  async rewrites() {
    const backend = (process.env.BACKEND_API_URL || "https://gym-trainer-production-3c7f.up.railway.app").replace(/\/$/, "");
    return [{ source: "/backend-api/:path*", destination: `${backend}/:path*` }];
  },
};

export default nextConfig;
