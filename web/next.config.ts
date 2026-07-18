import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const backend = (process.env.BACKEND_API_URL || "https://gym-trainer-production-a2c6.up.railway.app").replace(/\/$/, "");
    return [{ source: "/backend-api/:path*", destination: `${backend}/:path*` }];
  },
};

export default nextConfig;
