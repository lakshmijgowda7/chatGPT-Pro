/** @type {import('next').NextConfig} */
const backendHost = process.env.BACKEND_HOST
  ? `https://${process.env.BACKEND_HOST}`
  : (process.env.BACKEND_INTERNAL_URL || "http://127.0.0.1:8000");

const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: "/api/v1",
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendHost}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
