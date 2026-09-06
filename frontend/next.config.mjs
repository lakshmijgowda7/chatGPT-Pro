/** @type {import('next').NextConfig} */
const backendTarget =
  process.env.BACKEND_URL ||
  (process.env.NODE_ENV === "production"
    ? "https://chatgpt-pro-backend.onrender.com"
    : "http://127.0.0.1:8000");

const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: "/api/v1",
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendTarget}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
