import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep local development from writing generated agent instruction files.
  agentRules: false,
};

export default nextConfig;
