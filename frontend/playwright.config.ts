import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  workers: 1,
  timeout: 60_000,
  use: {
    channel: "chromium",
    baseURL: "http://127.0.0.1:13000",
    viewport: { width: 1440, height: 1000 },
    trace: "retain-on-failure",
  },
  webServer: {
    command: "BACKEND_PORT=18000 FRONTEND_PORT=13000 python3 ../scripts/dev.py",
    url: "http://127.0.0.1:13000/login",
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
