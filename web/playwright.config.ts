import { defineConfig, devices } from "@playwright/test";

/**
 * The end-to-end tests run against a *built* site, not the dev server.
 *
 * That matters here more than usual: the bundler moves and renames the ONNX
 * runtime's files, and the difference between "works in development" and
 * "hangs at load in production" was exactly that. Testing the artifact that
 * gets deployed is the only way that shows up before a visitor finds it.
 *
 * `vite preview` also sends the two cross-origin headers, without which the
 * runtime silently falls back to a single thread and every search takes twice
 * as long.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 30_000 },
  reporter: process.env["CI"] ? "github" : "list",
  retries: process.env["CI"] ? 1 : 0,

  use: {
    baseURL: "http://localhost:4173",
    trace: "retain-on-failure",
  },

  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],

  webServer: {
    command: "npm run build && npm run preview -- --port 4173 --strictPort",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env["CI"],
    timeout: 180_000,
  },
});
