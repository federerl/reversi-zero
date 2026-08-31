/**
 * Empty `dist/` before a production build, and say something useful if it cannot.
 *
 * Vite normally clears its own output directory. That is turned off here
 * (`build.emptyOutDir: false`) because the development loop runs
 * `vite build --watch` and `vite preview` at the same time: preview serves files
 * out of `dist/` while the watch build wants to delete them, and on Windows an
 * open file handle makes the delete fail outright rather than wait.
 *
 * So clearing became an explicit step that only the real build takes. When it
 * does fail, the reason is almost always a server still running, and the message
 * says so -- the alternative is an EPERM stack trace forty lines deep in the
 * bundler, which names the symptom and not the cause.
 */

import { rmSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const dist = join(dirname(fileURLToPath(import.meta.url)), "..", "dist");

if (!existsSync(dist)) process.exit(0);

// A couple of retries: a handle released a moment ago can still linger briefly
// on Windows, and failing on the first attempt would be needlessly brittle.
for (let attempt = 1; attempt <= 3; attempt++) {
  try {
    rmSync(dist, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 });
    process.exit(0);
  } catch (error) {
    if (attempt === 3) {
      console.error(
        `\nCould not empty ${dist}\n\n` +
          `  ${error.message}\n\n` +
          `Something still has those files open. Almost always that is a\n` +
          `\`npm run dev\` or \`npm run preview\` running in another terminal --\n` +
          `stop it with Ctrl+C and run this again.\n\n` +
          `If nothing is running, a stray process survived a hard kill:\n` +
          `  Windows   taskkill /F /IM node.exe\n` +
          `  macOS     pkill -f vite\n`,
      );
      process.exit(1);
    }
  }
}
