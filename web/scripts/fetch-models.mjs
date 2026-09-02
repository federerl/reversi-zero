/**
 * Fetch the trained networks from the GitHub Release, and check them.
 *
 * The `.onnx` files are not in git. They are model weights, and the rule that
 * keeps checkpoints out applies to them too — so a build that clones the
 * repository has the whole site except the thing that plays. Without this step
 * the page loads, renders a board, and sits at "Loading the agent…" forever.
 *
 * Cloudflare Pages builds from the repository rather than from an artifact we
 * assembled, so the fetching has to happen inside the build. That is what this
 * is: the deploy workflow's model step, moved to where the build actually runs.
 *
 * **Every file is verified against the checksum recorded when it was exported.**
 * A truncated download would still be a valid-looking file, ONNX Runtime would
 * refuse it at load, and the site would fail in the browser with no clue why. It
 * is cheaper to fail here, loudly, with the name of the file.
 *
 * Skips anything already present, so a local build uses the models you exported
 * rather than re-downloading them.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const manifestPath = join(here, "..", "src", "engine", "models.json");
const modelsDir = join(here, "..", "public", "models");

// Overridable so a build can be pinned to an older set of weights, which is what
// you would want if a deploy ever had to be rolled back to match a checkpoint.
const REPO = process.env["MODELS_REPO"] ?? "federerl/reversi-zero";
const TAG = process.env["MODELS_TAG"] ?? "models-v1";

const base = `https://github.com/${REPO}/releases/download/${TAG}`;

/** The filenames the app expects, taken from the manifest rather than guessed. */
function wanted() {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  return manifest.models.map((model) => model.url.split("/").pop());
}

async function download(name) {
  const url = `${base}/${name}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `could not fetch ${name} (${response.status} from ${url}).\n` +
        `  The release '${TAG}' must exist on ${REPO} and carry every .onnx and .json.\n` +
        `  Create it with:\n` +
        `    gh release create ${TAG} web/public/models/*.onnx web/public/models/*.json`,
    );
  }
  return Buffer.from(await response.arrayBuffer());
}

async function main() {
  mkdirSync(modelsDir, { recursive: true });
  const names = wanted();

  let fetched = 0;
  for (const onnx of names) {
    const sidecar = onnx.replace(/\.onnx$/, ".json");
    const onnxPath = join(modelsDir, onnx);
    const sidecarPath = join(modelsDir, sidecar);

    if (existsSync(onnxPath) && existsSync(sidecarPath)) {
      console.log(`  ${onnx.padEnd(28)} already here`);
      continue;
    }

    const [weights, meta] = await Promise.all([download(onnx), download(sidecar)]);

    // Check before writing, so a bad download never lands on disk where a later
    // build would find it and treat it as already fetched.
    const expected = JSON.parse(meta.toString("utf8")).sha256;
    const actual = createHash("sha256").update(weights).digest("hex");
    if (expected && expected !== actual) {
      throw new Error(
        `${onnx} does not match the checksum recorded when it was exported.\n` +
          `  expected ${expected}\n  got      ${actual}\n` +
          `  The download was damaged, or the release asset is not the file it claims.`,
      );
    }

    writeFileSync(onnxPath, weights);
    writeFileSync(sidecarPath, meta);
    fetched++;
    console.log(`  ${onnx.padEnd(28)} ${(weights.length / 1e6).toFixed(2)} MB  ok ${actual.slice(0, 12)}…`);
  }

  console.log(
    fetched === 0
      ? `  all ${names.length} networks were already present`
      : `  fetched ${fetched} of ${names.length} from ${TAG}`,
  );
}

main().catch((error) => {
  console.error(`\n${error.message}\n`);
  process.exit(1);
});
