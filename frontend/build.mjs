import esbuild from "esbuild";
import { cpSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const watch = process.argv.includes("--watch");

mkdirSync(resolve(__dirname, "dist/assets"), { recursive: true });

// copy HTML
cpSync(
  resolve(__dirname, "src/index.html"),
  resolve(__dirname, "dist/index.html")
);

const ctx = await esbuild.context({
  entryPoints: [resolve(__dirname, "src/main.ts")],
  bundle: true,
  outfile: resolve(__dirname, "dist/assets/main.js"),
  target: "es2020",
  format: "iife",
  minify: !watch,
  sourcemap: watch ? "inline" : false,
});

if (watch) {
  await ctx.watch();
  console.log("Watching for changes…");
} else {
  await ctx.rebuild();
  await ctx.dispose();
  console.log("Build complete → dist/");
}
