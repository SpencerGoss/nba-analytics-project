#!/usr/bin/env node
/**
 * optimize_dashboard.js
 * Minifies inline JS/CSS in dashboard HTML files and copies everything to dashboard/dist/.
 * Uses only Node.js built-in modules. Run with --dry-run to preview savings.
 */
const fs = require('fs');
const path = require('path');

const DASHBOARD_DIR = path.resolve(__dirname, '..', 'dashboard');
const DIST_DIR = path.join(DASHBOARD_DIR, 'dist');
const HTML_FILES = ['index.html', 'about.html'];
const COPY_FILES = ['manifest.json', 'robots.txt', 'sitemap.xml', 'sw.js', 'og-image.svg'];
const COPY_DIRS = ['data'];
const DRY_RUN = process.argv.includes('--dry-run');

function minifyCSS(css) {
  // Remove /* */ comments
  css = css.replace(/\/\*[\s\S]*?\*\//g, '');
  // Collapse whitespace
  css = css.replace(/\s+/g, ' ');
  // Remove spaces around punctuation
  css = css.replace(/\s*([{}:;,>~+])\s*/g, '$1');
  // Remove trailing semicolons before closing braces
  css = css.replace(/;}/g, '}');
  return css.trim();
}

function minifyJS(js) {
  // Remove single-line comments but not inside strings
  // Strategy: walk through, skip string literals, remove // comments outside them
  let result = '';
  let i = 0;
  while (i < js.length) {
    // String literals
    if (js[i] === '"' || js[i] === "'" || js[i] === '`') {
      const quote = js[i];
      result += js[i++];
      while (i < js.length && js[i] !== quote) {
        if (js[i] === '\\') { result += js[i++]; }
        if (i < js.length) { result += js[i++]; }
      }
      if (i < js.length) result += js[i++]; // closing quote
    }
    // Block comments
    else if (js[i] === '/' && js[i + 1] === '*') {
      i += 2;
      while (i < js.length && !(js[i] === '*' && js[i + 1] === '/')) i++;
      i += 2; // skip */
    }
    // Single-line comments
    else if (js[i] === '/' && js[i + 1] === '/') {
      i += 2;
      while (i < js.length && js[i] !== '\n') i++;
    }
    else {
      result += js[i++];
    }
  }
  // Collapse whitespace (newlines/spaces/tabs) into single space
  result = result.replace(/[ \t]*\n[ \t]*/g, '\n');
  result = result.replace(/\n+/g, '\n');
  result = result.replace(/[ \t]+/g, ' ');
  return result.trim();
}

function minifyHTML(html) {
  // Minify contents of <style>...</style> tags
  html = html.replace(/(<style[^>]*>)([\s\S]*?)(<\/style>)/gi, (_, open, css, close) => {
    return open + minifyCSS(css) + close;
  });
  // Minify contents of <script>...</script> tags (skip external src scripts)
  html = html.replace(/(<script(?![^>]*\bsrc\b)[^>]*>)([\s\S]*?)(<\/script>)/gi, (_, open, js, close) => {
    return open + minifyJS(js) + close;
  });
  return html;
}

function copyDirSync(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirSync(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  return (bytes / 1024).toFixed(1) + ' KB';
}

// --- Main ---
console.log(DRY_RUN ? '=== DRY RUN (no files written) ===' : '=== Optimizing dashboard ===');

let totalOriginal = 0;
let totalMinified = 0;

if (!DRY_RUN) {
  fs.mkdirSync(DIST_DIR, { recursive: true });
  console.log(`Created ${path.relative(process.cwd(), DIST_DIR)}/`);
}

// Minify HTML files
for (const file of HTML_FILES) {
  const srcPath = path.join(DASHBOARD_DIR, file);
  if (!fs.existsSync(srcPath)) { console.log(`  SKIP ${file} (not found)`); continue; }
  const original = fs.readFileSync(srcPath, 'utf8');
  const minified = minifyHTML(original);
  const origSize = Buffer.byteLength(original, 'utf8');
  const minSize = Buffer.byteLength(minified, 'utf8');
  const pct = ((1 - minSize / origSize) * 100).toFixed(1);
  totalOriginal += origSize;
  totalMinified += minSize;
  console.log(`  ${file}: ${formatBytes(origSize)} -> ${formatBytes(minSize)} (${pct}% smaller)`);
  if (!DRY_RUN) {
    fs.writeFileSync(path.join(DIST_DIR, file), minified, 'utf8');
  }
}

// Copy static files
for (const file of COPY_FILES) {
  const srcPath = path.join(DASHBOARD_DIR, file);
  if (!fs.existsSync(srcPath)) { console.log(`  SKIP ${file} (not found)`); continue; }
  const size = fs.statSync(srcPath).size;
  totalOriginal += size;
  totalMinified += size;
  console.log(`  ${file}: ${formatBytes(size)} (copied)`);
  if (!DRY_RUN) fs.copyFileSync(srcPath, path.join(DIST_DIR, file));
}

// Copy directories
for (const dir of COPY_DIRS) {
  const srcPath = path.join(DASHBOARD_DIR, dir);
  if (!fs.existsSync(srcPath)) { console.log(`  SKIP ${dir}/ (not found)`); continue; }
  console.log(`  ${dir}/: copying recursively...`);
  if (!DRY_RUN) copyDirSync(srcPath, path.join(DIST_DIR, dir));
}

// Summary
const saved = totalOriginal - totalMinified;
const pctTotal = totalOriginal > 0 ? ((saved / totalOriginal) * 100).toFixed(1) : '0.0';
console.log(`\nTotal: ${formatBytes(totalOriginal)} -> ${formatBytes(totalMinified)} (saved ${formatBytes(saved)}, ${pctTotal}%)`);
if (DRY_RUN) console.log('Run without --dry-run to write files.');
else console.log(`Output written to ${path.relative(process.cwd(), DIST_DIR)}/`);
