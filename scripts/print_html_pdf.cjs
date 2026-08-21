const path = require("path");
const { pathToFileURL } = require("url");
const { chromium } = require("playwright");

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

async function main() {
  if (process.argv.length !== 4) {
    throw new Error("usage: node print_html_pdf.cjs <input.html> <output.pdf>");
  }

  const input = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  const browser = await chromium.launch({
    headless: true,
    executablePath: CHROME,
  });

  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(input).href, { waitUntil: "networkidle" });
    await page.emulateMedia({ media: "print" });
    await page.pdf({
      path: output,
      format: "A4",
      printBackground: true,
      preferCSSPageSize: true,
      displayHeaderFooter: false,
    });
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
