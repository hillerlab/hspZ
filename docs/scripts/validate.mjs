import { readdir, readFile, stat } from "node:fs/promises"
import { dirname, join, relative, resolve } from "node:path"
import process from "node:process"
import { fileURLToPath } from "node:url"

const docsRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const journeysRoot = join(docsRoot, "src", "journeys")
const repoRoot = resolve(docsRoot, "..")
const requiredFields = ["WHAT GOES IN", "WHAT HAPPENS", "WHAT COMES OUT", "INVARIANT"]
const ids = new Set()
const errors = []

async function markdownFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const nested = await Promise.all(entries.map((entry) => {
    const path = join(dir, entry.name)
    return entry.isDirectory() ? markdownFiles(path) : entry.name.endsWith(".md") ? [path] : []
  }))
  return nested.flat()
}

function fail(file, message) {
  errors.push(`${relative(docsRoot, file)}: ${message}`)
}

async function validateSource(file, source) {
  if (!/^[^:\s]+::\S+$/.test(source)) {
    fail(file, `invalid @source "${source}"; expected path::symbol`)
    return
  }
  if (!process.argv.includes("--sources")) return

  const separator = source.indexOf("::")
  const path = source.slice(0, separator)
  const symbol = source.slice(separator + 2)
  const fullPath = resolve(repoRoot, path)
  if (!fullPath.startsWith(`${repoRoot}/`)) return fail(file, `@source escapes the repository: ${source}`)
  try {
    if (!(await stat(fullPath)).isFile()) return fail(file, `@source is not a file: ${path}`)
    const body = await readFile(fullPath, "utf8")
    const leaf = symbol.split("::").at(-1)
    if (!body.includes(leaf)) fail(file, `@source symbol was not found: ${source}`)
  } catch {
    fail(file, `@source file was not found: ${path}`)
  }
}

async function validate(file) {
  const markdown = await readFile(file, "utf8")
  const slides = [...markdown.matchAll(/^##\s+(.+)\n([\s\S]*?)(?=^##\s+|(?![\s\S]))/gm)]
  if (slides.length === 0) fail(file, "journey needs at least one ## slide")

  for (const [, title, body] of slides) {
    const id = body.match(/<!--\s*@id:\s*([^>]+?)\s*-->/)?.[1]
    if (!id) fail(file, `slide "${title}" is missing @id`)
    else if (ids.has(id)) fail(file, `duplicate @id "${id}"`)
    else ids.add(id)

    for (const field of requiredFields) {
      if (!new RegExp(`^${field}:\\s+\\S`, "m").test(body)) fail(file, `slide "${title}" is missing ${field}`)
    }

    const fences = [...body.matchAll(/```mermaid\s*\n([\s\S]*?)```/g)]
    if (fences.length !== 1) {
      fail(file, `slide "${title}" needs exactly one mermaid fence`)
      continue
    }
    const chart = fences[0][1]
    if (!/^\s*sequenceDiagram\s*$/m.test(chart.split("\n")[0])) fail(file, `slide "${title}" is not a sequenceDiagram`)
    const participants = [...chart.matchAll(/^\s*participant\s+/gm)].length
    const messages = [...chart.matchAll(/^\s*\S+\s*-+>>?\s*\S+\s*:/gm)].length
    if (participants > 5) fail(file, `slide "${title}" has ${participants} participants (max 5)`)
    if (messages > 8) fail(file, `slide "${title}" has ${messages} messages (max 8)`)
  }

  const sources = [...markdown.matchAll(/<!--\s*@source:\s*(.+?)\s*-->/g)].map((match) => match[1])
  if (sources.length === 0) fail(file, "journey needs at least one @source")
  await Promise.all(sources.map((source) => validateSource(file, source)))
}

const files = await markdownFiles(journeysRoot)
await Promise.all(files.map(validate))
if (errors.length) {
  console.error(errors.sort().join("\n"))
  process.exit(1)
}
console.log(`Validated ${files.length} journey file(s), ${ids.size} slide(s)${process.argv.includes("--sources") ? ", and source links" : ""}.`)
