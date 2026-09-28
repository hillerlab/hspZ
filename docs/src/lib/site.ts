export const BASE = "/hspZ/"
export const DOC_VERSION = process.env.HSPZ_DOC_VERSION || "v0.0.4"
export const SOURCE_COMMIT = "f4dd8d56e732c10cd3e1c6cc7810b42e87a75150"

export function withBase(path = "") {
  return `${BASE}${path.replace(/^\/+/, "")}`
}

export function sourceUrl(source: string) {
  const path = source.split("::", 1)[0]
  return `https://github.com/hillerlab/hspZ/blob/${DOC_VERSION}/${path}`
}
