import { defineConfig } from "astro/config"
import mdx from "@astrojs/mdx"
import react from "@astrojs/react"
import tailwindcss from "@tailwindcss/vite"

export default defineConfig({
  site: "https://hillerlab.github.io",
  base: "/hspZ",
  integrations: [mdx(), react()],
  vite: { plugins: [tailwindcss()] },
})
