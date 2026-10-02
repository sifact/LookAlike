// Client for the FastAPI backend (src/visual_search/service.py).
// Relative URLs: proxied by Vite in dev, same origin when served by FastAPI.

export type Health = {
  status: string
  model: string
  index: string
  images: number
  device: string
}

export type SearchResult = {
  image_id: number
  score: number
  item_id: string | null
  category: string | null
  url: string
}

export type SearchResponse = {
  results: SearchResult[]
  timings_ms: { embed: number; ann: number; rerank: number }
}

export type IndexedImage = {
  image_id: number
  item_id: string
  category: string
}

/** Crop box as fractions of the image: [x0, y0, x1, y1], each 0..1. */
export type Box = [number, number, number, number]

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

export async function getHealth(signal?: AbortSignal): Promise<Health> {
  return parse(await fetch('/health', { signal }))
}

export async function search(file: Blob, k: number, box: Box | null, signal?: AbortSignal): Promise<SearchResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('k', String(k))
  if (box) form.append('box', box.map((v) => v.toFixed(4)).join(','))
  return parse(await fetch('/search', { method: 'POST', body: form, signal }))
}

export async function addImage(file: Blob): Promise<IndexedImage> {
  const form = new FormData()
  form.append('file', file)
  return parse(await fetch('/images', { method: 'POST', body: form }))
}

export async function fetchImage(url: string): Promise<Blob> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load image (${res.status})`)
  return res.blob()
}
