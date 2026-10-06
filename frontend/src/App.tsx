import { Crop, ImagePlus, Loader2, RefreshCw, ScanSearch, Upload, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'

import { CropImage } from '@/components/crop-image'
import { Dropzone } from '@/components/dropzone'
import { ResultCard } from '@/components/result-card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { Toaster } from '@/components/ui/sonner'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  addImage,
  type Box,
  fetchImage,
  getHealth,
  type Health,
  search,
  type SearchResponse,
  type SearchResult,
} from '@/lib/api'

type Query = {
  blob: Blob
  url: string // object URL for display
  itemId: string | null // known when the query came from a catalog result
}

const HISTORY = 8

function makeQuery(blob: Blob, itemId: string | null = null): Query {
  return { blob, url: URL.createObjectURL(blob), itemId }
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [offline, setOffline] = useState(false)
  const [query, setQuery] = useState<Query | null>(null)
  const [history, setHistory] = useState<Query[]>([])
  const [box, setBox] = useState<Box | null>(null)
  const [k, setK] = useState(20)
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [adding, setAdding] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)

  const refreshHealth = useCallback(() => {
    getHealth()
      .then((h) => {
        setHealth(h)
        setOffline(false)
      })
      .catch(() => setOffline(true))
  }, [])

  useEffect(refreshHealth, [refreshHealth])

  const select = useCallback((q: Query) => {
    setQuery(q)
    setBox(null)
    setHistory((prev) => {
      const next = [q, ...prev.filter((p) => p !== q)]
      next.slice(HISTORY).forEach((p) => URL.revokeObjectURL(p.url))
      return next.slice(0, HISTORY)
    })
  }, [])

  // Search whenever the query image, crop or k changes. Abort stale requests.
  useEffect(() => {
    if (!query) return
    const ctrl = new AbortController()
    setLoading(true)
    search(query.blob, k, box, ctrl.signal)
      .then(setResponse)
      .catch((e: Error) => {
        if (e.name !== 'AbortError') toast.error(e.message)
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false)
      })
    return () => ctrl.abort()
  }, [query, box, k])

  // Paste an image from the clipboard; Esc clears the crop.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith('image/'))
      if (file) select(makeQuery(file))
    }
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setBox(null)
    window.addEventListener('paste', onPaste)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('paste', onPaste)
      window.removeEventListener('keydown', onKey)
    }
  }, [select])

  const searchFromResult = async (r: SearchResult) => {
    try {
      select(makeQuery(await fetchImage(r.url), r.item_id))
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  const addToIndex = async () => {
    if (!query) return
    setAdding(true)
    try {
      const img = await addImage(query.blob)
      toast.success(`Indexed as image #${img.image_id}`, { description: 'It is now searchable by everyone.' })
      refreshHealth()
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setAdding(false)
    }
  }

  const results = response?.results ?? []

  return (
    <div className="min-h-svh bg-background">
      <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-lg bg-sky-500/15 text-sky-400">
              <ScanSearch className="size-5" />
            </div>
            <div>
              <h1 className="font-heading text-base leading-tight font-semibold">LookAlike</h1>
              <p className="text-xs text-muted-foreground">Find visually similar products from a photo or a crop</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {offline ? (
              <Badge variant="destructive">API offline</Badge>
            ) : health ? (
              <>
                <Badge variant="outline" className="gap-1.5">
                  <span className="size-1.5 rounded-full bg-emerald-500" /> Online
                </Badge>
                <Badge variant="secondary">{health.images.toLocaleString()} images</Badge>
                <Badge variant="secondary" className="uppercase">
                  {health.index}
                </Badge>
                <Badge variant="secondary">{health.device}</Badge>
              </>
            ) : (
              <Skeleton className="h-5 w-48" />
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
          <Card>
            <CardHeader>
              <CardTitle>Query image</CardTitle>
              <CardDescription>
                {query ? 'Drag on the image to search just one region.' : 'Upload a product photo to start.'}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {query ? (
                <CropImage src={query.url} box={box} onBoxChange={setBox} />
              ) : (
                <Dropzone onFile={(f) => select(makeQuery(f))} />
              )}

              {query && (
                <div className="flex flex-wrap gap-2">
                  <Button variant="outline" size="sm" onClick={() => fileInput.current?.click()}>
                    <Upload /> Replace
                  </Button>
                  <Button variant="outline" size="sm" disabled={!box} onClick={() => setBox(null)}>
                    <X /> Clear crop
                  </Button>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="outline" size="sm" disabled={adding} onClick={addToIndex}>
                        {adding ? <Loader2 className="animate-spin" /> : <ImagePlus />} Add to index
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Embed this image and make it searchable</TooltipContent>
                  </Tooltip>
                </div>
              )}
              <input
                ref={fileInput}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) select(makeQuery(f))
                  e.target.value = ''
                }}
              />

              <Separator />

              <div className="flex items-center justify-between gap-3">
                <span className="text-sm text-muted-foreground">Results</span>
                <Select value={String(k)} onValueChange={(v) => setK(Number(v))}>
                  <SelectTrigger size="sm" className="w-24">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[10, 20, 40].map((n) => (
                      <SelectItem key={n} value={String(n)}>
                        Top {n}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {response && (
                <div className="grid grid-cols-3 gap-2 text-center">
                  {(
                    [
                      ['Embed', response.timings_ms.embed],
                      ['ANN', response.timings_ms.ann],
                      ['Re-rank', response.timings_ms.rerank],
                    ] as const
                  ).map(([label, ms]) => (
                    <div key={label} className="rounded-lg bg-muted/50 px-2 py-2">
                      <div className="font-mono text-sm tabular-nums">{ms} ms</div>
                      <div className="text-xs text-muted-foreground">{label}</div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {history.length > 1 && (
            <Card size="sm">
              <CardHeader>
                <CardTitle>Recent queries</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-2">
                  {history.map((h) => (
                    <button
                      key={h.url}
                      type="button"
                      onClick={() => select(h)}
                      className={`aspect-square overflow-hidden rounded-md ring-1 ring-foreground/10 hover:ring-foreground/40 ${h === query ? 'ring-2 ring-sky-400' : ''}`}
                    >
                      <img src={h.url} alt="" className="size-full object-cover" />
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </aside>

        <section className="min-w-0 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <h2 className="font-heading text-lg font-semibold">Similar images</h2>
              {response && !loading && <Badge variant="secondary">{results.length}</Badge>}
              {box && (
                <Badge variant="outline" className="gap-1 border-sky-400/50 text-sky-300">
                  <Crop className="size-3" /> cropped
                </Badge>
              )}
              {loading && <Loader2 className="size-4 animate-spin text-muted-foreground" />}
            </div>
            {query?.itemId && (
              <p className="text-xs text-muted-foreground">
                Green border = same product as the query (item {query.itemId})
              </p>
            )}
          </div>

          {!query ? (
            <EmptyState offline={offline} onRetry={refreshHealth} />
          ) : loading && !response ? (
            <Grid>
              {Array.from({ length: 8 }, (_, i) => (
                <Skeleton key={i} className="aspect-[4/5] rounded-xl" />
              ))}
            </Grid>
          ) : (
            <Grid className={loading ? 'opacity-60 transition-opacity' : 'transition-opacity'}>
              {results.map((r, i) => (
                <ResultCard
                  key={r.image_id}
                  result={r}
                  rank={i + 1}
                  sameProduct={!!query.itemId && r.item_id === query.itemId}
                  onSelect={searchFromResult}
                />
              ))}
            </Grid>
          )}
        </section>
      </main>
      <Toaster position="bottom-right" />
    </div>
  )
}

function Grid({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 ${className}`}>{children}</div>
  )
}

function EmptyState({ offline, onRetry }: { offline: boolean; onRetry: () => void }) {
  return (
    <div className="flex min-h-[420px] flex-col items-center justify-center gap-3 rounded-xl border border-dashed px-6 text-center">
      <ScanSearch className="size-10 text-muted-foreground" />
      {offline ? (
        <>
          <p className="font-medium">Can't reach the search API</p>
          <p className="max-w-sm text-sm text-muted-foreground">
            Start it with <code className="rounded bg-muted px-1 py-0.5">uv run lookalike serve --index …</code>
          </p>
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw /> Retry
          </Button>
        </>
      ) : (
        <>
          <p className="font-medium">No query yet</p>
          <p className="max-w-sm text-sm text-muted-foreground">
            Upload, drop or paste a photo of a product. Results appear here, best match first. Click any result to
            search from it.
          </p>
        </>
      )}
    </div>
  )
}
