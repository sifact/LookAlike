import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import type { SearchResult } from '@/lib/api'
import { cn } from '@/lib/utils'

type Props = {
  result: SearchResult
  rank: number
  sameProduct: boolean
  onSelect: (result: SearchResult) => void
}

export function ResultCard({ result, rank, sameProduct, onSelect }: Props) {
  return (
    <button
      type="button"
      onClick={() => onSelect(result)}
      title="Search with this image"
      className={cn(
        'group overflow-hidden rounded-xl bg-card text-left ring-1 ring-foreground/10 transition hover:ring-foreground/30',
        sameProduct && 'ring-2 ring-emerald-500/70 hover:ring-emerald-400',
      )}
    >
      <div className="relative aspect-square overflow-hidden bg-muted">
        <img
          src={result.url}
          alt={result.category ?? ''}
          loading="lazy"
          className="size-full object-cover transition-transform duration-300 group-hover:scale-105"
        />
        <Badge variant="secondary" className="absolute top-2 left-2 bg-background/80 backdrop-blur">
          #{rank}
        </Badge>
        {sameProduct && <Badge className="absolute top-2 right-2 bg-emerald-600 text-white">Same product</Badge>}
      </div>
      <div className="space-y-2 p-3">
        <div className="flex items-center justify-between gap-2 text-sm">
          <span className="truncate font-medium capitalize">{result.category ?? 'unknown'}</span>
          <span className="font-mono text-xs text-muted-foreground tabular-nums">{result.score.toFixed(3)}</span>
        </div>
        <Progress value={Math.max(0, result.score) * 100} className="h-1" />
        <div className="truncate font-mono text-xs text-muted-foreground">item {result.item_id}</div>
      </div>
    </button>
  )
}
