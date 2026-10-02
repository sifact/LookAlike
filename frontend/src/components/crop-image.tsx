import { useRef, useState } from 'react'

import type { Box } from '@/lib/api'

const MIN_SIZE = 0.05 // ignore accidental clicks / tiny drags

type Props = {
  src: string
  box: Box | null
  onBoxChange: (box: Box | null) => void
}

/** Image with a drag-to-select crop rectangle. Reports the box as fractions. */
export function CropImage({ src, box, onBoxChange }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const start = useRef<[number, number] | null>(null)
  const [draft, setDraft] = useState<Box | null>(null)

  const point = (e: React.PointerEvent): [number, number] => {
    const r = imgRef.current!.getBoundingClientRect()
    const clamp = (v: number) => Math.min(Math.max(v, 0), 1)
    return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)]
  }

  const onDown = (e: React.PointerEvent) => {
    e.preventDefault()
    start.current = point(e)
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  const boxTo = (e: React.PointerEvent): Box => {
    const [x, y] = point(e)
    const [sx, sy] = start.current!
    return [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)]
  }

  const onMove = (e: React.PointerEvent) => {
    if (start.current) setDraft(boxTo(e))
  }

  // Use the release position itself: the last pointermove can lag behind it.
  const onUp = (e: React.PointerEvent) => {
    if (!start.current) return
    const final = boxTo(e)
    start.current = null
    if (final[2] - final[0] >= MIN_SIZE && final[3] - final[1] >= MIN_SIZE) onBoxChange(final)
    setDraft(null)
  }

  const shown = draft ?? box

  return (
    // The inner wrapper shrinks to the rendered image, so box fractions map 1:1 onto
    // the photo (object-contain would add letterboxing and skew them).
    <div className="flex justify-center rounded-lg bg-muted select-none touch-none">
      <div className="relative overflow-hidden rounded-lg">
        <img
          ref={imgRef}
          src={src}
          alt="Query"
          draggable={false}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          className="block max-h-[420px] max-w-full cursor-crosshair"
        />
        {shown && (
          <div
            className="pointer-events-none absolute rounded-sm border-2 border-sky-400 bg-sky-400/10 shadow-[0_0_0_9999px_rgb(0_0_0/0.45)]"
            style={{
              left: `${shown[0] * 100}%`,
              top: `${shown[1] * 100}%`,
              width: `${(shown[2] - shown[0]) * 100}%`,
              height: `${(shown[3] - shown[1]) * 100}%`,
            }}
          />
        )}
      </div>
    </div>
  )
}
