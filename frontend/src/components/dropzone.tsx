import { ImageUp } from 'lucide-react'
import { useRef, useState } from 'react'

import { cn } from '@/lib/utils'

type Props = { onFile: (file: File) => void }

/** Drop, click to browse. (Paste is handled globally in App.) */
export function Dropzone({ onFile }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)

  const take = (files: FileList | null) => {
    const file = files?.[0]
    if (file?.type.startsWith('image/')) onFile(file)
  }

  return (
    <button
      type="button"
      onClick={() => input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        take(e.dataTransfer.files)
      }}
      className={cn(
        'flex aspect-[4/3] w-full flex-col items-center justify-center gap-3 rounded-lg border border-dashed text-muted-foreground transition-colors hover:border-foreground/40 hover:bg-muted/40',
        over && 'border-sky-400 bg-sky-400/5 text-foreground',
      )}
    >
      <ImageUp className="size-8" />
      <div className="text-center text-sm">
        <div className="font-medium text-foreground">Drop an image or click to browse</div>
        <div>or paste one with ⌘V</div>
      </div>
      <input ref={input} type="file" accept="image/*" hidden onChange={(e) => take(e.target.files)} />
    </button>
  )
}
