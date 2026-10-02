import argparse
from pathlib import Path

from .config import DEFAULT_BACKBONE, INDEXES


def cmd_data(a):
    from .data import download, prepare

    if not a.skip_download:
        download()
    df = prepare(a.per_category_train, a.per_category_test, a.max_queries, a.seed)
    print(df.groupby(["split", "role"]).agg(images=("image_id", "size"), products=("item_id", "nunique")))


def cmd_train(a):
    from .train import train

    out = train(a.mode, a.backbone, a.proj_dim, a.epochs, a.batch_size, a.lr, a.temperature,
                a.noise, a.name, a.max_steps, a.workers, a.seed)
    print(f"saved {out}")


def cmd_evaluate(a):
    from .evaluate import run

    print(run(a.models, a.k, a.ann))


def cmd_index(a):
    from .data import load_catalog
    from .embed import catalog_embeddings
    from .index import VectorIndex
    from .model import model_tag

    cat = load_catalog()
    vecs = catalog_embeddings(a.model, cat)
    index = VectorIndex.build(vecs, cat["image_id"].to_numpy(), a.kind)
    out = a.out or INDEXES / f"{model_tag(a.model)}_{a.kind}"
    index.save(out, {"model": str(a.model)})
    print(f"indexed {len(index)} images -> {out}")


def cmd_serve(a):
    import uvicorn

    from .service import create_app

    uvicorn.run(create_app(a.index), host=a.host, port=a.port)


def main():
    p = argparse.ArgumentParser(prog="visual-search")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("data", help="download Stanford Online Products and build the catalog")
    s.add_argument("--per-category-train", type=int, default=300, help="products per category")
    s.add_argument("--per-category-test", type=int, default=300)
    s.add_argument("--max-queries", type=int, default=2000)
    s.add_argument("--skip-download", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("train", help="contrastive fine-tuning")
    s.add_argument("--mode", choices=["selfsup", "interactions"], default="interactions")
    s.add_argument("--noise", type=float, default=0.0, help="interactions: fraction of random positives")
    s.add_argument("--backbone", default=DEFAULT_BACKBONE)
    s.add_argument("--proj-dim", type=int, default=256)
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=2e-5)
    s.add_argument("--temperature", type=float, default=0.05)
    s.add_argument("--max-steps", type=int)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--name")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="offline metrics for one or more models")
    s.add_argument("models", nargs="+", help="timm model names or .pt checkpoints")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--ann", help="also benchmark exact vs ANN indexes for this model")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("index", help="embed the whole catalog and build a serving index")
    s.add_argument("--model", required=True)
    s.add_argument("--kind", choices=["flat", "hnsw", "ivfpq"], default="hnsw")
    s.add_argument("--out", type=Path)
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("serve", help="run the search API and demo page")
    s.add_argument("--index", type=Path, required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
