#!/usr/bin/env python3
"""CLI TechCulture AI Studio."""
import argparse
import json
from pathlib import Path

from engine import MoteurExtraction
from vision_engine import VisionEngine


def main():
    p = argparse.ArgumentParser(description="TechCulture AI Studio CLI")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("text", help="Analyser un texte")
    t.add_argument("input")
    t.add_argument("-o", "--output")
    t.add_argument("-f", "--format", choices=["json", "xml", "txt"], default="json")

    i = sub.add_parser("image", help="Analyser une image")
    i.add_argument("input")
    i.add_argument("-o", "--output")
    i.add_argument("-f", "--format", choices=["json", "xml", "txt"], default="json")
    i.add_argument("--no-preprocess", action="store_true")

    sub.add_parser("stats", help="Statistiques")
    args = p.parse_args()

    if args.cmd == "text":
        txt = Path(args.input).read_text(encoding="utf-8")
        m = MoteurExtraction()
        r = m.analyser_complete(txt)
        out = json.dumps(r, ensure_ascii=False, indent=2) if args.format == "json" else (
            r["xml"] if args.format == "xml" else
            "\n".join(f"[{e['type']}] {e['entite']} ({e['confiance']}%)" for e in r["entites"])
        )
        if args.output:
            Path(args.output).write_text(out, encoding="utf-8")
            print(f"✅ {args.output}")
        else:
            print(out)

    elif args.cmd == "image":
        img = Path(args.input).read_bytes()
        mv = VisionEngine()
        mn = MoteurExtraction()
        txt = mv.extraire_texte(img, use_preprocessing=not args.no_preprocess)
        r = mn.analyser_complete(txt)
        out = json.dumps(r, ensure_ascii=False, indent=2) if args.format == "json" else (
            r["xml"] if args.format == "xml" else txt
        )
        if args.output:
            Path(args.output).write_text(out, encoding="utf-8")
            print(f"✅ {args.output}")
        else:
            print(out)

    elif args.cmd == "stats":
        print(json.dumps(MoteurExtraction().get_stats(), indent=2))
    else:
        p.print_help()


if __name__ == "__main__":
    main()