"""Command-line interface.

  python make_demo.py
  python run.py analyze data/demo/forged.png [--model models/splicing.joblib]
  python run.py train --root /path/to/CASIA2 --n 300
  python run.py eval --forged data/demo --gt data/demo --authentic data/demo_auth
"""
import argparse

from forensics.metrics import evaluate_copy_move
from forensics.pipeline import analyze
from forensics.splicing import train


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("image")
    a.add_argument("--model", default="models/splicing.joblib")
    a.add_argument("--out", default="results")

    t = sub.add_parser("train")
    t.add_argument("--root", required=True, help="folder containing Au/ and Tp/")
    t.add_argument("--n", type=int, default=300, help="images per class")
    t.add_argument("--no-gabor", action="store_true")
    t.add_argument("--out", default="models/splicing.joblib")

    e = sub.add_parser("eval")
    e.add_argument("--forged", required=True)
    e.add_argument("--gt")
    e.add_argument("--authentic")
    e.add_argument("--limit", type=int, default=100)

    args = ap.parse_args()
    if args.cmd == "analyze":
        r = analyze(args.image, args.model, out_dir=args.out)
        cm = r["copy_move"]
        print(f"Verdict        : {r['verdict']}")
        print(f"Keypoints      : {cm.n_keypoints}")
        print(f"Matches/Inliers: {cm.n_matches} / {cm.n_inliers}")
        if r["splicing_prob"] is not None:
            print(f"Splicing prob. : {r['splicing_prob']:.2f}")
        print(f"Overlay saved  : {r['overlay_path']}")
    elif args.cmd == "train":
        train(args.root, args.n, not args.no_gabor, args.out)
    elif args.cmd == "eval":
        evaluate_copy_move(args.forged, args.gt, args.authentic, args.limit)


if __name__ == "__main__":
    main()
