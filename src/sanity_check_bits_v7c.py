# sanity_check_bits_v7c.py — печатает извлечённые числа и итоговый битовый вектор
import os, argparse
from data.dataset import parse_bits_from_basename, _extract_ints

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--regex", default=r"(\\d{1,2})")
    ap.add_argument("--numbers_total", type=int, default=11)
    ap.add_argument("--bits", type=int, default=5)
    ap.add_argument("--ignore_last", action="store_true", default=True)
    args = ap.parse_args()

    shown = 0
    for dirpath, _, files in os.walk(args.root):
        for f in sorted(files):
            name, ext = os.path.splitext(f)
            if ext.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                ints = _extract_ints(name, args.regex)
                bits = parse_bits_from_basename(name, numbers_total=args.numbers_total, bits_per_number=args.bits,
                                                regex=args.regex, value_mode="mod", ignore_last_number=args.ignore_last)
                print(f"{f}")
                print("  numbers :", ints[:args.numbers_total], "(ignored last)" if args.ignore_last else "")
                print("  bits[K] :", bits.astype(int).tolist(), "K=", len(bits))
                shown += 1
                if shown >= 10: return

if __name__ == "__main__":
    main()
