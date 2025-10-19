# sanity_check_bits_noregex.py — проверка распознавания чисел БЕЗ regex
import os, argparse
from data.dataset import extract_ints_no_regex, parse_bits_from_basename

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--numbers_total", type=int, default=11)
    ap.add_argument("--bits", type=int, default=5)
    ap.add_argument("--ignore_last", action="store_true", default=True)
    args = ap.parse_args()

    shown = 0
    for dirpath, _, files in os.walk(args.root):
        for f in sorted(files):
            name, ext = os.path.splitext(f)
            if ext.lower() in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
                ints = extract_ints_no_regex(name)
                bits = parse_bits_from_basename(name, numbers_total=args.numbers_total,
                                                bits_per_number=args.bits, ignore_last_number=args.ignore_last)
                print(f"{f}")
                print("  numbers :", ints[:args.numbers_total], "(ignored last)" if args.ignore_last else "")
                print("  bits[K] :", bits.astype(int).tolist(), "K=", len(bits))
                shown += 1
                if shown >= 10: return

if __name__ == "__main__":
    main()
