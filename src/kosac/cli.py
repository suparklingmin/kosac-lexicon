"""Command-line interface: ``kosac analyze "문장"`` and friends."""
import argparse
import json
import sys


def _build_parser():
  parser = argparse.ArgumentParser(
      prog='kosac', description='KOSAC Korean sentiment lexicon')
  sub = parser.add_subparsers(dest='command')

  analyze = sub.add_parser('analyze', help='analyze the sentiment of text')
  analyze.add_argument('text', nargs='?',
                       help='text to analyze; reads stdin (one text per line) if omitted')
  analyze.add_argument('--features', default='polarity',
                       help="comma-separated feature names, or 'all' (default: polarity)")
  analyze.add_argument('--ngrams', default='1,2,3', help='comma-separated n-gram lengths')
  analyze.add_argument('--min-freq', type=int, default=0)
  analyze.add_argument('--negation', action='store_true', help='enable negation handling')
  analyze.add_argument('--intensifier', action='store_true', help='enable intensifier handling')
  analyze.add_argument('--align', action='store_true',
                       help="seed Kiwi's user dictionary from the lexicon")
  analyze.add_argument('--compact', action='store_true',
                       help='emit one compact JSON object per line')

  sub.add_parser('features', help='list the available sentiment features')
  sub.add_parser('citation', help='print a BibTeX citation')
  return parser


def main(argv=None):
  parser = _build_parser()
  args = parser.parse_args(argv)

  if args.command == 'features':
    from . import FEATURES
    print('\n'.join(FEATURES))
    return 0

  if args.command == 'citation':
    from . import citation
    print(citation())
    return 0

  if args.command != 'analyze':
    parser.print_help()
    return 1

  from . import FEATURES, SentimentAnalyzer
  features = list(FEATURES) if args.features == 'all' else args.features.split(',')
  ngrams = [int(n) for n in args.ngrams.split(',')]
  try:
    analyzer = SentimentAnalyzer(
        features, ngrams=ngrams, min_freq=args.min_freq,
        negation=args.negation, intensifier=args.intensifier, align=args.align)
  except ImportError as exc:  # e.g. the Kiwi extra is not installed
    parser.error(str(exc))

  if args.text:
    texts = [args.text]
  else:
    texts = [line.rstrip('\n') for line in sys.stdin if line.strip()]

  results = analyzer.analyze_batch(texts)
  if args.compact:
    for result in results:
      print(json.dumps(result, ensure_ascii=False))
  else:
    payload = results[0] if len(results) == 1 else results
    print(json.dumps(payload, ensure_ascii=False, indent=2))
  return 0


if __name__ == '__main__':
  sys.exit(main())
