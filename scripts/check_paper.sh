#!/bin/zsh
# Gate for the final build. Fails if the manuscript still has placeholders or undefined
# references/citations.
set -u
cd "$(dirname "$0")/../paper"
fail=0
if grep -n '\\pending' sections/*.tex appendix.tex tables/*.tex | grep -vE '^[^:]+:[0-9]+:\s*%'; then
  echo "FAIL: \\pending placeholders remain (above)"; fail=1
fi
tectonic -X compile main.tex --keep-logs > ../logs/paper_build.txt 2>&1 || { echo "FAIL: build"; fail=1; }
if grep -E "undefined|There were undefined" main.log ../logs/paper_build.txt; then
  echo "FAIL: undefined references/citations"; fail=1
fi
[ $fail -eq 0 ] && echo "PAPER CHECK PASSED ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
exit $fail
