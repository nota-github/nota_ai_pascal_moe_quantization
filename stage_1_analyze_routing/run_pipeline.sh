#!/usr/bin/env bash
# Dispatcher: run one phase or all in order (uses run1…run4 + config.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 {1|2|3|4|all}" >&2
    echo "  1  run1_first.sh   (steps 2–4; step1 still commented like pipeline.sh)" >&2
    echo "  2  run2_second.sh (expert threshold: step3/4 with thr only & step5 no token thr)" >&2
    echo "  3  run3_third.sh  (step5 with token thr)" >&2
    echo "  4  run4_fourth.sh (step6)" >&2
    echo "  all  run 1 then 2 then 3 then 4" >&2
    exit 1
}

case "${1:-}" in
    1) exec "${SCRIPT_DIR}/run1_first.sh" ;;
    2) exec "${SCRIPT_DIR}/run2_second.sh" ;;
    3) exec "${SCRIPT_DIR}/run3_third.sh" ;;
    4) exec "${SCRIPT_DIR}/run4_fourth.sh" ;;
    all)
        "${SCRIPT_DIR}/run1_first.sh"
        "${SCRIPT_DIR}/run2_second.sh"
        "${SCRIPT_DIR}/run3_third.sh"
        "${SCRIPT_DIR}/run4_fourth.sh"
        ;;
    *) usage ;;
esac
