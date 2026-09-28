#!/bin/sh
set -eu

binary=${1:?usage: generate-cli-reference.sh HSPZ OUTPUT}
output=${2:?usage: generate-cli-reference.sh HSPZ OUTPUT}
divider=------------------------------------------------------------------------------

emit() {
  command=$1
  if [ -n "$command" ]; then
    printf '$ hspZ %s --help\n\n' "$command"
    "$binary" "$command" --help
  else
    printf '$ hspZ --help\n\n'
    "$binary" --help
  fi
}

{
  emit ''
  for command in run index benchmark compare hits-estimate; do
    printf '\n%s\n\n' "$divider"
    emit "$command"
  done
} > "$output"
