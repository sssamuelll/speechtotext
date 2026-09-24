#!/bin/sh
# Writes meeting.m4a, the clip behind docs/img/demo.gif: six turns between two
# macOS text-to-speech voices with 1.6 s of silence between turns, encoded as
# AAC so the demo opens a container, not a wav. No person's voice is in it.
# Needs macOS (`say`) and ffmpeg.
set -eu
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

say -v Daniel   -o "$tmp/l1.aiff" "Morning. Did the overnight benchmark finish?"
say -v Samantha -o "$tmp/l2.aiff" "It did. The large model lost nothing on the twenty-five points. The small one dropped three."
say -v Daniel   -o "$tmp/l3.aiff" "Then large stays the default. What did it cost?"
say -v Samantha -o "$tmp/l4.aiff" "About five times the clock. Eleven minutes for a fourteen minute meeting, on the CPU."
say -v Daniel   -o "$tmp/l5.aiff" "Fine. Write it up with the numbers and I will read it after lunch."
say -v Samantha -o "$tmp/l6.aiff" "Will do."

ffmpeg -loglevel error -y -f lavfi -i anullsrc=r=16000:cl=mono -t 1.6 "$tmp/pause.wav"
for i in 1 2 3 4 5 6; do
  ffmpeg -loglevel error -y -i "$tmp/l$i.aiff" -ar 16000 -ac 1 "$tmp/l$i.wav"
done
{
  for i in 1 2 3 4 5 6; do
    printf "file '%s/l%s.wav'\n" "$tmp" "$i"
    [ "$i" -lt 6 ] && printf "file '%s/pause.wav'\n" "$tmp"
  done
} > "$tmp/list.txt"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$tmp/list.txt" -c:a aac -b:a 96k meeting.m4a
echo "meeting.m4a: $(ffprobe -v error -show_entries format=duration -of csv=p=0 meeting.m4a) s"
