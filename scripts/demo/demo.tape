# Records docs/img/demo.gif: a real run of the CLI over the synthetic
# two-voice clip that make_audio.sh produces. Nothing is staged; what the
# terminal prints is what the GIF shows.
#
#   brew install vhs                      # ttyd and ffmpeg come with it
#   scripts/demo/make_audio.sh            # writes meeting.m4a (macOS text-to-speech)
#   vhs scripts/demo/demo.tape            # on a machine with the [diarize] extra
#
# The published GIF was recorded on a machine with a 4 GB GTX 980, so the probe
# took the whisper.cpp route; on a CPU-only machine the run takes longer and the
# GIF is longer with it. PlaybackSpeed compresses the wait, not the content.

Output docs/img/demo.gif

Set Shell "zsh"
Set FontFamily "Menlo"
Set FontSize 22
Set Width 1100
Set Height 600
Set Padding 28
Set Framerate 24
Set PlaybackSpeed 2.5
Set TypingSpeed 110ms
Set Theme { "name": "speechtotext", "background": "#1b1424", "foreground": "#f6f1f7", "cursor": "#ee8797", "selection": "#5a3690", "black": "#1b1424", "red": "#ee8797", "green": "#8fd3a6", "yellow": "#f3d0b5", "blue": "#a892d2", "magenta": "#dcb7cb", "cyan": "#9ad0d6", "white": "#f6f1f7", "brightBlack": "#7a6d8c", "brightRed": "#ee8797", "brightGreen": "#8fd3a6", "brightYellow": "#f3d0b5", "brightBlue": "#a892d2", "brightMagenta": "#dcb7cb", "brightCyan": "#9ad0d6", "brightWhite": "#ffffff" }

# A bare prompt: no user, no host name, no path in the recording.
Hide
Type "PROMPT='> '; rm -f meeting.txt meeting.srt meeting.json; clear"
Enter
Sleep 1s
Show

Sleep 800ms
Type "speechtotext transcribe meeting.m4a --diarize --speakers 2"
Sleep 700ms
Enter
Wait+Screen@600s /OK meeting.txt/
Sleep 1800ms
Type "cat meeting.txt"
Sleep 400ms
Enter
Sleep 5s
