#!/usr/bin/env fish
# Lossless PAL export from matching VHS-Decode luma/chroma TBCs and SQLite metadata.
# Usage: fish scripts/export-lossless.fish TBC_BASE [OUTPUT.mkv] [FRAMES]
# FIFO producers run as background jobs; do not replace them with fish psub -F.
if test (count $argv) -lt 1; or test (count $argv) -gt 3
    echo 'Usage: export-lossless.fish TBC_BASE [OUTPUT.mkv] [FRAMES]' >&2
    exit 1
end
set BASE (string replace -r '\.tbc$' '' -- "$argv[1]")
set OUTPUT "$BASE-lossless.mkv"
if test (count $argv) -ge 2
    set OUTPUT "$argv[2]"
end
set CHROMA /tmp/vhs-colour-check/build/bin/ld-chroma-decoder
if set -q VHS_CHROMA_DECODER
    set CHROMA "$VHS_CHROMA_DECODER"
else if type -q ld-chroma-decoder
    set CHROMA (command -s ld-chroma-decoder)
end
if not test -x "$CHROMA"
    echo 'Set VHS_CHROMA_DECODER to the ld-chroma-decoder executable.' >&2
    exit 1
end
if not type -q ffmpeg
    echo 'FFmpeg is required.' >&2
    exit 1
end
for INPUT in "$BASE.tbc" "$BASE"_chroma.tbc "$BASE.tbc.db"
    if not test -s "$INPUT"
        echo "Missing or empty input: $INPUT. SQLite metadata requires VHS-Decode --write_db." >&2
        exit 1
    end
end
if test -e "$OUTPUT"
    echo "Output already exists: $OUTPUT. Choose another name." >&2
    exit 1
end
set LENGTH_ARGS
if test (count $argv) -eq 3
    if not string match -qr '^[1-9][0-9]*$' -- "$argv[3]"
        echo 'FRAMES must be a positive integer.' >&2
        exit 1
    end
    set LENGTH_ARGS -l "$argv[3]"
end
# Validate both inputs before FFmpeg opens the FIFOs, avoiding a wait on a
# producer that rejected its metadata before it could open its output pipe.
"$CHROMA" -q -f mono -p y4m -l 1 --input-metadata "$BASE.tbc.db" "$BASE.tbc" /dev/null
or exit 1
"$CHROMA" -q -f pal2d -p y4m --luma-nr 0 -l 1 --input-metadata "$BASE.tbc.db" "$BASE"_chroma.tbc /dev/null
or exit 1
set PARENT (dirname -- "$OUTPUT")
mkdir -p -- "$PARENT"
or exit 1
set PIPES (mktemp -d "$PARENT/.vhs-export-XXXXXX")
or exit 1
set PRODUCERS
function cleanup_export --on-event fish_exit
    for PID in $PRODUCERS
        kill -TERM "$PID" 2>/dev/null
    end
    # Only remove FIFO names in the private directory created by this process.
    command rm -f -- "$PIPES/luma.y4m" "$PIPES/chroma.y4m"
    command rmdir -- "$PIPES" 2>/dev/null
end
function interrupt_export --on-signal INT
    exit 130
end
mkfifo "$PIPES/luma.y4m" "$PIPES/chroma.y4m"
or exit 1
"$CHROMA" -f mono -p y4m $LENGTH_ARGS --input-metadata "$BASE.tbc.db" "$BASE.tbc" "$PIPES/luma.y4m" &
set -a PRODUCERS $last_pid
"$CHROMA" -f pal2d -p y4m --luma-nr 0 $LENGTH_ARGS --input-metadata "$BASE.tbc.db" "$BASE"_chroma.tbc "$PIPES/chroma.y4m" &
set -a PRODUCERS $last_pid
ffmpeg -hide_banner -n \
    -thread_queue_size 64 -i "$PIPES/luma.y4m" \
    -thread_queue_size 64 -i "$PIPES/chroma.y4m" \
    -filter_complex '[0:v]format=yuv444p16le[y];[1:v]format=yuv444p16le[c];[y][c]mergeplanes=map0s=0:map0p=0:map1s=1:map1p=1:map2s=1:map2p=2:format=yuv444p16le,setdar=4/3,setfield=tff[out]' \
    -map '[out]' -c:v ffv1 -level 3 -g 1 -slices 16 -slicecrc 1 \
    -pix_fmt yuv444p16le -color_range tv -colorspace bt470bg "$OUTPUT"
set RESULT $status
if test $RESULT -ne 0
    exit $RESULT
end
wait $PRODUCERS
set -e PRODUCERS
echo "Lossless MKV saved: $OUTPUT"
