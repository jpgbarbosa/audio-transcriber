#!/bin/bash
#
# Example batch processing script
# Process all audio files in a directory
#

INPUT_DIR="${1:-./recordings}"
OUTPUT_DIR="${2:-./transcripts}"
LANGUAGE="${3:-auto}"
MODEL="${4:-large-v3}"

echo "═══ Batch Transcription ═══"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Language: $LANGUAGE"
echo "Model: $MODEL"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Counter
count=0

# Process each audio file
for file in "$INPUT_DIR"/*.{mp3,wav,m4a,flac}; do
    # Skip if no files match
    [ -e "$file" ] || continue

    count=$((count + 1))
    echo "[$count] Processing: $(basename "$file")"

    python transcribe.py "$file" \
        --language "$LANGUAGE" \
        --model "$MODEL" \
        --output "$OUTPUT_DIR" \
        --format all

    echo ""
done

echo "═══ Complete ═══"
echo "Processed $count file(s)"
echo "Results saved to: $OUTPUT_DIR"
