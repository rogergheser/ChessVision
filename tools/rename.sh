for file in *" Background Removed".png; do
  mv "$file" "${file/ Background Removed/}"
done