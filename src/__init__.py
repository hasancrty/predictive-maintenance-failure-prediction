import sys

# Windows konsolları varsayılan olarak cp1252 kullanır; Türkçe karakterler için UTF-8'e geç.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
