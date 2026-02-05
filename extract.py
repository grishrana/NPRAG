import fitz
import unicodedata
import re

PDF_PATH = "files/gagan_info.pdf"
OUTPUT_TXT = "files/gagan_info.txt"


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text) # one or more spaces or tabs into single space
    text = re.sub(r"\n{3,}", "\n\n", text) # 3 or more newline into 2 newlines
    return text.strip()

doc = fitz.open(PDF_PATH)
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        
        text = clean_text(text)

        f.write("\n\n")
        f.write(text)

doc.close()
print(f"Saved text to: {OUTPUT_TXT}")

