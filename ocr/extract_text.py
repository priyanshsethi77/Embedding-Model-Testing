import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = [doc[i].get_text() for i in range(len(doc))]
    doc.close()
    return pages_text  # List[str] => One string per page
