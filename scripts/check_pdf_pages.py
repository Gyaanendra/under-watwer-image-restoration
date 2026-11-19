import fitz # PyMuPDF
doc = fitz.open('e:/sem5_project/reports/paper_ieee_5page.pdf')
print(f'Total pages: {len(doc)}')
for i, page in enumerate(doc):
    print(f'=== PAGE {i+1} ===')
    text = page.get_text().strip()
    first_lines = text.split('\n')[:3]
    last_lines = text.split('\n')[-3:]
    print('  TOP:', ' | '.join(first_lines))
    print('  BOTTOM:', ' | '.join(last_lines))
