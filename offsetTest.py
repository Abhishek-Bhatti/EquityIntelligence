def test_two_up_detection(pdf_path: str = "data/testing/testPDF.pdf"):
    """
    Standalone check for 2-up page detection. Anchors to the document's own
    minimum aspect ratio (the narrowest/tallest pages are assumed to be true
    single pages) rather than a hardcoded constant — self-calibrates across
    different trim sizes (A4, Letter, custom AR formats) instead of assuming
    one fixed portrait ratio applies everywhere.
    """
    import fitz

    TWO_UP_TOLERANCE = 0.15  # 15% band around exactly 2x the minimum ratio

    doc = fitz.open(pdf_path)
    try:
        aspects = [doc[i].rect.width / doc[i].rect.height for i in range(doc.page_count)]
        min_aspect = min(aspects)
        two_up_target = min_aspect * 2
        lower_bound = two_up_target * (1 - TWO_UP_TOLERANCE)
        upper_bound = two_up_target * (1 + TWO_UP_TOLERANCE)

        def is_two_up(page: fitz.Page) -> bool:
            rect = page.rect
            aspect = rect.width / rect.height
            if not (lower_bound <= aspect <= upper_bound):
                return False
            return True

        page_map: dict[int, list] = {}
        two_up_indices = []

        for idx in range(doc.page_count):
            page = doc[idx]
            result = is_two_up(page)
            if result is True:
                page_map[idx] = [(idx, "L"), (idx, "R")]
                two_up_indices.append(idx)
            else:
                page_map[idx] = [idx]

        print(f"Document: {pdf_path}")
        print(f"Total pages: {doc.page_count}")
        print(f"Minimum aspect ratio (assumed single-page baseline): {min_aspect:.3f}")
        print(f"2-up target range: {lower_bound:.3f} - {upper_bound:.3f}")
        print(f"2-up pages detected: {len(two_up_indices)}")
        if two_up_indices:
            print(f"  Indices: {two_up_indices}")
        print()

        print(f"{'PDF index':<12}{'Aspect':<10}{'Virtual keys':<20}")
        print("-" * 42)
        for idx in range(doc.page_count):
            aspect = aspects[idx]
            flag = " <-- 2-up" if idx in two_up_indices else ""
            print(f"{idx:<12}{aspect:<10.3f}{str(page_map[idx]):<20}{flag}")

        return page_map

    finally:
        doc.close()


if __name__ == "__main__":
    test_two_up_detection()